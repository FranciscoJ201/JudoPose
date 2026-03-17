import pyrealsense2 as rs
import numpy as np
import cv2
import time
import threading
import queue
import sys
import os
import csv
import json

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

W, H       = 848, 480
TARGET_FPS = 60

# Output paths
COLOR_VIDEO_PATH  = 'realsense_color.mp4'
DEPTH_FOLDER      = 'realsense_depth_frames'   # One 16-bit PNG per frame saved here
TIMESTAMP_CSV     = 'realsense_timestamps.csv'
INTRINSICS_JSON   = 'realsense_intrinsics.json'  # fx, fy, cx, cy + image size

# How long to run the hardware before we start saving anything.
# RealSense hardware sync is unstable for the first few seconds after start.
WARMUP_SECONDS = 5


# ─────────────────────────────────────────────
#  DEPTH WRITER THREAD
#
#  Saving 16-bit PNGs at 60fps would block the main capture loop if done
#  synchronously. This thread pulls (frame_index, depth_array) tuples off a
#  queue and writes them in the background so the pipeline never stalls.
# ─────────────────────────────────────────────

def depth_writer_thread(depth_queue, output_folder, stop_flag):
    """
    Consumes depth frames from the queue and saves them as lossless 16-bit PNGs.

    Each pixel value is the raw depth in millimetres as reported by the RealSense
    infrared sensor. This preserves full sub-centimetre precision — no lossy
    compression is applied.

    After recording, sample these maps at your YOLO keypoint (u, v) coordinates
    to get Z in metres:
        z_metres = depth_png[v, u] / 1000.0

    Arguments:
    - depth_queue:   Queue of (frame_index, numpy uint16 array) tuples.
    - output_folder: Directory to write depth_XXXXXX.png files.
    - stop_flag:     threading.Event — set when recording ends.
    """
    while not stop_flag.is_set() or not depth_queue.empty():
        try:
            frame_idx, depth_arr = depth_queue.get(timeout=0.1)
            filename = os.path.join(output_folder, f'depth_{frame_idx:06d}.png')
            cv2.imwrite(filename, depth_arr)
            depth_queue.task_done()
        except queue.Empty:
            continue


# ─────────────────────────────────────────────
#  COLOR WRITER THREAD
# ─────────────────────────────────────────────

def color_writer_thread(frame_queue, writer, stop_flag):
    """
    Consumes BGR color frames from the queue and encodes them to the MP4 file.

    Arguments:
    - frame_queue: Queue of numpy BGR arrays.
    - writer:      cv2.VideoWriter instance.
    - stop_flag:   threading.Event — set when recording ends.
    """
    while not stop_flag.is_set() or not frame_queue.empty():
        try:
            frame = frame_queue.get(timeout=0.1)
            writer.write(frame)
            frame_queue.task_done()
        except queue.Empty:
            continue


# ─────────────────────────────────────────────
#  TIMESTAMP HELPERS
# ─────────────────────────────────────────────

def get_frame_timestamps(color_frame):
    """
    Extracts three timestamp values from a RealSense color frame.

    - sensor_timestamp_ms:
        The hardware clock on the RealSense sensor itself, in milliseconds.
        This is the most precise timing source — it reflects when the exposure
        actually occurred, independent of USB transfer delays.

    - time_of_arrival_ms:
        When the frame physically arrived at your PC (host clock, milliseconds).
        This is what you feed into the linear regression alongside sensor_timestamp
        to compute a unified global timeline.

    - host_time_s:
        Python's time.time() at the moment we call this function. Used as a
        coarse cross-reference and for the regression offline step.

    Arguments:
    - color_frame: A pyrealsense2 video_frame object.

    Returns:
    - Tuple of (sensor_timestamp_ms, time_of_arrival_ms, host_time_s).
    """
    sensor_ts  = color_frame.get_timestamp()          # ms, hardware clock
    arrival_ts = color_frame.get_frame_metadata(
        rs.frame_metadata_value.time_of_arrival)      # ms, host clock via SDK
    host_ts    = time.time()                           # s, Python host clock

    return sensor_ts, arrival_ts, host_ts


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    os.makedirs(DEPTH_FOLDER, exist_ok=True)

    # ── Pipeline setup ──────────────────────────────────────────────────────
    pipeline = rs.pipeline()
    config   = rs.config()

    # Color stream: BGR8 at full target resolution and FPS
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8,   TARGET_FPS)

    # Depth stream: Z16 (millimetres, uint16) at the same resolution and FPS.
    # Aligning depth to color ensures every depth pixel maps 1:1 to its color
    # counterpart, so sampling depth at a keypoint (u, v) is trivial.
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16,    TARGET_FPS)

    print(f"Starting RealSense pipeline at {W}x{H} @ {TARGET_FPS} FPS...")
    profile = pipeline.start(config)

    # ── Intrinsics — extracted once immediately after pipeline starts ────────
    # Done here, before the preview loop, so it has zero impact on recording
    # throughput. These values are needed in post to project depth keypoints
    # into 3D:  X = (u - cx) * Z / fx,  Y = (v - cy) * Z / fy
    color_intr = (profile
                  .get_stream(rs.stream.color)
                  .as_video_stream_profile()
                  .get_intrinsics())

    intrinsics = {
        "fx": color_intr.fx,
        "fy": color_intr.fy,
        "cx": color_intr.ppx,          # principal point x
        "cy": color_intr.ppy,          # principal point y
        "width":  color_intr.width,
        "height": color_intr.height,
        "distortion_model": str(color_intr.model),
        "distortion_coeffs": list(color_intr.coeffs),  # k1,k2,p1,p2,k3
    }

    with open(INTRINSICS_JSON, 'w') as f:
        json.dump(intrinsics, f, indent=4)
    print(f"Intrinsics saved to {INTRINSICS_JSON}  "
          f"(fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}, "
          f"cx={intrinsics['cx']:.2f}, cy={intrinsics['cy']:.2f})")

    # Align depth frames to the color frame viewport.
    # After alignment, depth_frame.get_data()[v, u] gives the depth in mm at
    # color pixel (u, v) — no separate intrinsic transform needed.
    align = rs.align(rs.stream.color)

    # ── Video writer ─────────────────────────────────────────────────────────
    fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
    color_writer = cv2.VideoWriter(COLOR_VIDEO_PATH, fourcc, TARGET_FPS, (W, H))

    # ── Queues and threads ───────────────────────────────────────────────────
    color_queue = queue.Queue(maxsize=300)   # ~5s buffer at 60fps
    depth_queue = queue.Queue(maxsize=300)
    stop_flag   = threading.Event()

    t_color = threading.Thread(
        target=color_writer_thread,
        args=(color_queue, color_writer, stop_flag),
        daemon=True
    )
    t_depth = threading.Thread(
        target=depth_writer_thread,
        args=(depth_queue, DEPTH_FOLDER, stop_flag),
        daemon=True
    )
    t_color.start()
    t_depth.start()

    # ── CSV setup ────────────────────────────────────────────────────────────
    csv_file   = open(TIMESTAMP_CSV, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'frame_index',
        'sensor_timestamp_ms',    # hardware clock — use for regression
        'time_of_arrival_ms',     # host-side arrival — use for regression
        'host_time_s',            # coarse Python clock cross-reference
    ])

    # ── Preview loop ─────────────────────────────────────────────────────────
    print("Showing preview. Press 'c' to arm recording, or 'q' to quit.")
    while True:
        frames       = pipeline.wait_for_frames()
        aligned      = align.process(frames)
        color_frame  = aligned.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow("Preview — press 'c' to record, 'q' to quit", color_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            cv2.destroyAllWindows()
            break
        elif key == ord('q'):
            stop_flag.set()
            t_color.join(); t_depth.join()
            pipeline.stop(); color_writer.release(); csv_file.close()
            sys.exit(0)

    # ── Warm-up period ───────────────────────────────────────────────────────
    # Consume and discard frames for WARMUP_SECONDS.
    # The RealSense hardware sync (inter-frame timing) is unstable immediately
    # after pipeline.start(). Discarding the first N seconds ensures the
    # timestamps you save are drawn from the stable, converged regime.
    print(f"Warming up for {WARMUP_SECONDS} seconds (discarding frames)...")
    warmup_end = time.time() + WARMUP_SECONDS
    while time.time() < warmup_end:
        pipeline.wait_for_frames()
        remaining = warmup_end - time.time()
        print(f"  {remaining:.1f}s remaining...", end='\r')
    print("\nWarm-up complete. Recording started. Press Ctrl+C to stop.")

    # ── Main recording loop ───────────────────────────────────────────────────
    frame_count = 0
    start_time  = time.time()

    try:
        while True:
            frames  = pipeline.wait_for_frames()
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # ── Timestamps ──────────────────────────────────────────────────
            sensor_ts, arrival_ts, host_ts = get_frame_timestamps(color_frame)
            csv_writer.writerow([frame_count, sensor_ts, arrival_ts, host_ts])

            # ── Color → queue ────────────────────────────────────────────────
            color_image = np.asanyarray(color_frame.get_data())
            try:
                color_queue.put(color_image, block=False)
            except queue.Full:
                print(f"Warning: color queue full at frame {frame_count} — dropping frame.")

            # ── Depth → queue ────────────────────────────────────────────────
            # get_data() returns a uint16 array where each value = depth in mm.
            # We pass (frame_index, array) so the depth thread names the file correctly.
            depth_image = np.asanyarray(depth_frame.get_data())   # uint16, mm
            try:
                depth_queue.put((frame_count, depth_image), block=False)
            except queue.Full:
                print(f"Warning: depth queue full at frame {frame_count} — dropping depth frame.")

            frame_count += 1

            if frame_count % (TARGET_FPS * 10) == 0:
                elapsed = time.time() - start_time
                print(f"  {frame_count} frames recorded ({elapsed:.1f}s, "
                      f"{frame_count/elapsed:.1f} fps actual)")

    except KeyboardInterrupt:
        print("\nCtrl+C — stopping recording cleanly...")

    finally:
        # Signal writer threads to drain their queues, then wait for them
        stop_flag.set()
        t_color.join()
        t_depth.join()

        elapsed     = time.time() - start_time
        actual_fps  = frame_count / elapsed if elapsed > 0 else 0

        pipeline.stop()
        color_writer.release()
        csv_file.close()
        cv2.destroyAllWindows()

        print(f"\nDone!")
        print(f"  Color video : {COLOR_VIDEO_PATH}")
        print(f"  Depth frames: {DEPTH_FOLDER}/  ({frame_count} PNGs)")
        print(f"  Timestamps  : {TIMESTAMP_CSV}")
        print(f"  Intrinsics  : {INTRINSICS_JSON}")
        print(f"  Frames saved: {frame_count} @ {actual_fps:.2f} fps actual")
        print(f"\nNext steps:")
        print(f"  1. Run YOLO pose estimation on {COLOR_VIDEO_PATH} to get keypoint (u,v) per frame.")
        print(f"  2. For each keypoint, sample depth:  z_m = depth_XXXXXX.png[v, u] / 1000.0")
        print(f"  3. Feed sensor_timestamp_ms + time_of_arrival_ms from {TIMESTAMP_CSV}")
        print(f"     into a linear regression (numpy.polyfit) to build a unified global clock.")
        print(f"  4. Use that global clock to align RealSense frames with GoPro SD card footage.")


if __name__ == "__main__":
    main()