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

# Leave as empty list to auto-detect all connected RealSense devices.
# Or specify serial numbers explicitly to control which camera is cam_0
# and which is cam_1, e.g: ['123456789', '987654321']
# Serial numbers are printed on the camera label and shown at startup.
SERIAL_NUMBERS = []

WARMUP_SECONDS = 5

# ─────────────────────────────────────────────
#  PER-CAMERA OUTPUT STRUCTURE
#
#  Each camera gets its own isolated output folder:
#    realsense_cam_0/
#        color.mp4
#        depth_frames/       ← 16-bit PNGs, one per frame
#        timestamps.csv
#        intrinsics.json
#    realsense_cam_1/
#        color.mp4
#        depth_frames/
#        timestamps.csv
#        intrinsics.json
#
#  Per-camera folders avoid any filename collision and make it trivial
#  to pass each camera's outputs to the rest of the pipeline.
# ─────────────────────────────────────────────

def make_camera_output_dir(cam_label):
    """Creates and returns the output directory for one camera."""
    out_dir = f'realsense_{cam_label}'
    os.makedirs(os.path.join(out_dir, 'depth_frames'), exist_ok=True)
    return out_dir


# ─────────────────────────────────────────────
#  WRITER THREADS
#  Each camera gets its own independent instances of both threads
#  so they never contend with each other on queues or file handles.
# ─────────────────────────────────────────────

def depth_writer_thread(depth_queue, output_folder, stop_flag):
    """
    Consumes (frame_index, uint16 depth array) tuples and saves them
    as lossless 16-bit PNGs. Each pixel value = depth in millimetres.

    Sample in post:  z_m = depth_XXXXXX.png[v, u] / 1000.0
    """
    while not stop_flag.is_set() or not depth_queue.empty():
        try:
            frame_idx, depth_arr = depth_queue.get(timeout=0.1)
            path = os.path.join(output_folder, f'depth_{frame_idx:06d}.png')
            cv2.imwrite(path, depth_arr)
            depth_queue.task_done()
        except queue.Empty:
            continue


def color_writer_thread(frame_queue, writer, stop_flag):
    """Consumes BGR numpy arrays and encodes them to the MP4 file."""
    while not stop_flag.is_set() or not frame_queue.empty():
        try:
            frame = frame_queue.get(timeout=0.1)
            writer.write(frame)
            frame_queue.task_done()
        except queue.Empty:
            continue


# ─────────────────────────────────────────────
#  TIMESTAMP HELPER
# ─────────────────────────────────────────────

def get_frame_timestamps(color_frame):
    """
    Returns (sensor_timestamp_ms, time_of_arrival_ms, host_time_s).

    sensor_timestamp_ms — hardware clock at moment of exposure. Most precise.
                          Each camera has its own independent hardware clock
                          so values are not directly comparable between cameras.
    time_of_arrival_ms  — when the frame reached the PC via USB (SDK clock).
                          Both cameras share the same host clock so these
                          values ARE comparable and are what the offline
                          linear regression uses to align the two cameras.
    host_time_s         — Python time.time() coarse cross-reference.
    """
    sensor_ts  = color_frame.get_timestamp()
    arrival_ts = color_frame.get_frame_metadata(
        rs.frame_metadata_value.time_of_arrival)
    host_ts    = time.time()
    return sensor_ts, arrival_ts, host_ts


# ─────────────────────────────────────────────
#  CAMERA WORKER
#
#  Each RealSense runs in its own thread with fully independent:
#    - rs.pipeline + rs.config (targeted to its serial number)
#    - rs.align instance
#    - color queue + writer thread
#    - depth queue + writer thread
#    - CSV file handle
#    - local stop flag for its writer threads
#
#  The two cameras share only global_stop (set by Ctrl+C in main)
#  and ready_event (to signal warmup completion to main thread).
# ─────────────────────────────────────────────

def camera_worker(serial, cam_label, ready_event, global_stop):
    """
    Full recording lifecycle for one RealSense camera.

    Arguments:
    - serial:       Device serial number string.
    - cam_label:    Human-readable label e.g. 'cam_0'.
    - ready_event:  threading.Event set when warmup is complete and this
                    camera is actively recording. Main thread waits on all.
    - global_stop:  threading.Event set by main thread on Ctrl+C to stop
                    all cameras simultaneously.
    """
    out_dir    = make_camera_output_dir(cam_label)
    depth_dir  = os.path.join(out_dir, 'depth_frames')
    video_path = os.path.join(out_dir, 'color.mp4')
    csv_path   = os.path.join(out_dir, 'timestamps.csv')
    intr_path  = os.path.join(out_dir, 'intrinsics.json')

    print(f"  [{cam_label}] Initialising pipeline (serial: {serial})...")

    # ── Pipeline targeted to this specific serial number ─────────────────────
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, TARGET_FPS)
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16,  TARGET_FPS)

    profile = pipeline.start(config)
    align   = rs.align(rs.stream.color)

    # ── Intrinsics — saved immediately after pipeline starts ─────────────────
    # Zero cost to recording — done once before warmup begins.
    color_intr = (profile
                  .get_stream(rs.stream.color)
                  .as_video_stream_profile()
                  .get_intrinsics())

    intrinsics = {
        "serial":            serial,
        "cam_label":         cam_label,
        "fx":                color_intr.fx,
        "fy":                color_intr.fy,
        "cx":                color_intr.ppx,
        "cy":                color_intr.ppy,
        "width":             color_intr.width,
        "height":            color_intr.height,
        "distortion_model":  str(color_intr.model),
        "distortion_coeffs": list(color_intr.coeffs),  # k1,k2,p1,p2,k3
    }
    with open(intr_path, 'w') as f:
        json.dump(intrinsics, f, indent=4)
    print(f"  [{cam_label}] Intrinsics saved → {intr_path} "
          f"(fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f})")

    # ── Video writer ──────────────────────────────────────────────────────────
    fourcc       = cv2.VideoWriter_fourcc(*'mp4v')
    color_writer = cv2.VideoWriter(video_path, fourcc, TARGET_FPS, (W, H))

    # ── Per-camera queues and writer threads ──────────────────────────────────
    color_queue = queue.Queue(maxsize=300)   # ~5s buffer at 60fps
    depth_queue = queue.Queue(maxsize=300)
    local_stop  = threading.Event()          # signals this camera's writers

    t_color = threading.Thread(
        target=color_writer_thread,
        args=(color_queue, color_writer, local_stop),
        daemon=True
    )
    t_depth = threading.Thread(
        target=depth_writer_thread,
        args=(depth_queue, depth_dir, local_stop),
        daemon=True
    )
    t_color.start()
    t_depth.start()

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_file   = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'frame_index',
        'sensor_timestamp_ms',   # hardware clock — use for per-camera regression
        'time_of_arrival_ms',    # host clock — use for cross-camera alignment
        'host_time_s',           # coarse Python clock cross-reference
    ])

    # ── Warm-up — discard frames until hardware timing stabilises ─────────────
    print(f"  [{cam_label}] Warming up for {WARMUP_SECONDS}s...")
    warmup_end = time.time() + WARMUP_SECONDS
    while time.time() < warmup_end:
        pipeline.wait_for_frames()
    print(f"  [{cam_label}] Warm-up complete — recording started.")

    # Signal the main thread that this camera is past warmup
    ready_event.set()

    # ── Main recording loop ───────────────────────────────────────────────────
    frame_count = 0
    start_time  = time.time()

    try:
        while not global_stop.is_set():
            frames  = pipeline.wait_for_frames()
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # Timestamps
            sensor_ts, arrival_ts, host_ts = get_frame_timestamps(color_frame)
            csv_writer.writerow([frame_count, sensor_ts, arrival_ts, host_ts])

            # Color → queue
            color_image = np.asanyarray(color_frame.get_data())
            try:
                color_queue.put(color_image, block=False)
            except queue.Full:
                print(f"  [{cam_label}] Warning: color queue full at frame "
                      f"{frame_count} — dropping frame.")

            # Depth → queue
            # uint16 array, values in millimetres
            depth_image = np.asanyarray(depth_frame.get_data())
            try:
                depth_queue.put((frame_count, depth_image), block=False)
            except queue.Full:
                print(f"  [{cam_label}] Warning: depth queue full at frame "
                      f"{frame_count} — dropping depth frame.")

            frame_count += 1

            if frame_count % (TARGET_FPS * 10) == 0:
                elapsed = time.time() - start_time
                print(f"  [{cam_label}] {frame_count} frames "
                      f"({elapsed:.1f}s, {frame_count/elapsed:.1f} fps actual)")

    finally:
        # Drain both writer threads before closing file handles
        local_stop.set()
        t_color.join()
        t_depth.join()

        elapsed    = time.time() - start_time
        actual_fps = frame_count / elapsed if elapsed > 0 else 0

        pipeline.stop()
        color_writer.release()
        csv_file.close()

        print(f"\n  [{cam_label}] Finished — "
              f"{frame_count} frames @ {actual_fps:.2f} fps actual")
        print(f"    Color video  : {video_path}")
        print(f"    Depth frames : {depth_dir}/")
        print(f"    Timestamps   : {csv_path}")
        print(f"    Intrinsics   : {intr_path}")


# ─────────────────────────────────────────────
#  DEVICE DISCOVERY
# ─────────────────────────────────────────────

def discover_devices():
    """
    Finds all connected RealSense devices and returns their serial numbers.
    Prints a summary so the operator can confirm the right cameras are found.
    """
    ctx     = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        raise RuntimeError(
            "No RealSense devices found. Check USB connections.")

    serials = []
    print(f"\nDetected {len(devices)} RealSense device(s):")
    for i, dev in enumerate(devices):
        serial = dev.get_info(rs.camera_info.serial_number)
        name   = dev.get_info(rs.camera_info.name)
        print(f"  [{i}] {name}  serial: {serial}")
        serials.append(serial)

    return serials


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():

    # ── Discover or use configured serials ───────────────────────────────────
    if SERIAL_NUMBERS:
        serials = SERIAL_NUMBERS
        print(f"\nUsing configured serial numbers: {serials}")
    else:
        serials = discover_devices()

    if len(serials) == 0:
        print("No cameras to record. Exiting.")
        sys.exit(1)

    if len(serials) > 2:
        print(f"Warning: {len(serials)} cameras detected — "
              f"only 2 recommended on a single host. Using first 2.")
        serials = serials[:2]

    cam_labels = [f'cam_{i}' for i in range(len(serials))]

    # ── Preview — all cameras side by side before committing to recording ─────
    # Lightweight preview pipelines open first so the operator can verify
    # framing. They are stopped and released before the recording workers start.
    print(f"\nOpening preview for {len(serials)} camera(s).")
    print("Press 'c' to start recording, 'q' to quit.")

    preview_pipelines = []
    for serial, label in zip(serials, cam_labels):
        p   = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, TARGET_FPS)
        p.start(cfg)
        preview_pipelines.append((label, p))

    while True:
        imgs = []
        for label, p in preview_pipelines:
            frames = p.wait_for_frames()
            cf     = frames.get_color_frame()
            if cf:
                img = np.asanyarray(cf.get_data())
                cv2.putText(img, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                imgs.append(img)

        if imgs:
            cv2.imshow("Preview — 'c' record, 'q' quit", np.hstack(imgs))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            cv2.destroyAllWindows()
            break
        elif key == ord('q'):
            for _, p in preview_pipelines:
                p.stop()
            sys.exit(0)

    # Stop preview pipelines and give USB a moment to settle
    for _, p in preview_pipelines:
        p.stop()
    time.sleep(1.0)

    # ── Launch one recording thread per camera ────────────────────────────────
    global_stop  = threading.Event()
    ready_events = [threading.Event() for _ in serials]
    cam_threads  = []

    for serial, label, ready_ev in zip(serials, cam_labels, ready_events):
        t = threading.Thread(
            target=camera_worker,
            args=(serial, label, ready_ev, global_stop),
            daemon=True
        )
        t.start()
        cam_threads.append(t)

    # ── Wait until ALL cameras have cleared warmup ────────────────────────────
    print(f"\nWaiting for all {len(serials)} cameras to complete warm-up...")
    for ev, label in zip(ready_events, cam_labels):
        ev.wait()
        print(f"  {label} — ready")

    print("\nAll cameras recording simultaneously.")
    print("Press Ctrl+C to stop all cameras at once.\n")

    # ── Hold main thread until Ctrl+C ─────────────────────────────────────────
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nCtrl+C — stopping all cameras simultaneously...")
        global_stop.set()

    for t in cam_threads:
        t.join()

    print("\nAll cameras stopped cleanly.")
    print("Output folders:")
    for label in cam_labels:
        print(f"  realsense_{label}/")
    print("\nNext steps:")
    print("  1. Run YOLO pose estimation on each realsense_cam_X/color.mp4")
    print("  2. Sample realsense_cam_X/depth_frames/ at YOLO keypoints for Z")
    print("  3. Run offline linear regression on each timestamps.csv")
    print("     using time_of_arrival_ms to align the two cameras' timelines")


if __name__ == "__main__":
    main()