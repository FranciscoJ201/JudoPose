import asyncio
from bleak import BleakClient,BleakScanner

# --- 1. Your Specific Camera Details ---
# Replace these with the actual Bluetooth MAC addresses of your GoPros
GOPRO_1_MAC = "A35956DE-0D2C-5E73-1A08-5A7EA914F073"
# GOPRO_2_MAC = "YY:YY:YY:YY:YY:YY"

# --- 2. The GoPro Bluetooth API Constants ---
# This is the official UUID the GoPro listens to for shutter commands
COMMAND_UUID = "b5f90072-aa8d-11e3-9046-0002a5d5c51b"

# The raw hexadecimal bytes that tell the camera to start and stop
SHUTTER_START = bytearray([0x03, 0x01, 0x01, 0x01])
SHUTTER_STOP = bytearray([0x03, 0x01, 0x01, 0x00])

async def trigger_camera(mac_address, command, cam_name):
    """
    Scans for the camera to wake up the macOS Bluetooth cache, 
    then connects and sends the hex command.
    
    Arguments:
    - mac_address: The Apple UUID string of the GoPro.
    - command: The bytearray hex command to start/stop.
    - cam_name: String label for console output.
    """
    try:
        print(f"[{cam_name}] Pre-scanning to wake up macOS cache...")
        # Force the Mac to find the device object first
        device = await BleakScanner.find_device_by_address(mac_address, timeout=5.0)
        
        if not device:
            print(f"[{cam_name}] Failed: Could not find GoPro. Is it awake and disconnected from your phone?")
            return

        print(f"[{cam_name}] Found it! Connecting...")
        
        # Pass the actual device object to the client, not just the string
        async with BleakClient(device) as client:
            print(f"[{cam_name}] Connected! Sending command...")
            await client.write_gatt_char(COMMAND_UUID, command, response=True)
            print(f"[{cam_name}] Command successful.")
            
    except Exception as e:
        print(f"[{cam_name}] Failed: {e}")

async def main():
    print("--- Sending Global Sync Trigger ---")
    
    # asyncio.gather is the magic trick here. 
    # It runs both trigger functions concurrently, at the exact same time.
    await asyncio.gather(
        trigger_camera(GOPRO_1_MAC, SHUTTER_START, "GoPro A"),
        # trigger_camera(GOPRO_2_MAC, SHUTTER_START, "GoPro B")
    )
    
    print("\n>>> BOTH CAMERAS RECORDING! Execute throw! <<<\n")

    # Let the cameras record the action for 6 seconds
    await asyncio.sleep(6)

    print("--- Stopping Cameras ---")
    await asyncio.gather(
        trigger_camera(GOPRO_1_MAC, SHUTTER_STOP, "GoPro A"),
        # trigger_camera(GOPRO_2_MAC, SHUTTER_STOP, "GoPro B")
    )
    print("\nData collection complete.")

if __name__ == "__main__":
    # Start the asynchronous event loop
    asyncio.run(main())