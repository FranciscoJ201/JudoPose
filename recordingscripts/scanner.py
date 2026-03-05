import asyncio
from bleak import BleakScanner

async def scan_for_gopros():
    """
    Scans the room for active Bluetooth Low Energy devices.
    
    Arguments for BleakScanner.discover():
    - timeout (float): The duration in seconds to listen for devices (we use 10.0 to ensure cameras have time to broadcast).
    - return_adv (bool): Set to True if you also want to return the raw advertising data payloads.
    """
    print("Scanning for Bluetooth devices... (Make sure GoPros are awake!)")
    
    # Run the scanner
    devices = await BleakScanner.discover(timeout=10.0)
    
    for device in devices:
        # GoPros usually broadcast a name like "GoPro XXXX"
        if device.name and "GoPro" in device.name:
            print(f"Found {device.name}: MAC Address is {device.address}")

if __name__ == "__main__":
    asyncio.run(scan_for_gopros())