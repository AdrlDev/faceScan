# handle_scan.py

from scan import scan_live

def handle_scan(result):
    print(">>> FACE DETECTED:", result)

scan_live(callback=handle_scan)
