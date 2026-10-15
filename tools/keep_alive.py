"""
Visit the Streamlit app in a real headless browser so it counts as traffic.
If the app is asleep, click the wake-up button and wait until it has loaded.

Environment:
    APP_URL      full app address, e.g. https://skincancerimaging.streamlit.app
    APP_MARKER   text that appears on a loaded app (default: "Image Upload")
Exit code 1 means the app did not load, so GitHub will mark the run as failed.
"""
import os
import sys
import time

from playwright.sync_api import sync_playwright

url = os.environ.get("APP_URL", "").strip()
marker = os.environ.get("APP_MARKER", "Image Upload")

if not url:
    sys.exit("APP_URL is not set (Settings > Secrets and variables > Actions > Variables).")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(url, wait_until="domcontentloaded", timeout=120_000)
    page.wait_for_timeout(8_000)

    try:
        page.get_by_role("button", name="Yes, get this app back up").click(timeout=10_000)
        print("App was asleep: clicked the wake-up button.")
    except Exception:
        print("No wake-up button (app already awake).")

    deadline = time.time() + 240
    loaded = False
    while time.time() < deadline and not loaded:
        for frame in page.frames:
            try:
                if frame.get_by_text(marker).count() > 0:
                    loaded = True
                    break
            except Exception:
                pass
        if not loaded:
            page.wait_for_timeout(5_000)

    browser.close()

if loaded:
    print(f"OK: app loaded ('{marker}' found).")
else:
    sys.exit(f"FAILED: '{marker}' not found within 4 minutes.")
