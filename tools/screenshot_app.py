"""
Capture screenshots of the running Streamlit app for the report and slides by
driving headless Chrome over the DevTools protocol (no extra dependencies:
uses tornado, which Streamlit already installs).

    streamlit run app/app.py --server.port 8599 &
    python tools/screenshot_app.py --port 8599

Writes reports/figures/app_*.png.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from tornado.ioloop import IOLoop
from tornado.websocket import websocket_connect

ROOT = Path(__file__).resolve().parent.parent
CHROME_CANDIDATES = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                     r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", "google-chrome", "chromium"]

SHOTS = [
    ("app_1_reentrancy_result", "?sample=02_reentrancy_bank.sol&scan=1", 0, 1500),
    ("app_2_highlighted_code", "?sample=02_reentrancy_bank.sol&scan=1", 1150, 1500),
    ("app_3_clean_token", "?sample=01_clean_token.sol&scan=1", 0, 1300),
]


async def _shoot(ws_url: str, url: str, out: Path, scroll: int, height: int, wait: float) -> None:
    conn = await websocket_connect(ws_url, max_message_size=200 * 1024 * 1024)
    msg_id = 0

    async def call(method, **params):
        nonlocal msg_id
        msg_id += 1
        my = msg_id
        conn.write_message(json.dumps({"id": my, "method": method, "params": params}))
        while True:
            reply = json.loads(await conn.read_message())
            if reply.get("id") == my:
                return reply.get("result", {})

    await call("Emulation.setDeviceMetricsOverride", width=1400, height=height, deviceScaleFactor=1, mobile=False)
    await call("Page.enable")
    await call("Page.navigate", url=url)
    time.sleep(wait)
    await call("Runtime.evaluate", expression=(
        "(() => { const m = document.querySelector('section.main, [data-testid=\"stMain\"]');"
        f" if (m) m.scrollTop = {scroll}; window.scrollTo(0, {scroll}); }})()"))
    time.sleep(1.5)
    shot = await call("Page.captureScreenshot", format="png")
    out.write_bytes(base64.b64decode(shot["data"]))
    conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8599)
    ap.add_argument("--wait", type=float, default=25.0)
    a = ap.parse_args()
    chrome = next((c for c in CHROME_CANDIDATES if Path(c).exists()), CHROME_CANDIDATES[-1])
    profile = tempfile.mkdtemp()
    proc = subprocess.Popen([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                             "--remote-debugging-port=9333", f"--user-data-dir={profile}", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                pages = json.loads(urllib.request.urlopen("http://127.0.0.1:9333/json").read())
                break
            except OSError:
                time.sleep(0.3)
        ws = next(p["webSocketDebuggerUrl"] for p in pages if p.get("type") == "page")
        figs = ROOT / "reports" / "figures"
        figs.mkdir(parents=True, exist_ok=True)
        for name, query, scroll, height in SHOTS:
            out = figs / f"{name}.png"
            IOLoop.current().run_sync(lambda: _shoot(ws, f"http://localhost:{a.port}/{query}", out, scroll,
                                                     height, a.wait))
            print("wrote", out)
    finally:
        proc.terminate()


if __name__ == "__main__":
    sys.exit(main())
