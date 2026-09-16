"""Keeps the model resident and writes clock.json: the passage for this minute
and the next, so the widget can switch at the boundary without waiting on a
generation. Wakes once a minute, just past the boundary. Every line also goes into store.json by minute of day; below 25%
battery a minute the store already knows is served from there instead of
being generated.

Started by tick.sh and stays up while the widget keeps ticking; exits on its
own once the ticks stop.
"""
import asyncio, datetime, json, os, random, subprocess, sys, time
from pathlib import Path
import mlx.core as mx
from gen import Clock

HERE = Path(__file__).resolve().parent
OUT, STORE, HEARTBEAT = HERE / "clock.json", HERE / "store.json", HERE / ".tick"
KEEP = 8                    # lines remembered per minute of day
LOW_BATTERY = 25            # percent
IDLE_EXIT = 300             # seconds without a tick before exiting

def battery():
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    pct = next((int(t.rstrip("%;")) for t in out.split() if t.endswith("%;")), 100)
    return pct, "discharging" in out

def write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)

def load(path):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return {}

async def line_for(clock, store, when):
    key = when.strftime("%H:%M")
    pct, on_battery = battery()
    if on_battery and pct < LOW_BATTERY and store.get(key):
        return random.choice(store[key]), False
    line = await clock.generate(when)
    if "error" not in line:
        store[key] = (store.get(key, []) + [line])[-KEEP:]
    return line, True

async def main():
    clock = Clock()
    store = load(STORE)
    lines = {}
    while True:
        now = datetime.datetime.now().replace(second=0, microsecond=0)
        wanted = [now, now + datetime.timedelta(minutes=1)]
        keys = {w.strftime("%H:%M") for w in wanted}
        lines = {k: v for k, v in lines.items() if k in keys}
        for when in wanted:
            key = when.strftime("%H:%M")
            if key not in lines:
                lines[key], generated = await line_for(clock, store, when)
                write(OUT, {"lines": lines})
                if generated:
                    write(STORE, store)
                    mx.clear_cache()
        if time.time() - HEARTBEAT.stat().st_mtime > IDLE_EXIT if HEARTBEAT.exists() else True:
            return
        # Sleep to just past the boundary, then write the minute after it.
        now = datetime.datetime.now()
        await asyncio.sleep(60 - now.second - now.microsecond / 1e6 + 1)

if __name__ == "__main__":
    asyncio.run(main())
