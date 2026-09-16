"""Keeps the model resident and writes clock.json: the passage for this minute
and the next, so the widget can switch at the boundary without waiting on a
generation. Wakes once a minute, just past the boundary. Every line also goes
into store.jsonl by minute of day; below 25% battery a minute the store already
knows is served from there instead of being generated.

Started by tick.sh and stays up while the widget keeps ticking; exits on its
own once the ticks stop.
"""
import asyncio, datetime, json, os, random, subprocess, time
from collections import defaultdict
from pathlib import Path
import mlx.core as mx
from gen import Clock, minute_key

HERE = Path(__file__).resolve().parent
OUT, STORE, HEARTBEAT = HERE / "clock.json", HERE / "store.jsonl", HERE / ".tick"
KEEP = 8                    # lines remembered per minute of day
LOW_BATTERY = 25            # percent
IDLE_EXIT = 300             # seconds without a tick before exiting

def on_low_battery():
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    pct = next((int(t.rstrip("%;")) for t in out.split() if t.endswith("%;")), 100)
    return "discharging" in out and pct < LOW_BATTERY

def load_store():
    store = defaultdict(list)
    if STORE.exists():
        for raw in STORE.read_text(encoding="utf-8").splitlines():
            line = json.loads(raw)
            store[line["time"]] = (store[line["time"]] + [line])[-KEEP:]
    return store

async def line_for(clock, store, when):
    known = store[minute_key(when)]
    if known and on_low_battery():
        return random.choice(known)
    line = await clock.generate(when)
    if line:
        known.append(line)
        del known[:-KEEP]
        with STORE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        mx.clear_cache()
    return line

def write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)

async def main():
    clock = Clock()
    store = load_store()
    lines = {}
    while True:
        now = datetime.datetime.now().replace(second=0, microsecond=0)
        wanted = [minute_key(now + datetime.timedelta(minutes=i)) for i in (0, 1)]
        lines = {k: v for k, v in lines.items() if k in wanted}
        for i, key in enumerate(wanted):
            if key not in lines:
                line = await line_for(clock, store, now + datetime.timedelta(minutes=i))
                if line:
                    lines[key] = line
                    write(OUT, {"lines": lines})
        if time.time() - HEARTBEAT.stat().st_mtime > IDLE_EXIT:
            return
        # Sleep to just past the boundary, then write the minute after it.
        now = datetime.datetime.now()
        await asyncio.sleep(60 - now.second - now.microsecond / 1e6 + 1)

if __name__ == "__main__":
    asyncio.run(main())
