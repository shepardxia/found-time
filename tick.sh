#!/bin/bash
# Prints clock.json -- this minute's passage and the next -- and keeps clockd
# alive: one resident process, started here when none is running, that exits
# on its own once these ticks stop.
cd "$(dirname "$0")"
touch .tick
if ! kill -0 "$(cat clockd.pid 2>/dev/null)" 2>/dev/null; then
    PYTHONPATH=$HOME/Desktop/directory/genlm/mlx/genlm-backend-modernize:$HOME/Desktop/directory/genlm/mlx/llamppl-modernize \
    nohup $HOME/Desktop/directory/genlm/mlx/genlm-control/.venv/bin/python clockd.py >/dev/null 2>clockd.log &
    echo $! > clockd.pid
fi
cat clock.json 2>/dev/null
