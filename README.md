# found-time

A clock for [Rücksicht](https://github.com/shepardxia/ruecksicht). Every minute
a 1.2B model writes two sentences of Japanese; the hour is in the first, the
minute in the second, and nothing else in them is a number.

![](docs/desktop.png)

<p align="center"><img src="docs/clock.png" width="720"></p>

<p align="center">11:42 — <i>The spring wind cast eleven shadows on the river. / The pebbles left at
the edge of the mountain path, about forty-two.</i></p>

## Install

```sh
brew tap shepardxia/ruecksicht https://github.com/shepardxia/ruecksicht
brew install ruecksicht
brew services start ruecksicht
rk add https://github.com/shepardxia/found-time
```

Needs `llamppl` with the `mlx` extra on the Python that `tick.sh` names; the
model, `LiquidAI/LFM2.5-1.2B-JP-202606-MLX-4bit`, downloads on the first
minute.

## Files

`gen.py` writes one minute. `clockd.py` keeps the model loaded and writes this
minute and the next to `clock.json`; below 25% battery it serves a minute it
has seen before from `store.json`. `tick.sh` keeps `clockd` running.
`index.jsx` draws it.
