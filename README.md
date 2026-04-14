# npm-sim

`npm-sim` is a tiny Newton-on-Warp rigid-body demo repo containing exactly one passive 3D scene: a ball rolls down a ramp, crosses a floor, and hits a 3-cube tower.

## Install

```bash
python -m pip install -e .
```

## Interactive Run

```bash
python scripts/run_ramp_tower.py --viewer gl
```

## USD Export

```bash
python scripts/run_ramp_tower.py --viewer usd --output-path outputs/ramp_tower.usd
```

## MP4 Video Export

```bash
python scripts/render_ramp_tower_video.py --output-path outputs/ramp_tower.mp4
```

## Headless Run

```bash
python scripts/run_ramp_tower.py --viewer null --num-frames 180
```

The repo currently contains only this single demo.
