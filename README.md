# PitchWise — Vision & Game State

[![CI](https://github.com/arturbuszka/PitchWise.Lab/actions/workflows/ci.yml/badge.svg)](https://github.com/arturbuszka/PitchWise.Lab/actions/workflows/ci.yml)

From a raw football broadcast (1080p, main camera), this .NET pipeline reconstructs the *game state*:
every person's position in pitch coordinates, plus their role (player, goalkeeper or referee), team
and jersey number.

It is a .NET port of the SoccerNet
[Game State Reconstruction](https://www.soccer-net.org/tasks/game-state-reconstruction) baseline
([sn-gamestate](https://github.com/SoccerNet/sn-gamestate) +
[TrackLab](https://github.com/TrackingLaboratory/tracklab)), which is written in Python. Here the
neural networks run through ONNX Runtime with no Python at runtime, and the tracker is my own
[StrongSort.Net](https://github.com/arturbuszka/StrongSort.NET) package.

[PitchWise.Platform](https://github.com/arturbuszka/PitchWise.Platform) is the related backend, which
handles live ingest, uploads, highlight reels and HLS delivery. It has no vision code of its own and
isn't connected to this pipeline yet. The planned link is the `analyze` command below, run as a
separate process.

Licensed under **GPL-3.0** — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Results

GS-HOTA on 10 SoccerNet-GSR `valid` sequences that were held out from tuning, scored with a .NET port
of the official metric:

| Variant | GS-HOTA |
|---|---:|
| Position, role, team and number | 31.5% |
| Without jersey numbers | 47.2% |
| Position and identity only | 57.7% |

Jersey numbers cost the most: the OCR weights are a substitute for the original baseline's, because
its export toolchain needs a much older Python and PyTorch. Measurements, diagnoses and experiments
are in [docs/eval-baseline.md](docs/eval-baseline.md) (in Polish).

## Quickstart

Needs the .NET 10 SDK.

```bash
dotnet build PitchWise.slnx
dotnet test PitchWise.slnx        # tests that need the dataset and models report as skipped

tools/models/download.sh          # about 750 MB, into pretrained_models/

# Run the pipeline on the first 100 frames of any video, on the CPU.
dotnet run --project src/PitchWise.Cli -c Release -- -c configs/cpu.yaml track match.mp4 out.mp4 pretrained_models 100
```

CPU runs take seconds per frame. See [docs/MODELS.md](docs/MODELS.md) for GPU setup, the dataset,
and where each model comes from.

## Structure

- `src/PitchWise.Tracking` — tracking-framework core (port of `tracklab`)
- `src/PitchWise.GameState` — football modules (port of `sn_gamestate`) + `PipelineConfig`
- `src/PitchWise.Engine` — dependency-free football rule/world-state engine
- `src/PitchWise.Cli` — command-line entry point (equivalent of `tracklab -cn soccernet`)
- `src/PitchWise.App.Core` — desktop view-models, workspace and playback (no UI framework)
- `src/PitchWise.App` — Avalonia desktop app

`PitchWise.Cli` and `PitchWise.App` are peers: both are composition roots over the same libraries,
neither references the other, and the pipeline they assemble comes from the one `PipelineConfig` in
`PitchWise.GameState`.

How each part maps to the original Python code is in [docs/PORT_STATUS.md](docs/PORT_STATUS.md).

## Commands

```
# full pipeline on any video (requires ONNX models in pretrained_models/):
dotnet run --project src/PitchWise.Cli -- track match.mp4 out.mp4 [models_dir] [max_frames] [start_seconds]
# pipeline + match statistics as JSON:
dotnet run --project src/PitchWise.Cli -- stats match.mp4 out.json [models_dir] [max_frames] [start_seconds]
# pipeline + GS-HOTA metric vs dataset ground truth:
dotnet run --project src/PitchWise.Cli -- eval <SoccerNetGS_path> valid SNGS-021 [models_dir] [max_frames]
# render ground truth to mp4 (ellipses + minimap):
dotnet run --project src/PitchWise.Cli -- render <SoccerNetGS_path> valid SNGS-021 out.mp4
# for other programs, see "The analyze command":
dotnet run --project src/PitchWise.Cli -- analyze match.mp4 out_dir [models_dir] [max_frames]
```

Module parameters are configured in `configs/soccernet.yaml` (the Hydra equivalent; use the `-c other.yaml`
flag to switch). Modules with `enabled: auto` turn on when their ONNX models are present in `models_dir`.

`track` assembles the pipeline from the modules whose models are available: `yolo11m.onnx` (required) →
`prtreid-soccernet-baseline.onnx` (teams / roles) → StrongSORT → `nbjw_kp/lines.onnx`
(homography + minimap) → `mmocr_det/rec.onnx` (numbers).

## The `analyze` command

This is the entry point for programs that run the pipeline as a separate process:

```bash
dotnet run --project src/PitchWise.Cli -c Release -- analyze match.mp4 out_dir pretrained_models 200
```

`models_dir` and `max_frames` are optional, and a `max_frames` of 0 means the whole video. It writes
three files into `out_dir`:

| File | Contents |
|---|---|
| `result.json` | Statistics and events. Written last, so its presence means the run finished. |
| `world_state.jsonl` | A header line, then what the pipeline saw in each frame |
| `render.mp4` | The video with the overlay, in MPEG-4 Part 2, which browsers generally don't play |

Progress goes to stdout as lines like `PROGRESS 0.42`, mixed with log lines, so read only the lines
that start with `PROGRESS`. A failure gives a non-zero exit code. A missing video or missing models
currently end as an unhandled exception (exit code 134), with a stack trace and no `result.json`.

Example `result.json`, shortened to one player:

```json
{
  "schema_version": 1,
  "fps": 25,
  "stats": {
    "frames_processed": 200,
    "duration_seconds": 7.96,
    "controlled_seconds": 2.52,
    "loose_seconds": 5.44,
    "team_a": { "possession_pct": 100, "passes": 0, "turnovers": 0, "pass_accuracy_pct": 0 },
    "team_b": { "possession_pct": 0, "passes": 0, "turnovers": 0, "pass_accuracy_pct": 0 },
    "players": [
      {
        "player_id": -15,
        "track_id": 13,
        "team": "B",
        "distance_meters": 33.5,
        "top_speed_kmh": 42.7,
        "avg_speed_kmh": 19.7,
        "sprints": 0,
        "time_on_pitch_seconds": 7.6
      }
    ]
  },
  "events": []
}
```

Each item in `events` has `type`, `timestamp_seconds`, `confidence`, `label`, `player_id` and `team`.
Passes and possession changes are counted in `stats`, not listed as events, so a short clip often has
none. A breaking change to this file raises `schema_version`.

In `world_state.jsonl`, each frame line has the frame number `f`, the time `t` in seconds from the
start of the video, and the `ball` and players `p`, in metres from the centre of the pitch. Both have
a confidence `c`. Players also have an `id`, a role `r` (0 player, 1 goalkeeper, 2 referee) and a team
`tm` (0 unknown, 1 A, 2 B). A negative `id` is a short-lived tracker identity, which doesn't survive a
re-track.

## Develop

### Debug the CLI on a few frames

The whole pipeline takes seconds per frame on a CPU, so debug on a few. A synthetic test video
has no people in it, but it exercises the whole path from decoding to `result.json`:

```bash
mkdir -p data/debug
ffmpeg -f lavfi -i testsrc2=size=1920x1080:rate=25:d=2 -c:v libx264 -pix_fmt yuv420p data/debug/testsrc2.mp4
dotnet run --project src/PitchWise.Cli -- -c configs/cpu.yaml analyze data/debug/testsrc2.mp4 data/debug/out pretrained_models 5
```

To check detection or tracking, use a video with players in it. Everything in `data/` is ignored by git.

In VS Code, with the C# extension, start **CLI: analyze 5 frames (CPU)** from the Run and Debug view.
It builds the code and creates the test video first.

### GPU and the CUDA libraries

Install the libraries as described in [docs/MODELS.md](docs/MODELS.md#run-on-a-gpu). In a terminal,
set the library path and leave out `-c configs/cpu.yaml`:

```bash
export LD_LIBRARY_PATH="$(ls -d ~/cuda12-libs/lib/python3*/site-packages/nvidia/*/lib | paste -sd:)"
dotnet run --project src/PitchWise.Cli -- analyze data/debug/testsrc2.mp4 data/debug/out pretrained_models 5
```

The log shows `running on CUDA (device 0)` for each model, or `falling back to CPU` with the reason.

In VS Code, run the task **write CUDA env file** once. It saves the path to `.vscode/cuda.env`, which
is ignored by git and read by **CLI: analyze 5 frames (GPU)** and **Desktop app (GPU)**.

### Tests

```bash
dotnet test PitchWise.slnx
```

Tests that need the SoccerNet frames and ONNX models report as skipped when those are missing.

## Desktop app

An internal tool for watching, debugging and tuning the analysis: queue a video, watch it run, then
browse the result.

```
dotnet run --project src/PitchWise.App
```

Pick a source, queue it, and open the job once it finishes:

- **Frame sequence** — a folder of numbered images, used in place (no copy). This is what the
  SoccerNet data is: `data/SoccerNetGS/valid/SNGS-021/img1`.
- **Video file** — frames are extracted into the job's own folder.
- **SoccerNet video** — a video folder with `Labels-GameState.json`; also loads ground truth, which
  enables the comparison radar.

Four panels over one analysis: the video with the overlay and radar (each layer toggleable), an event
timeline whose markers seek to the frame, match statistics, and a debug/tuning tab with the raw
detections, a tracklet inspector, the job log, and sliders over every `EngineConfig` threshold that
re-derive the timeline and stats in milliseconds.

Jobs live in `PitchWise/workspaces` under the local application data folder (`%LOCALAPPDATA%` on Windows, `~/.local/share` on Linux), one self-describing folder each (`manifest.json`
plus artifacts), so discovery is a directory scan and a job survives being copied elsewhere.

Two things worth knowing:

- **Playback reads the frames on disk, not the video.** Seeking an H.264 file lands on the nearest
  keyframe, and since the overlay is keyed by `image_id`, an approximate seek would draw the wrong
  players. Deleting a job's frames reclaims the disk and falls back to the video, which the player
  labels as approximate.
- **Analysis runs one job at a time**, because `PipelineConfig.BuildPipeline` sets the static ONNX
  provider. The GPU is the bottleneck anyway.
