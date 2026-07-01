import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ================= USER SETTINGS =================

LEADER_PORT = "COM9"
FOLLOWER_PORT = "COM8"

LEADER_ID = "so101_leader_1"
FOLLOWER_ID = "so101_follower_1"

# Camera indices
WRIST_INDEX = 1
ZED_INDEX = 2

# Wrist camera
WRIST_WIDTH = 640
WRIST_HEIGHT = 480
WRIST_FPS = 30

# ZED stereo stream. 1344x376 is common ZED VGA side-by-side mode.
# Each eye becomes 672x376 after crop.
ZED_WIDTH = 1344
ZED_HEIGHT = 376
ZED_FPS = 30

ZED_SIDE = "left"  # "left" or "right"

# Streaming encoding reduces the pause after pressing Right Arrow.
STREAMING_ENCODING = True
ENCODER_THREADS = 1
VIDEO_CODEC = "libx264"

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_ROOT = PROJECT_ROOT / "so101_recordings"
LEROBOT_DATA_ROOT = DATA_ROOT / "lerobot_home"
EXPORT_ROOT = DATA_ROOT / "exports"


# ================= BASIC HELPERS =================

def clean_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
    return name or datetime.now().strftime("run_%Y%m%d_%H%M%S")


def default_run_name() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def repo_id(name: str) -> str:
    return f"local/{clean_name(name)}"


def dataset_dir(name: str) -> Path:
    return LEROBOT_DATA_ROOT / "local" / clean_name(name)


def default_cache_dataset_dir(name: str) -> Path:
    return Path.home() / ".cache" / "huggingface" / "lerobot" / "local" / clean_name(name)


def make_env() -> dict:
    """
    Do NOT set HF_LEROBOT_HOME here.
    Calibration stays in the default LeRobot cache.
    Dataset location is controlled with --dataset.root.
    """
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    return env


def find_cli(name: str) -> str:
    scripts_dir = Path(sys.executable).resolve().parent

    candidates = [
        scripts_dir / f"{name}.exe",
        scripts_dir / name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    found = shutil.which(name)
    if found:
        return found

    raise FileNotFoundError(
        f"Could not find {name}. Make sure your venv is activated and LeRobot is installed."
    )


def run_cmd(cmd: list[str]) -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    LEROBOT_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

    print("\nRunning:\n")
    print(" ".join(f'"{x}"' if " " in x else x for x in cmd))

    print("\nProject dataset folder:")
    print(LEROBOT_DATA_ROOT)

    print("\nCalibration stays in default LeRobot cache.\n")

    subprocess.run(cmd, env=make_env(), check=True)


# ================= CAMERA CONFIG =================

def camera_config(mode: str = "both") -> str | None:
    """
    mode:
      none  = no cameras
      wrist = wrist camera only
      zed   = ZED overhead only
      both  = wrist + ZED
    """
    if mode == "none":
        return None

    if mode == "wrist":
        return (
            "{ "
            f"wrist: {{type: opencv, index_or_path: {WRIST_INDEX}, "
            f"width: {WRIST_WIDTH}, height: {WRIST_HEIGHT}, fps: {WRIST_FPS}}}"
            " }"
        )

    if mode == "zed":
        return (
            "{ "
            f"overhead_zed: {{type: opencv, index_or_path: {ZED_INDEX}, "
            f"width: {ZED_WIDTH}, height: {ZED_HEIGHT}, fps: {ZED_FPS}}}"
            " }"
        )

    if mode == "both":
        return (
            "{ "
            f"wrist: {{type: opencv, index_or_path: {WRIST_INDEX}, "
            f"width: {WRIST_WIDTH}, height: {WRIST_HEIGHT}, fps: {WRIST_FPS}}}, "
            f"overhead_zed: {{type: opencv, index_or_path: {ZED_INDEX}, "
            f"width: {ZED_WIDTH}, height: {ZED_HEIGHT}, fps: {ZED_FPS}}}"
            " }"
        )

    raise ValueError(f"Unknown camera mode: {mode}")


# ================= ROBOT ACTIONS =================

def teleop() -> None:
    cmd = [
        find_cli("lerobot-teleoperate"),

        "--teleop.type=so101_leader",
        f"--teleop.port={LEADER_PORT}",
        f"--teleop.id={LEADER_ID}",

        "--robot.type=so101_follower",
        f"--robot.port={FOLLOWER_PORT}",
        f"--robot.id={FOLLOWER_ID}",

        "--display_data=false",
    ]

    run_cmd(cmd)


def record(
    name: str,
    task: str,
    episodes: int,
    episode_time: int,
    reset_time: int,
    camera_mode: str,
) -> None:
    """
    Records LeRobot training data.

    Same --name automatically overwrites the old dataset.
    Streaming encoding is enabled to reduce wait time between episodes.
    """
    name = clean_name(name)

    dset_path = dataset_dir(name)
    old_cache_path = default_cache_dataset_dir(name)

    for path in [dset_path, old_cache_path]:
        if path.exists():
            print(f"Deleting old dataset: {path}")
            shutil.rmtree(path)

    cmd = [
        find_cli("lerobot-record"),

        "--teleop.type=so101_leader",
        f"--teleop.port={LEADER_PORT}",
        f"--teleop.id={LEADER_ID}",

        "--robot.type=so101_follower",
        f"--robot.port={FOLLOWER_PORT}",
        f"--robot.id={FOLLOWER_ID}",

        "--display_data=false",

        f"--dataset.root={str(dset_path)}",
        f"--dataset.repo_id={repo_id(name)}",
        f"--dataset.num_episodes={episodes}",
        f"--dataset.single_task={task}",
        f"--dataset.episode_time_s={episode_time}",
        f"--dataset.reset_time_s={reset_time}",
        "--dataset.push_to_hub=False",
    ]

    if STREAMING_ENCODING:
        cmd += [
            "--dataset.streaming_encoding=true",
            f"--dataset.encoder_threads={ENCODER_THREADS}",
            f"--dataset.vcodec={VIDEO_CODEC}",
        ]

    cam_cfg = camera_config(camera_mode)
    if cam_cfg is not None:
        cmd.append(f"--robot.cameras={cam_cfg}")

    run_cmd(cmd)

    print("\nSaved LeRobot training dataset:")
    print(dset_path)

    print("\nExporting readable servo CSV and videos...")
    export(name=name, zed_side=ZED_SIDE)


def replay(name: str, episode: int) -> None:
    name = clean_name(name)
    dset_path = dataset_dir(name)

    if not dset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dset_path}")

    cmd = [
        find_cli("lerobot-replay"),

        "--robot.type=so101_follower",
        f"--robot.port={FOLLOWER_PORT}",
        f"--robot.id={FOLLOWER_ID}",

        f"--dataset.root={str(dset_path)}",
        f"--dataset.repo_id={repo_id(name)}",
        f"--dataset.episode={episode}",
    ]

    run_cmd(cmd)


def list_runs() -> None:
    root = LEROBOT_DATA_ROOT / "local"

    if not root.exists():
        print("No recordings yet.")
        return

    runs = sorted([p for p in root.iterdir() if p.is_dir()])

    if not runs:
        print("No recordings yet.")
        return

    print("\nRecordings:\n")
    for run in runs:
        print(f"  {run.name}")


# ================= CAMERA PREVIEW =================

def preview_cameras() -> None:
    import cv2

    wrist = cv2.VideoCapture(WRIST_INDEX, cv2.CAP_DSHOW)
    zed = cv2.VideoCapture(ZED_INDEX, cv2.CAP_DSHOW)

    wrist.set(cv2.CAP_PROP_FRAME_WIDTH, WRIST_WIDTH)
    wrist.set(cv2.CAP_PROP_FRAME_HEIGHT, WRIST_HEIGHT)
    wrist.set(cv2.CAP_PROP_FPS, WRIST_FPS)

    zed.set(cv2.CAP_PROP_FRAME_WIDTH, ZED_WIDTH)
    zed.set(cv2.CAP_PROP_FRAME_HEIGHT, ZED_HEIGHT)
    zed.set(cv2.CAP_PROP_FPS, ZED_FPS)

    if not wrist.isOpened():
        raise RuntimeError(f"Could not open wrist camera index {WRIST_INDEX}")

    if not zed.isOpened():
        raise RuntimeError(f"Could not open ZED camera index {ZED_INDEX}")

    print("Previewing wrist and cropped ZED. Press q to quit.")

    while True:
        ret_w, frame_w = wrist.read()
        ret_z, frame_z = zed.read()

        if ret_w:
            cv2.imshow("wrist camera", frame_w)

        if ret_z:
            h, w = frame_z.shape[:2]
            half = w // 2

            if ZED_SIDE == "left":
                zed_crop = frame_z[:, :half]
            else:
                zed_crop = frame_z[:, half:]

            cv2.imshow(f"overhead ZED {ZED_SIDE} eye", zed_crop)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    wrist.release()
    zed.release()
    cv2.destroyAllWindows()


# ================= EXPORT HELPERS =================

def crop_zed_video(src: Path, dst: Path, side: str) -> None:
    import cv2

    cap = cv2.VideoCapture(str(src))

    if not cap.isOpened():
        print(f"Could not open video: {src}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = ZED_FPS

    ret, frame = cap.read()
    if not ret:
        print(f"Could not read first frame: {src}")
        cap.release()
        return

    h, w = frame.shape[:2]
    half = w // 2

    if side == "left":
        cropped = frame[:, :half]
    else:
        cropped = frame[:, half:]

    out_h, out_w = cropped.shape[:2]

    dst.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (out_w, out_h))

    writer.write(cropped)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        half = w // 2

        if side == "left":
            cropped = frame[:, :half]
        else:
            cropped = frame[:, half:]

        writer.write(cropped)

    writer.release()
    cap.release()


def export(name: str, zed_side: str = ZED_SIDE) -> None:
    """
    Exports:
    - one merged servo/action/state CSV
    - wrist videos
    - cropped single-eye ZED videos

    Raw side-by-side ZED videos are not copied into the clean review folder.
    """
    name = clean_name(name)
    dset = dataset_dir(name)

    if not dset.exists():
        print(f"Dataset not found: {dset}")
        return

    export_dir = EXPORT_ROOT / name

    if export_dir.exists():
        shutil.rmtree(export_dir)

    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDataset folder:\n{dset}")
    print(f"\nExport folder:\n{export_dir}")

    # ---------- CSV export ----------
    try:
        import pandas as pd
    except ImportError:
        print("\nMissing pandas/pyarrow. Install with:")
        print("python -m pip install pandas pyarrow")
        pd = None

    if pd is not None:
        parquet_files = sorted(dset.rglob("*.parquet"))

        data_parquets = [
            p for p in parquet_files
            if "data" in [part.lower() for part in p.parts]
        ]

        if not data_parquets:
            data_parquets = parquet_files

        frames = []

        for p in data_parquets:
            try:
                df = pd.read_parquet(p)
                df["_source_file"] = str(p.relative_to(dset))
                frames.append(df)
            except Exception as e:
                print(f"Could not read parquet {p}: {e}")

        if frames:
            df = pd.concat(frames, ignore_index=True)

            keep_keywords = [
                "timestamp",
                "episode",
                "episode_index",
                "frame",
                "frame_index",
                "index",
                "action",
                "state",
                "position",
                "observation",
                "task",
            ]

            keep_cols = [
                c for c in df.columns
                if any(k in c.lower() for k in keep_keywords)
            ]

            if "_source_file" not in keep_cols:
                keep_cols.append("_source_file")

            out = df[keep_cols].copy()

            expanded = {}
            drop_cols = []

            for col in list(out.columns):
                non_null = out[col].dropna()

                if non_null.empty:
                    continue

                first = non_null.iloc[0]

                if isinstance(first, (str, bytes, dict)):
                    continue

                try:
                    if hasattr(first, "__len__") and len(first) > 1:
                        values = out[col].apply(
                            lambda x: list(x)
                            if hasattr(x, "__len__") and not isinstance(x, (str, bytes, dict))
                            else []
                        )

                        max_len = max((len(v) for v in values), default=0)

                        if max_len > 1:
                            for i in range(max_len):
                                expanded[f"{col}.{i}"] = values.apply(
                                    lambda v: v[i] if i < len(v) else None
                                )

                            drop_cols.append(col)

                except Exception:
                    pass

            for col in drop_cols:
                out = out.drop(columns=[col])

            for col, values in expanded.items():
                out[col] = values

            csv_path = export_dir / f"{name}_servo_data.csv"
            out.to_csv(csv_path, index=False)

            print("\nServo/action/state CSV saved:")
            print(csv_path)

            episode_cols = [c for c in out.columns if "episode" in c.lower()]
            if episode_cols:
                ep_col = episode_cols[0]
                try:
                    episodes = sorted(out[ep_col].dropna().unique().tolist())
                    print(f"\nEpisodes found in CSV: {episodes}")
                    print(f"Episode count: {len(episodes)}")
                except Exception:
                    pass

        else:
            print("\nNo readable parquet files found.")

    # ---------- Clean video export ----------
    videos = sorted(dset.rglob("*.mp4"))

    if not videos:
        print("\nNo MP4 videos found.")
        return

    videos_dir = export_dir / "videos"
    wrist_dir = videos_dir / "wrist"
    zed_dir = videos_dir / f"overhead_zed_{zed_side}_eye"

    wrist_dir.mkdir(parents=True, exist_ok=True)
    zed_dir.mkdir(parents=True, exist_ok=True)

    wrist_count = 0
    zed_count = 0
    other_count = 0

    print("\nExporting videos...")

    for video in videos:
        video_str = str(video).lower()

        if "wrist" in video_str:
            dst = wrist_dir / video.name
            if dst.exists():
                dst = wrist_dir / f"{video.parent.name}_{video.name}"

            shutil.copy2(video, dst)
            wrist_count += 1
            continue

        if "overhead_zed" in video_str or "zed" in video_str:
            dst = zed_dir / video.name
            if dst.exists():
                dst = zed_dir / f"{video.parent.name}_{video.name}"

            crop_zed_video(video, dst, zed_side)
            zed_count += 1
            continue

        other_count += 1

    print(f"\nWrist videos exported: {wrist_count}")
    print(f"ZED {zed_side}-eye videos exported: {zed_count}")

    if other_count:
        print(f"Other raw videos ignored: {other_count}")

    print("\nClean review folder:")
    print(videos_dir)


def review(name: str, zed_side: str = ZED_SIDE) -> None:
    export(name=name, zed_side=zed_side)

    export_dir = EXPORT_ROOT / clean_name(name)

    print("\nReview folder:")
    print(export_dir)

    videos = sorted(export_dir.rglob("*.mp4"))

    if not videos:
        print("\nNo videos found.")
        return

    print("\nVideos found:")
    for video in videos:
        print(f"  {video}")

    os.startfile(export_dir)


def stats(name: str) -> None:
    """
    Shows how many episodes, rows, and video files were saved.
    """
    name = clean_name(name)
    export_dir = EXPORT_ROOT / name
    csv_path = export_dir / f"{name}_servo_data.csv"

    if not csv_path.exists():
        print("CSV not found. Exporting first...")
        export(name=name, zed_side=ZED_SIDE)

    if not csv_path.exists():
        print(f"Still no CSV found: {csv_path}")
        return

    import pandas as pd

    df = pd.read_csv(csv_path)

    print("\nCSV:")
    print(csv_path)
    print(f"Total rows: {len(df)}")
    print(f"CSV size MB: {csv_path.stat().st_size / 1_000_000:.2f}")

    episode_cols = [c for c in df.columns if "episode" in c.lower()]
    frame_cols = [c for c in df.columns if "frame" in c.lower()]
    timestamp_cols = [c for c in df.columns if "timestamp" in c.lower()]

    if episode_cols:
        ep_col = episode_cols[0]
        episodes = sorted(df[ep_col].dropna().unique().tolist())
        print(f"\nEpisode column: {ep_col}")
        print(f"Episodes saved: {episodes}")
        print(f"Episode count: {len(episodes)}")

        print("\nRows per episode:")
        print(df.groupby(ep_col).size())

        if frame_cols:
            fr_col = frame_cols[0]
            print(f"\nFrame stats per episode using {fr_col}:")
            print(df.groupby(ep_col)[fr_col].agg(["min", "max", "count"]))

        if timestamp_cols:
            ts_col = timestamp_cols[0]
            print(f"\nTimestamp stats per episode using {ts_col}:")
            print(df.groupby(ep_col)[ts_col].agg(["min", "max", "count"]))

    print("\nExported videos:")
    videos = sorted(export_dir.rglob("*.mp4"))

    if not videos:
        print("No exported videos found.")
    else:
        total = 0
        for video in videos:
            size = video.stat().st_size / 1_000_000
            total += size
            print(f"{size:8.2f} MB  {video}")
        print(f"\nTotal video size: {total:.2f} MB")


# ================= CLI =================

def main() -> None:
    parser = argparse.ArgumentParser(description="SO-101 two-camera record/replay tool")

    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("preview", help="Preview wrist camera and cropped ZED camera")
    sub.add_parser("teleop", help="Teleoperate leader COM9 -> follower COM8, no cameras")
    sub.add_parser("list", help="List saved recordings")

    rec = sub.add_parser("record", help="Record a new LeRobot training dataset")
    rec.add_argument("--name", default=default_run_name())
    rec.add_argument("--task", default="pick up the object and place it down")
    rec.add_argument("--episodes", type=int, default=1)
    rec.add_argument("--episode-time", type=int, default=3600)
    rec.add_argument("--reset-time", type=int, default=0)
    rec.add_argument(
        "--camera-mode",
        choices=["none", "wrist", "zed", "both"],
        default="both",
        help="Choose camera setup: none, wrist, zed, or both",
    )

    rep = sub.add_parser("replay", help="Replay a saved episode on the follower arm")
    rep.add_argument("--name", required=True)
    rep.add_argument("--episode", type=int, default=0)

    exp = sub.add_parser("export", help="Export servo CSV and clean review videos")
    exp.add_argument("--name", required=True)
    exp.add_argument("--zed-side", choices=["left", "right"], default=ZED_SIDE)

    rev = sub.add_parser("review", help="Export and open review folder")
    rev.add_argument("--name", required=True)
    rev.add_argument("--zed-side", choices=["left", "right"], default=ZED_SIDE)

    st = sub.add_parser("stats", help="Show saved episodes, rows, and video sizes")
    st.add_argument("--name", required=True)

    args = parser.parse_args()

    if args.cmd == "preview":
        preview_cameras()

    elif args.cmd == "teleop":
        teleop()

    elif args.cmd == "record":
        record(
            name=args.name,
            task=args.task,
            episodes=args.episodes,
            episode_time=args.episode_time,
            reset_time=args.reset_time,
            camera_mode=args.camera_mode,
        )

    elif args.cmd == "replay":
        replay(
            name=args.name,
            episode=args.episode,
        )

    elif args.cmd == "export":
        export(
            name=args.name,
            zed_side=args.zed_side,
        )

    elif args.cmd == "review":
        review(
            name=args.name,
            zed_side=args.zed_side,
        )

    elif args.cmd == "stats":
        stats(name=args.name)

    elif args.cmd == "list":
        list_runs()


if __name__ == "__main__":
    main()