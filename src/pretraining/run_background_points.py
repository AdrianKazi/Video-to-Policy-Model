from src.pretraining.background import build_background_points_per_episode
from src.pretraining.config import ensure_dirs, lunar_video_paths


def main():
    ensure_dirs()
    paths = lunar_video_paths()
    output_paths = build_background_points_per_episode(paths)
    print("saved episode files:", len(output_paths))


if __name__ == "__main__":
    main()

