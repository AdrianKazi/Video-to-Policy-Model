from src.pretraining.config import ensure_dirs, lunar_video_paths
from src.pretraining.control_points import build_control_points, save_control_points


def main():
    ensure_dirs()
    paths = lunar_video_paths()
    df = build_control_points(paths)
    output_path = save_control_points(df)
    print("saved:", output_path)
    print("shape:", df.shape)


if __name__ == "__main__":
    main()

