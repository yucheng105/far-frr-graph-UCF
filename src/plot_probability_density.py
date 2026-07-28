from pathlib import Path
import csv

import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

DATASETS = [
    ("CelebDF", "threshold_sweep_ucf_celebdf.csv"),
    ("DFDC", "threshold_sweep_ucf_dfdc.csv"),
    ("FaceForensics", "threshold_sweep_ucf_faceforensics.csv"),
    ("Combined", "threshold_sweep_ucf_combined.csv"),
]


def get_split_dirs():
    split_dirs = [
        path for path in RESULTS_DIR.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]

    if not split_dirs:
        raise FileNotFoundError(f"No numeric split directories found in {RESULTS_DIR}")

    return sorted(split_dirs, key=lambda path: int(path.name))


def read_threshold_sweep(csv_path):
    thresholds = []
    fars = []
    frrs = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {"Threshold", "FAR", "FRR"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"{csv_path} is missing required columns: "
                f"{', '.join(sorted(missing_columns))}"
            )

        for row in reader:
            thresholds.append(float(row["Threshold"]))
            fars.append(float(row["FAR"]))
            frrs.append(float(row["FRR"]))

    if not thresholds:
        raise ValueError(f"{csv_path} does not contain any threshold rows")

    return thresholds, fars, frrs


def setup_axes(title):
    plt.figure(figsize=(9, 5.5))
    plt.title(title)
    plt.xlabel("Threshold")
    plt.ylabel("Probability")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.35)


def save_current_figure(output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Wrote {output_path}")


def plot_single_chart(dataset_name, split_name, csv_path):
    thresholds, fars, frrs = read_threshold_sweep(csv_path)

    setup_axes(f"FAR / FRR Threshold Sweep - {dataset_name} ({split_name} splits)")
    plt.plot(thresholds, fars, label="FAR", linewidth=2.2, color="#d62728")
    plt.plot(thresholds, frrs, label="FRR", linewidth=2.2, color="#1f77b4")
    plt.legend()

    output_path = (
        PLOTS_DIR
        / split_name
        / csv_path.with_suffix(".png").name
    )
    save_current_figure(output_path)


def plot_comparison_chart(dataset_name, filename, split_dirs):
    split_data = {}

    for split_dir in split_dirs:
        csv_path = split_dir / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")

        split_data[split_dir.name] = read_threshold_sweep(csv_path)

    setup_axes(f"FAR / FRR Threshold Sweep - {dataset_name}")

    colors = {
        "FAR": ["#ff9896", "#d62728", "#8c1d18"],
        "FRR": ["#aec7e8", "#1f77b4", "#174a7c"],
    }

    for index, (split_name, (thresholds, fars, frrs)) in enumerate(split_data.items()):
        is_dense = index == len(split_data) - 1
        linestyle = "-" if is_dense else "--"
        marker = None if is_dense else "o"
        markersize = 0 if is_dense else 3
        linewidth = 2.2 if is_dense else 1.4
        color_index = min(index, len(colors["FAR"]) - 1)

        plt.plot(
            thresholds,
            fars,
            label=f"FAR {split_name} splits",
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
            markersize=markersize,
            color=colors["FAR"][color_index],
        )
        plt.plot(
            thresholds,
            frrs,
            label=f"FRR {split_name} splits",
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
            markersize=markersize,
            color=colors["FRR"][color_index],
        )

    plt.legend()

    output_path = (
        PLOTS_DIR
        / "comparison"
        / filename.replace(".csv", "_comparison.png")
    )
    save_current_figure(output_path)


def main():
    split_dirs = get_split_dirs()

    for split_dir in split_dirs:
        for dataset_name, filename in DATASETS:
            csv_path = split_dir / filename
            if not csv_path.exists():
                raise FileNotFoundError(f"Input CSV not found: {csv_path}")

            plot_single_chart(dataset_name, split_dir.name, csv_path)

    for dataset_name, filename in DATASETS:
        plot_comparison_chart(dataset_name, filename, split_dirs)


if __name__ == "__main__":
    main()
