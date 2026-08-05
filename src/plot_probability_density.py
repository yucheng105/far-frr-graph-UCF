from pathlib import Path
import csv

import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

SPLITS = ["20", "1000"]
DATASETS = [
    ("CelebDF", "threshold_sweep_xception_celebdf.csv"),
    ("DFDC", "threshold_sweep_xception_dfdc.csv"),
    ("FaceForensics", "threshold_sweep_xception_faceforensics.csv"),
    ("Combined", "threshold_sweep_xception_combined.csv"),
]


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
    plt.xlim(0, )
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


def plot_comparison_chart(dataset_name, filename):
    split_data = {}

    for split_name in SPLITS:
        csv_path = RESULTS_DIR / split_name / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")

        split_data[split_name] = read_threshold_sweep(csv_path)

    setup_axes(f"FAR / FRR Threshold Sweep - {dataset_name}")

    thresholds_20, fars_20, frrs_20 = split_data["20"]
    thresholds_500, fars_500, frrs_500 = split_data["500"]

    plt.plot(
        thresholds_20,
        fars_20,
        label="FAR 20 splits",
        linewidth=1.4,
        linestyle="--",
        marker="o",
        markersize=3,
        color="#ff9896",
    )
    plt.plot(
        thresholds_20,
        frrs_20,
        label="FRR 20 splits",
        linewidth=1.4,
        linestyle="--",
        marker="o",
        markersize=3,
        color="#aec7e8",
    )
    plt.plot(
        thresholds_500,
        fars_500,
        label="FAR 500 splits",
        linewidth=2.2,
        color="#d62728",
    )
    plt.plot(
        thresholds_500,
        frrs_500,
        label="FRR 500 splits",
        linewidth=2.2,
        color="#1f77b4",
    )
    plt.legend()

    output_path = (
        PLOTS_DIR
        / "comparison"
        / filename.replace(".csv", "_comparison.png")
    )
    save_current_figure(output_path)


def main():
    for split_name in SPLITS:
        for dataset_name, filename in DATASETS:
            csv_path = RESULTS_DIR / split_name / filename
            if not csv_path.exists():
                raise FileNotFoundError(f"Input CSV not found: {csv_path}")

            plot_single_chart(dataset_name, split_name, csv_path)

    for dataset_name, filename in DATASETS:
        plot_comparison_chart(dataset_name, filename)


if __name__ == "__main__":
    main()
