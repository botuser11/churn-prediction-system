import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def main():
    data = np.load("reports/probas_lr.npz", allow_pickle=True)
    p_val = data["p_val"]
    y_val = data["y_val"]
    p_test = data["p_test"]
    y_test = data["y_test"]

    Path("reports/figs").mkdir(parents=True, exist_ok=True)

    # Calibration curves
    frac_pos_val, mean_pred_val = calibration_curve(y_val, p_val, n_bins=10, strategy="uniform")
    frac_pos_test, mean_pred_test = calibration_curve(y_test, p_test, n_bins=10, strategy="uniform")

    # Plot VAL
    fig, ax = plt.subplots()
    ax.plot(mean_pred_val, frac_pos_val, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve (Validation)")
    fig.savefig("reports/figs/calibration_val.png", dpi=200, bbox_inches="tight")

    # Plot TEST
    fig, ax = plt.subplots()
    ax.plot(mean_pred_test, frac_pos_test, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve (Test)")
    fig.savefig("reports/figs/calibration_test.png", dpi=200, bbox_inches="tight")

    # Save the curve data too
    curves = {
        "val": {"mean_pred": mean_pred_val.tolist(), "frac_pos": frac_pos_val.tolist()},
        "test": {"mean_pred": mean_pred_test.tolist(), "frac_pos": frac_pos_test.tolist()},
    }
    with open("reports/calibration_curves.json", "w") as f:
        json.dump(curves, f, indent=2)

    print("Saved calibration plots + curve data.")

if __name__ == "__main__":
    main()
