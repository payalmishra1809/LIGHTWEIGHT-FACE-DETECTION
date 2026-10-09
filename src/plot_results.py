"""plot_results.py - tables and figures from benchmark + metrics CSVs.

Run from anywhere (paths are resolved relative to this file's project root):
    python src/plot_results.py
    python src/plot_results.py --drop-first 2      # discard first 2 repeats per config (warm-up)

Expects in <project>/results/:
    benchmark_multithread.csv, benchmark_1thread.csv, metrics.csv
Outputs go to <project>/results/figures/.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = R / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MULTI = R / "benchmark_multithread.csv"
SINGLE = R / "benchmark_1thread.csv"
METRICS = R / "metrics.csv"


def pick(df, *keys):
    """Return first column whose lowercase name contains a keyword."""
    for k in keys:
        for c in df.columns:
            if k in c.lower():
                return c
    raise KeyError(f"None of {keys} found. Your columns: {list(df.columns)}")


def tidy(path, label, drop_first):
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "detector": df[pick(df, "detector", "model", "name")],
        "side": df[pick(df, "side", "size", "res")],
        "ms": df[pick(df, "ms_mean", "mean_ms", "latency", "time_ms", "ms_per", "ms")],
        "cpu_ms": df[pick(df, "cpu_ms", "cpu_time")],
        "setting": label,
    })
    # one row per repeat in the CSV: optionally drop warm-up repeats per config
    if drop_first > 0:
        rep = out.groupby(["detector", "side"]).cumcount()
        out = out[rep >= drop_first]
    # aggregate repeats -> mean and SD per config
    agg = (out.groupby(["detector", "side", "setting"], sort=False)
              .agg(ms=("ms", "mean"), ms_sd=("ms", "std"),
                   cpu_ms=("cpu_ms", "mean"), cpu_ms_sd=("cpu_ms", "std"),
                   repeats=("ms", "size"))
              .reset_index())
    agg[["ms_sd", "cpu_ms_sd"]] = agg[["ms_sd", "cpu_ms_sd"]].fillna(0)
    agg["fps"] = 1000.0 / agg["ms"]
    return agg


NOTE_RES10 = "res10 always receives a 300x300 blob; 'side' only changes the pre-resize"


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks([320, 416, 640]) if "side" in xlabel.lower() else None
    ax.grid(alpha=0.3)
    ax.legend()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-first", type=int, default=0,
                    help="discard the first N repeats of every config (warm-up)")
    a = ap.parse_args()

    frames = []
    for p, lab in [(MULTI, "multi-thread"), (SINGLE, "1 thread")]:
        if p.exists():
            frames.append(tidy(p, lab, a.drop_first))
        else:
            print(f"skipping missing file: {p}")
    if not frames:
        raise SystemExit(f"No benchmark CSVs found in {R}")
    bench = pd.concat(frames, ignore_index=True)
    settings = list(bench["setting"].unique())

    # ---- metrics (optional) ----
    met = None
    if METRICS.exists():
        m = pd.read_csv(METRICS)
        # metrics.csv holds one row per condition PLUS an 'overall' row. Averaging
        # them all double-counts; use the pooled (micro-averaged) overall row only.
        if "condition" in m.columns:
            m = m[m["condition"] == "overall"]
        met = pd.DataFrame({
            "detector": m[pick(m, "detector", "model", "name")],
            "side": m[pick(m, "side", "size", "res")],
            "f1": m[pick(m, "f1")],
        })
        for col in ("precision", "recall"):
            try:
                met[col] = m[pick(m, col)]
            except KeyError:
                pass
        assert not met.duplicated(["detector", "side"]).any(), "expected one metrics row per detector/side"
    else:
        print(f"note: {METRICS} not found - F1 plots skipped")

    # ---- 1) latency vs resolution (one panel per setting) ----
    fig, axes = plt.subplots(1, len(settings), figsize=(6 * len(settings), 4))
    axes = [axes] if len(settings) == 1 else list(axes)
    for ax, lab in zip(axes, settings):
        g = bench[bench["setting"] == lab]
        for det, d in g.groupby("detector"):
            d = d.sort_values("side")
            ax.errorbar(d["side"], d["ms"], yerr=d["ms_sd"], marker="o", capsize=3, label=det)
        style(ax, "Input side (px)", "Latency per image (ms)", f"Latency - {lab}")
    fig.tight_layout()
    fig.savefig(OUT / "latency_vs_resolution.png", dpi=200)
    plt.close(fig)

    # ---- 2) CPU time per image ----
    fig, axes = plt.subplots(1, len(settings), figsize=(6 * len(settings), 4))
    axes = [axes] if len(settings) == 1 else list(axes)
    for ax, lab in zip(axes, settings):
        g = bench[bench["setting"] == lab]
        for det, d in g.groupby("detector"):
            d = d.sort_values("side")
            ax.errorbar(d["side"], d["cpu_ms"], yerr=d["cpu_ms_sd"], marker="o", capsize=3, label=det)
        style(ax, "Input side (px)", "CPU time per image (cpu-ms)", f"CPU cost - {lab}")
    fig.tight_layout()
    fig.savefig(OUT / "cpu_ms_vs_resolution.png", dpi=200)
    plt.close(fig)

    # ---- 3) F1 / precision / recall + accuracy-vs-speed ----
    if met is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        for det, d in met.groupby("detector"):
            d = d.sort_values("side")
            ax.plot(d["side"], d["f1"], marker="o", label=det)
        style(ax, "Input side (px)", "F1 (IoU >= 0.5)", "F1 vs resolution")
        fig.tight_layout()
        fig.savefig(OUT / "f1_vs_resolution.png", dpi=200)
        plt.close(fig)

        if {"precision", "recall"} <= set(met.columns):
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for ax, col in zip(axes, ("precision", "recall")):
                for det, d in met.groupby("detector"):
                    d = d.sort_values("side")
                    ax.plot(d["side"], d[col], marker="o", label=det)
                style(ax, "Input side (px)", col.capitalize(), f"{col.capitalize()} vs resolution")
            fig.tight_layout()
            fig.savefig(OUT / "precision_recall_vs_resolution.png", dpi=200)
            plt.close(fig)

        pref = "1 thread" if "1 thread" in settings else settings[0]
        j = bench[bench["setting"] == pref].merge(met, on=["detector", "side"])
        fig, ax = plt.subplots(figsize=(6, 4))
        for det, d in j.groupby("detector"):
            ax.scatter(d["ms"], d["f1"], s=60, label=det)
            for _, r in d.iterrows():
                ax.annotate(int(r["side"]), (r["ms"], r["f1"]),
                            textcoords="offset points", xytext=(5, 4), fontsize=8)
        ax.set_xlabel(f"Latency per image (ms, {pref})")
        ax.set_ylabel("F1")
        ax.set_title("Accuracy vs speed (labels = input side)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "accuracy_vs_speed.png", dpi=200)
        plt.close(fig)

        table = bench.merge(met, on=["detector", "side"], how="left")
    else:
        table = bench

    # ---- summary table ----
    table = table.sort_values(["setting", "detector", "side"]).round(3)
    table.to_csv(OUT / "summary_table.csv", index=False)
    try:
        (OUT / "summary_table.md").write_text(table.to_markdown(index=False))
    except ImportError:
        print("tip: pip install tabulate to also get summary_table.md")
    print(table.to_string(index=False))
    print(f"\nFigures saved in {OUT}")


if __name__ == "__main__":
    main()
