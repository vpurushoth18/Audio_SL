#!/usr/bin/env python3
"""The paper's figures, in ACL/NAACL house style, built from the result TSVs.

Every value is read from eval/analysis/*.tsv, so nothing is transcribed by hand
and a figure changes when its numbers do.

Style is matched to the manuscript rather than to a screen:

  font      Nimbus Roman, the metric-compatible Times clone, because main.tex
            loads \\usepackage{times}. Figure text then matches body text
            instead of sitting beside it in another family.
  size      7-8 pt inside the figure, so it reads at print size without being
            rescaled. A figure drawn at 12 pt and shrunk to column width ends
            up with 6 pt labels.
  width     3.15 in for a one-column figure, 6.3 in for a figure* spanning both
            columns. Drawing at the final width means no rescaling at all.
  titles    none. The LaTeX \\caption carries the title; a title inside the
            image duplicates it and costs vertical space.
  colour    Okabe-Ito, the colour-blind-safe set conventional in this
            literature, assigned in fixed order and never cycled. The heatmap
            is greyscale, since its job is magnitude and it must survive a
            black-and-white printer.
  chrome    hairline axes, no top or right spine, dotted grid on the one axis
            a value is read off.

  fig1_layer_curves        two-column: capability against normalised depth
  fig2_controlled_matching one-column: content-over-speaker with intervals
  fig3_capability_heatmap  one-column: encoders x capabilities
  fig4_benchmark_transfer  two-column: MAEB standing against Sinhala standing
"""

import os

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.colors import LinearSegmentedColormap             # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
AN = os.path.join(ROOT, "eval/analysis")
OUT = os.path.join(ROOT, "eval/figures")

COL1, COL2 = 3.15, 6.30          # ACL one-column and two-column widths, inches

# Okabe-Ito, colour-blind safe, used in fixed order.
OI = {"blue": "#0072B2", "vermil": "#D55E00", "green": "#009E73",
      "orange": "#E69F00", "purple": "#CC79A7", "skyblue": "#56B4E9"}
SERIES = [OI["blue"], OI["vermil"], OI["green"], OI["orange"],
          OI["purple"], OI["skyblue"]]
INK, INK2, MUTED, GRID = "#000000", "#333333", "#777777", "#cccccc"

# hubert-large-ls960-ft is fine-tuned on LibriSpeech 960h, so it belongs with
# Whisper under a supervised ASR objective rather than with the purely
# self-supervised encoders. Getting this wrong mislabels the figure that the
# controlled-matching result rests on.
def ASR_SUPERVISED(model):
    return "whisper" in model or model.endswith("-ls960-ft")


# The MAEB subset grouped here covers gender, age, accent and emotion, which
# are speaker ATTRIBUTES, not speaker identity as the Sinhala tasks measure.
MAEB_LABEL = {"speaker": "speaker attributes", "content": "content",
              "non-speech": "non-speech"}

SHORT = {
    "facebook/hubert-large-ls960-ft": "HuBERT-L",
    "microsoft/wavlm-large": "WavLM-L",
    "facebook/w2v-bert-2.0": "w2v-BERT-2.0",
    "openai/whisper-large-v3": "Whisper-L-v3",
    "facebook/mms-300m": "MMS-300m",
    "facebook/wav2vec2-xls-r-300m": "XLS-R-300m",
    "facebook/wav2vec2-xls-r-1b": "XLS-R-1b",
    "facebook/wav2vec2-xls-r-2b": "XLS-R-2b",
    "openai/whisper-tiny": "Whisper-tiny",
    "openai/whisper-base": "Whisper-base",
    "openai/whisper-small": "Whisper-small",
    "openai/whisper-medium": "Whisper-medium",
    "baseline/mfcc-40": "MFCC-40",
}

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": INK, "axes.linewidth": 0.5,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.4, "grid.linestyle": ":",
    "legend.frameon": False, "legend.handlelength": 1.6,
    "legend.columnspacing": 1.0, "legend.handletextpad": 0.5,
    "lines.linewidth": 1.0, "lines.markersize": 3,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=400)
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def interval(ax, lo, hi, y, colour):
    """A horizontal interval with end caps, drawn thin enough for print."""
    ax.plot([lo, hi], [y, y], color=colour, lw=0.9, zorder=2)
    for x in (lo, hi):
        ax.plot([x, x], [y - 0.14, y + 0.14], color=colour, lw=0.9, zorder=2)


# ---------------------------------------------------------------- figure 1
def fig1_layer_curves():
    f = os.path.join(AN, "V_layer_curves.tsv")
    if not os.path.exists(f):
        print("  fig1 skipped: V_layer_curves.tsv not written yet")
        return
    d = pd.read_csv(f, sep="\t")
    order = (d.groupby("model").speaker_clf.max()
             .sort_values(ascending=False).index.tolist())
    colour = {m: SERIES[i % len(SERIES)] for i, m in enumerate(order)}

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.1), sharex=True)
    panels = [("speaker_clf", "speaker accuracy", None, "(a) speaker identity"),
              ("content_pref", r"$P[\mathrm{transcript}\succ\mathrm{speaker}]$",
               0.5, "(b) transcript-content preference")]
    for ax, (col, ylab, ref, tag) in zip(axes, panels):
        for m in order:
            g = d[d.model == m].sort_values("depth")
            ax.plot(g.depth, g[col], color=colour[m], lw=1.0,
                    label=SHORT.get(m, m), zorder=3)
            pk = g.loc[g[col].idxmax()]
            ax.plot(pk.depth, pk[col], "o", ms=2.8, color=colour[m],
                    mec="white", mew=0.5, zorder=4)
        if ref is not None:
            ax.axhline(ref, color=MUTED, lw=0.5, ls=(0, (2, 2)), zorder=1)
        ax.set_xlabel("normalised depth")
        ax.set_ylabel(ylab)
        ax.set_xlim(0, 1)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        # above the axes, since the speaker curves crowd the top-left corner
        ax.text(0.0, 1.03, tag, transform=ax.transAxes, fontsize=7.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    save(fig, "fig1_layer_curves")


# ---------------------------------------------------------------- figure 2
def fig2_controlled_matching():
    d = pd.read_csv(os.path.join(AN, "H_controlled_matching_final.tsv"), sep="\t")
    d = d.sort_values("p_content_over_speaker")
    fig, ax = plt.subplots(figsize=(COL1, 1.8))
    for i, (_, r) in enumerate(d.iterrows()):
        c = OI["blue"] if r.p_content_over_speaker > 0.5 else OI["vermil"]
        interval(ax, r.ci_lo, r.ci_hi, i, c)
        ax.plot(r.p_content_over_speaker, i, "o", ms=3.4, color=c,
                mec="white", mew=0.5, zorder=3)
    ax.axvline(0.5, color=MUTED, lw=0.5, ls=(0, (2, 2)), zorder=1)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([SHORT.get(m, m) for m in d.model])
    ax.set_xlabel(r"$P[\,$same words, other speaker closer$\,]$")
    ax.set_xlim(0.1, 0.72)
    ax.set_ylim(-0.6, len(d) - 0.4)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    save(fig, "fig2_controlled_matching")


# ---------------------------------------------------------------- figure 3
def fig3_capability_heatmap():
    d = pd.read_csv(os.path.join(AN, "S1_full_matrix_long.tsv"), sep="\t")
    piv = d.pivot_table(index="model", columns="capability", values="score",
                        aggfunc="mean")
    piv = piv[[c for c in ("speaker", "content", "channel") if c in piv.columns]]
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    # The three capabilities aggregate different metrics, so a shared scale
    # would imply a comparability that does not hold and would flatten the
    # content column. Shade encodes standing within a column; the cell prints
    # the score itself.
    norm = piv.apply(lambda c: (c - c.min()) / max(c.max() - c.min(), 1e-9))

    cmap = LinearSegmentedColormap.from_list("g", ["#ffffff", "#9a9a9a", "#1a1a1a"])
    fig, ax = plt.subplots(figsize=(COL1, 2.75))
    im = ax.imshow(norm.values, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([c.capitalize() for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([SHORT.get(m, m) for m in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.0f}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if norm.values[i, j] > 0.55 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, piv.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, piv.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2, linestyle="-")
    ax.tick_params(which="minor", length=0)
    ax.tick_params(length=0)
    # Without this key the shading reads as cross-column magnitude, which it is
    # not: each capability aggregates a different metric.
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[0, 1])
    cb.ax.set_yticklabels(["low", "high"], fontsize=6)
    cb.set_label("rank within column", fontsize=6.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    save(fig, "fig3_capability_heatmap")


# ---------------------------------------------------------------- figure 4
def fig4_benchmark_transfer():
    d = pd.read_csv(os.path.join(AN, "S6_maeb_capability_matched.tsv"), sep="\t")
    per = pd.read_csv(os.path.join(AN, "N_maeb_per_model.tsv"), sep="\t")
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.2),
                             gridspec_kw={"wspace": 0.5,
                                          "width_ratios": [1.3, 1]})

    d = d.sort_values("spearman")
    for i, (_, r) in enumerate(d.iterrows()):
        c = OI["blue"] if r.sinhala == "content" else OI["vermil"]
        interval(axes[0], r.ci_lo, r.ci_hi, i, c)
        axes[0].plot(r.spearman, i, "o", ms=3.4, color=c, mec="white", mew=0.5,
                     zorder=3)
    axes[0].axvline(0, color=MUTED, lw=0.5, ls=(0, (2, 2)), zorder=1)
    axes[0].set_yticks(range(len(d)))
    axes[0].set_yticklabels(
        [f"{MAEB_LABEL.get(r.maeb, r.maeb)} $\\rightarrow$ {r.sinhala}"
         for _, r in d.iterrows()], fontsize=6.5)
    axes[0].set_xlabel(r"Spearman $\rho$, 95% CI over 12 checkpoints")
    axes[0].set_ylim(-0.6, len(d) - 0.4)
    axes[0].grid(axis="x", zorder=0)
    axes[0].set_axisbelow(True)
    axes[0].text(0.0, 1.04, "(a)", transform=axes[0].transAxes, fontsize=7.5)

    per = per.sort_values("rank_sinhala_mean")
    for _, r in per.iterrows():
        c = OI["vermil"] if ASR_SUPERVISED(r.model) else OI["blue"]
        axes[1].plot([0, 1], [r.rank_maeb_overall, r.rank_sinhala_mean],
                     color=c, lw=0.8, zorder=2)
        axes[1].plot([0, 1], [r.rank_maeb_overall, r.rank_sinhala_mean], "o",
                     ms=2.2, color=c, mec="white", mew=0.4, zorder=3)
        axes[1].annotate(SHORT.get(r.model, r.model), xy=(1, r.rank_sinhala_mean),
                         xytext=(3, 0), textcoords="offset points", va="center",
                         fontsize=5.8, color=INK2)
    axes[1].set_xlim(-0.05, 2.0)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["MAEB", "Sinhala"])
    axes[1].set_yticks([1, 4, 8, 12])
    axes[1].invert_yaxis()
    axes[1].set_ylabel("rank")
    axes[1].grid(axis="y", zorder=0)
    axes[1].set_axisbelow(True)
    axes[1].text(0.0, 1.04, "(b)", transform=axes[1].transAxes, fontsize=7.5)
    axes[1].legend(handles=[Line2D([], [], color=OI["vermil"], lw=1.0,
                                   label="ASR-supervised"),
                            Line2D([], [], color=OI["blue"], lw=1.0,
                                   label="self-supervised")],
                   loc="lower center", bbox_to_anchor=(0.45, -0.45), ncol=2,
                   fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig4_benchmark_transfer")


# ---------------------------------------------------------------- figure 5
def fig5_layer_validation():
    """The validation itself: selected vs final vs all-layer, on both corpora.

    This is the comparison the recommendation rests on, so it belongs in the
    main paper rather than being summarised in prose. Three points per encoder
    per corpus, with the development-chosen layer frozen before scoring.
    """
    q = pd.read_csv(os.path.join(AN, "Q_layer_seeds_uniform.tsv"), sep="\t")
    o = pd.read_csv(os.path.join(AN, "O_layer_transfer_omnilingual.tsv"), sep="\t")
    q = q.rename(columns={"sel_mean": "sel", "final_mean": "final",
                          "uniform_mean": "uniform"})
    o = o.rename(columns={"sel_mean": "sel", "final_mean": "final",
                          "uniform_mean": "uniform"})
    order = q.sort_values("sel", ascending=False).model.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.3), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    panels = [(q, "(a) OpenSLR-30 (12 speakers)"),
              (o, "(b) Omnilingual (7 speakers)")]
    style = [("final", OI["vermil"], "o", "final layer"),
             ("uniform", OI["orange"], "s", "all-layer mean"),
             ("sel", OI["blue"], "D", "selected layer")]

    for ax, (d, tag) in zip(axes, panels):
        d = d.set_index("model").loc[order].reset_index()
        y = np.arange(len(d))[::-1]
        for i, r in d.iterrows():
            ax.plot([r["final"], r["sel"]], [y[i], y[i]], color=GRID, lw=3,
                    solid_capstyle="round", zorder=1)
        for col, c, mk, lab in style:
            ax.plot(d[col], y, mk, ms=3.6, color=c, mec="white", mew=0.5,
                    ls="none", label=lab, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT.get(m, m) for m in d.model])
        ax.set_xlabel("speaker clustering AMI")
        ax.set_xlim(0, 1.0)
        ax.grid(axis="x", zorder=0)
        ax.set_axisbelow(True)
        ax.text(0.0, 1.04, tag, transform=ax.transAxes, fontsize=7.5)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    save(fig, "fig5_layer_validation")


if __name__ == "__main__":
    import matplotlib.font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    print("font:", "Nimbus Roman (Times-compatible)" if "Nimbus Roman" in have
          else "FALLBACK - Nimbus Roman not installed")
    fig2_controlled_matching()
    fig3_capability_heatmap()
    fig4_benchmark_transfer()
    fig5_layer_validation()
    fig1_layer_curves()
    print(f"\n{OUT}")
