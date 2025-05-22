import typing
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tabulate
from IPython.display import Markdown, display
from loguru import logger
from palettable.colorbrewer.qualitative import Dark2_6 as olmo_cmap
from palettable.colorbrewer.qualitative import Paired_12 as llama_cmap
from shared_definitions import (
    BOTH,
    ICL,
    LONG,
    N_TOP_HEADS_VALUES,
    ORDERED_MODELS,
    SHORT,
    SPECIAL_RESULT_TYPE_SHORT_NAMES,
    SPECIAL_RESULT_TYPES,
    get_zs_prefix,
)

METRIC_RENAMES = {
    "10_shot_acc": "10-shot",
    "top_1_prompt_acc": "Top-1 prompt",
    "top_n_prompt_acc": "Top-5 prompts (mean)",
    "0_shot_acc": "0-shot",
    "shuffled_10_shot_acc": "Shuffled 10-shot",
}


# for prompt_type in (SHORT, LONG, BOTH, ICL):
#     for n_top_heads in N_TOP_HEADS_VALUES:
#         for zs in (False, True):
#             zs_prefix = get_zs_prefix(zs)
#             universal_layer_acc_key = f"{zs_prefix}_universal_{prompt_type}_all_{n_top_heads}_heads_same_layer_acc"
#             name_parts = []
#             if prompt_type != BOTH:
#                 name_parts.append(prompt_type.capitalize() if prompt_type != ICL else "ICL")
#             if prompt_type != ICL:
#                 name_parts.append("Prompt")
#             name_parts.append("FV")
#             if zs:
#                 name_parts.append("0-shot")
#             else:
#                 name_parts.append("shuffled 10-shot")

#             name_parts.append(f"({n_top_heads} heads)")

#             METRIC_RENAMES[universal_layer_acc_key] = " ".join(name_parts)


# for n_top_heads in N_TOP_HEADS_VALUES:
#     for zs in (False, True):
#         zs_prefix = get_zs_prefix(zs)
#         universal_layer_acc_key = f"{zs_prefix}_{n_top_heads}_heads_universal_FV_acc"
#         name_parts = ["Universal FV"]
#         if zs:
#             name_parts.append("0-shot")
#         else:
#             name_parts.append("shuffled 10-shot")
#         name_parts.append(f"({n_top_heads} heads)")
#         METRIC_RENAMES[universal_layer_acc_key] = " ".join(name_parts)


for prompt_type in (SHORT, LONG, BOTH, ICL):
    for n_top_heads in N_TOP_HEADS_VALUES:
        for zs in (False, True):
            zs_prefix = get_zs_prefix(zs)
            universal_layer_acc_key = f"{zs_prefix}_universal_{prompt_type}_all_{n_top_heads}_heads_same_layer_acc"
            special_type_keys = {
                special_type: f"{zs_prefix}_universal_{prompt_type}_{special_type}_all_{n_top_heads}_heads_same_layer_acc"
                for special_type in SPECIAL_RESULT_TYPES
            }

            name_parts = []
            special_type_name_parts = {special_type: [] for special_type in SPECIAL_RESULT_TYPES}

            if prompt_type != BOTH:
                name_parts.append(prompt_type.capitalize() if prompt_type != ICL else "ICL")
                for parts in special_type_name_parts.values():
                    parts.append(prompt_type.capitalize() if prompt_type != ICL else "ICL")
            if prompt_type != ICL:
                name_parts.append("Prompt")
                for spec_type, parts in special_type_name_parts.items():
                    parts.append(SPECIAL_RESULT_TYPE_SHORT_NAMES[spec_type])
            name_parts.append("FV")
            for parts in special_type_name_parts.values():
                parts.append("FV")
            if zs:
                name_parts.append("0-shot")
                for parts in special_type_name_parts.values():
                    parts.append("0-shot")
            else:
                name_parts.append("shuffled 10-shot")
                for parts in special_type_name_parts.values():
                    parts.append("shuffled 10-shot")

            name_parts.append(f"({n_top_heads} heads)")
            for parts in special_type_name_parts.values():
                parts.append(f"({n_top_heads} heads)")

            METRIC_RENAMES[universal_layer_acc_key] = " ".join(name_parts)
            if prompt_type != ICL:
                for key in special_type_keys:
                    METRIC_RENAMES[special_type_keys[key]] = " ".join(special_type_name_parts[key])


for n_top_heads in N_TOP_HEADS_VALUES:
    for zs in (False, True):
        zs_prefix = get_zs_prefix(zs)
        universal_layer_acc_key = f"{zs_prefix}_{n_top_heads}_heads_universal_FV_acc"
        special_type_keys = {
            special_type: f"{zs_prefix}_{n_top_heads}_universal_{special_type}_FV_acc"
            for special_type in SPECIAL_RESULT_TYPES
        }

        name_parts = ["Universal FV"]
        special_type_name_parts = {special_type: [] for special_type in SPECIAL_RESULT_TYPES}
        if zs:
            name_parts.append("0-shot")
            for parts in special_type_name_parts.values():
                parts.append("0-shot")
        else:
            name_parts.append("shuffled 10-shot")
            for parts in special_type_name_parts.values():
                parts.append("shuffled 10-shot")

        name_parts.append(f"({n_top_heads} heads)")
        for parts in special_type_name_parts.values():
            parts.append(f"({n_top_heads} heads)")

        METRIC_RENAMES[universal_layer_acc_key] = " ".join(name_parts)
        for key in special_type_keys:
            METRIC_RENAMES[special_type_keys[key]] = " ".join(special_type_name_parts[key])


for prefix in ("zs", "fs_shuffled"):
    orig_fv_type = "both_all" if prefix == "zs" else "icl_all"
    orig_fv_name = "Prompt FV" if prefix == "zs" else "ICL FV"

    for other_fv_type in ("icl_mean_activations", "icl_top_heads"):
        other_fv_name = "ICL activations" if "activations" in other_fv_type else "ICL top heads"
        for n_heads in (10, 20):
            METRIC_RENAMES[
                f"{prefix}_universal_{orig_fv_type}_minus_{other_fv_type}_{n_heads}_heads_same_layer_acc"
            ] = f"{orig_fv_name} - {other_fv_name} {'0-shot' if prefix == 'zs' else 'shuffled 10-shot'} ({n_heads} heads)"

            METRIC_RENAMES[f"{prefix}_universal_{orig_fv_type}_by_{other_fv_type}_{n_heads}_heads_same_layer_acc"] = (
                f"{orig_fv_name} / {other_fv_name} {'0-shot' if prefix == 'zs' else 'shuffled 10-shot'} ({n_heads} heads)"
            )


ZERO_SHOT_BASELINE_METRICS = [
    # removing the skyline numbers from the plots as they are misleading
    # "Top-5 prompts (mean)",
    "0-shot",
]

FEW_SHOT_SHUFFLED_BASELINE_METRICS = [
    # removing the skyline numbers from the plots as they are misleading
    # "10-shot",
    "Shuffled 10-shot",
]

METRIC_PLOT_LABELS = {
    "Top-5 prompts (mean)": "Instructed\n0-shot",
    "0-shot": "0-shot",
    "ICL FV 0-shot (20 heads)": "Demonstration FV",
    "Prompt FV 0-shot (20 heads)": "Instruction FV",
    "Joint FV 0-shot (20 heads)": "Both\nFVs",
    #
    "10-shot": "10-shot",
    "Shuffled 10-shot": "Shuffled\n10-shot",
    "ICL FV shuffled 10-shot (20 heads)": "Demonstration FV",
    "Prompt FV shuffled 10-shot (20 heads)": "Instruction FV",
    "Joint FV shuffled 10-shot (20 heads)": "Both\nFVs",
    #
    "ICL activations FV 0-shot (20 heads)": "Instruction heads\nDemonstration acts",
    "ICL top heads FV 0-shot (20 heads)": "Demonstration heads\nInstruction acts",
    "ICL activations FV shuffled 10-shot (20 heads)": "Instruction heads\nDemonstration acts",
    "ICL top heads FV shuffled 10-shot (20 heads)": "Demonstration heads\nInstruction acts",
    #
    "Prompt FV twice FV 0-shot (20 heads)": "Instruction FV\nAdded twice",
    "ICL FV twice FV 0-shot (20 heads)": "Demonstration FV\nAdded twice",
    "Prompt FV twice FV shuffled 10-shot (20 heads)": "Instruction FV\nAdded twice",
    "ICL FV twice FV shuffled 10-shot (20 heads)": "Demonstration FV\nAdded twice",
    #
    "ICL least imp heads FV 0-shot (20 heads)": "Demonstration FV\nLeast imp. heads",
    "Prompt least imp heads FV 0-shot (20 heads)": "Instruction FV\nLeast imp. heads",
    "ICL least imp heads FV shuffled 10-shot (20 heads)": "Demonstration FV\nLeast imp. heads",
    "Prompt least imp heads FV shuffled 10-shot (20 heads)": "Instruction FV\nLeast imp. heads",
    #
    "ICL bottom heads FV 0-shot (20 heads)": "Demonstration FV\nBottom heads",
    "Prompt bottom heads FV 0-shot (20 heads)": "Instruction FV\nBottom heads",
    "ICL bottom heads FV shuffled 10-shot (20 heads)": "Demonstration FV\nBottom heads",
    "Prompt bottom heads FV shuffled 10-shot (20 heads)": "Instruction FV\nBottom heads",
    #
    "Instruct model FV 0-shot (20 heads)": "Post-trained Model\nInstruction FV",
    "Instruct model FV shuffled 10-shot (20 heads)": "Post-trained Model\nInstruction FV",
    #
    "ICL FV - ICL activations shuffled 10-shot (20 heads)": "Instruction heads\nDemonstration acts",
    "ICL FV - ICL top heads shuffled 10-shot (20 heads)": "Demonstration heads\nInstruction acts",
    "Prompt FV - ICL activations 0-shot (20 heads)": "Instruction heads\nDemonstration acts",
    "Prompt FV - ICL top heads 0-shot (20 heads)": "Demonstration heads\nInstruction acts",
}

METRIC_GROUP_NAMES = ["Shuffled 10-shot Eval", "0-shot Eval"]
METRIC_GROUP_NAME_KEYWORDS = {"shuffled 10-shot": METRIC_GROUP_NAMES[0], "0-shot": METRIC_GROUP_NAMES[1]}

MAIN_PLOT_MODELS = ["Llama-3.2-3B", "Llama-3.2-3B-Instruct", "Llama-3.1-8B", "Llama-3.1-8B-Instruct"]
BASE_MODELS = ["Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.1-8B", "OLMo-2-1124-7B", "Llama-2-7b-hf"]
OLMO_MODELS = ["OLMo-2-1124-7B", "OLMo-2-1124-7B-SFT", "OLMo-2-1124-7B-DPO", "OLMo-2-1124-7B-Instruct"]

MODEL_PLOT_STYLES = {
    "Llama-3.2-1B": {"marker": "o", "color": [0.69140625, 0.47265625, 0.3359375]},
    "Llama-3.2-1B-Instruct": {"marker": "s", "color": llama_cmap.mpl_colors[11]},
    "Llama-3.2-3B": {"marker": "o", "color": llama_cmap.mpl_colors[8]},
    "Llama-3.2-3B-Instruct": {"marker": "s", "color": llama_cmap.mpl_colors[9]},
    "Llama-3.1-8B": {"marker": "o", "color": llama_cmap.mpl_colors[6]},
    "Llama-3.1-8B-Instruct": {"marker": "s", "color": llama_cmap.mpl_colors[7]},
    "OLMo-2-1124-7B": {"marker": ">", "color": olmo_cmap.mpl_colors[3]},
    "OLMo-2-1124-7B-SFT": {"marker": "v", "color": olmo_cmap.mpl_colors[5]},
    "OLMo-2-1124-7B-DPO": {"marker": "<", "color": olmo_cmap.mpl_colors[4]},
    "OLMo-2-1124-7B-Instruct": {"marker": "^", "color": olmo_cmap.mpl_colors[0]},
    "Llama-2-7b-hf": {"marker": "o", "color": [0.375, 0.5, 0.5]},
    "Llama-2-7b-chat-hf": {"marker": "s", "color": [0.25, 0.75, 0.75]},
}

METRIC_PLOT_STYLES = {
    "0-shot": {"linestyle": ":", "linewidth": 2},
    "Top-5 prompts (mean)": {"linestyle": "--", "linewidth": 2},
    "Shuffled 10-shot": {"linestyle": "--", "linewidth": 2},
    "10-shot": {"linestyle": "--", "linewidth": 2},
}

GLOBAL_PLOT_STYLE = {
    "markersize": 10,
    "linewidth": 3,
    "alpha": 0.8,
}

# def make_model_order_key(models: typing.List[str]):
#     def model_order_key(index):
#         return [models.index(k) if k in models else -1 for k in index]

#     return model_order_key


def model_order_key(index):
    return [ORDERED_MODELS.index(k) if k in ORDERED_MODELS else -1 for k in index]


def print_summary(
    df,
    columns: str,
    # models: typing.List[str],
    sep_rows: typing.Sequence[int] | None = None,
    groupby_col: str = "model",
    agg: str = "mean",
    use_sem: bool | None = None,
):
    summary_df = df.copy()
    # model_order_key = make_model_order_key(models)

    gb_model = summary_df.groupby(groupby_col)
    if use_sem is not None:
        agg = [agg, "sem" if use_sem else "std"]

    result = gb_model[columns].agg(agg).sort_index(key=model_order_key)

    if isinstance(agg, str):
        result_dicts = [{"metric": METRIC_RENAMES.get(k, k), **v} for k, v in result.T.to_dict(orient="index").items()]
    else:
        metric_dicts = result.T.to_dict(orient="index")
        agg_keys = [k for k in metric_dicts.keys() if k[1] == agg[0]]
        result_dicts = []
        for agg_key in agg_keys:
            other_key = (agg_key[0], agg[1])
            metric = agg_key[0]
            metric = METRIC_RENAMES.get(metric, metric)
            rd = dict(metric=metric)
            for model, model_agg in metric_dicts[agg_key].items():
                rd[model] = (model_agg, metric_dicts[other_key][model])

            result_dicts.append(rd)

    for d in result_dicts:
        if isinstance(agg, str) and agg == "mean":
            d["mean (no 13b)"] = np.nanmean([v for k, v in d.items() if (k != "metric") and ("13b" not in k)])
        elif agg[0] == "mean":
            d["mean (no 13b)"] = np.nanmean([v for k, v in d.items() if (k != "metric") and ("13b" not in k)], axis=0)

    headers = result_dicts[0].keys()
    if isinstance(agg, str):
        result_rows = [[d[k] for k in d] for d in result_dicts]
    else:
        result_rows = [
            [f"{d[k][0]:.4f} \u00b1 {d[k][1]:.4f}" if k != "metric" else d[k] for k in d] for d in result_dicts
        ]

    if sep_rows is not None:
        for i in sorted(sep_rows, reverse=True):
            result_rows.insert(i, tabulate.SEPARATING_LINE)

    display(Markdown(tabulate.tabulate(result_rows, headers=headers, tablefmt="github")))
    return result_dicts


def plot_metric_grouped_bar_chart(
    model_dicts: typing.Dict[str, typing.Dict[str, float]],
    models: typing.Sequence[str],
    metric_names: typing.Sequence[str],
    colors: typing.Sequence[str] = None,
    title: str | None = None,
    metric_labels: typing.Sequence[str] | None = None,
    ylabel: str = "Accuracy",
    fontsize: int = 20,
    font_inc: int = 4,
    legend_outside: bool = True,
):
    n_metrics = len(metric_names)
    n_models = len(models)

    if metric_labels is not None and len(metric_labels) != n_metrics:
        raise ValueError(
            f"Length of metric_labels ({len(metric_labels)}) does not match number of metrics ({n_metrics})."
        )

    if colors is not None and len(colors) != n_metrics:
        raise ValueError(f"Length of colors ({len(colors)}) does not match number of metrics ({n_metrics}).")

    bar_width = 0.8 / n_metrics
    bar_positions = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(12 + (2 * int(legend_outside)), 8))

    for i, metric_name in enumerate(metric_names):
        metric_values = [model_dicts[model].get(metric_name, np.nan) for model in models]
        if np.any(np.isnan(metric_values)):
            logger.warning(f"Missing data for metric {metric_name} in models {models}.")
            continue
        bar_kwargs = dict(width=bar_width, label=metric_name)
        if colors is not None:
            bar_kwargs["color"] = colors[i]
        ax.bar(bar_positions + i * bar_width, metric_values, **bar_kwargs)
        if metric_labels is not None:
            for j, bp in enumerate(bar_positions):
                ax.text(
                    bp + (i * bar_width),
                    0.02,
                    metric_labels[i],
                    ha="center",
                    fontsize=fontsize - font_inc,
                    fontweight="semibold",
                    rotation="vertical",
                )

    ax.set_xticks(bar_positions + bar_width * (n_metrics - 1) / 2)
    ax.set_xticklabels(models, fontsize=fontsize + font_inc)
    ax.set_ylabel(ylabel, fontsize=fontsize + font_inc)
    if title is not None:
        ax.set_title(title, fontsize=fontsize + (2 * font_inc))

    legend_kwargs = dict(fontsize=fontsize - (font_inc * 2))
    if legend_outside:
        legend_kwargs["bbox_to_anchor"] = (1.05, 1)
        legend_kwargs["loc"] = "upper left"
    ax.legend(**legend_kwargs)
    # ax.legend(fontsize=fontsize - font_inc)
    ax.tick_params(axis="y", labelsize=fontsize)
    # plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_connected_scatter(
    model_dicts: typing.Dict[str, typing.Dict[str, float]],
    models: typing.Sequence[str],
    metric_and_baseline_names: typing.Sequence[
        typing.Tuple[str | typing.Sequence[str], str | typing.Sequence[str] | None]
    ],
    metric_group_names_from_keywords: bool = True,
    metric_group_names: str | typing.Sequence[str] | None = METRIC_GROUP_NAMES,
    model_plot_styles: typing.Dict[str, typing.Dict[str, typing.Any]] = None,
    metric_plot_styles: typing.Dict[str, typing.Dict[str, typing.Any]] = None,
    global_plot_style: typing.Dict[str, typing.Any] = None,
    title: str | None = None,
    metric_labels: typing.Dict[str, str] | None = None,
    data_scaler: typing.Callable | None = None,
    annotate_baselines: bool = False,
    baseline_x_inside_increment: float = 0.15,
    baseline_x_outside_increment: float | None = None,
    baseline_annotation_nudges: typing.Dict[str, float] | None = None,
    baseline_y_loc_func: typing.Callable = np.min,
    baseline_default_nudge: float = -0.02,
    baseline_va: str = "top",
    show_error_bars: bool = True,
    ylim: typing.Tuple[int | float, int | float] = (0, 1),
    ylabel: str = "Accuracy",
    yticks: typing.Sequence[float] | None = None,
    xtick_nudges: float | typing.Sequence[float] = 0,
    nudge_metric_groups: bool = True,
    fontsize: int = 20,
    font_inc: int = 4,
    text_font_inc: int = 2,
    show_legend: bool = True,
    legend_outside: bool = True,
    legend_loc: str | typing.Tuple[int, int] = None,
    legend_fontsize: int | None = None,
    legend_width: float = 2.0,
    add_legend_baseline_entries: bool = False,
    width: float = 12.0,
    height: float = 8.0,
    ax: plt.Axes = None,
    annotate_panel: str | None = None,
    annotate_panel_position: typing.Tuple[float, float] = (1.01, 1.0),
    annotate_font_inc: int = 8,
    fontfamily: str | None = None,
    textfontweight: str = "semibold",
    should_show: bool | None = None,
):
    n_metrics_by_group = [
        1 if isinstance(metric_name, str) else len(metric_name) for metric_name, _ in metric_and_baseline_names
    ]
    n_metrics = sum(n_metrics_by_group)
    n_metrics_by_group_cumsum = np.cumsum([0] + n_metrics_by_group)

    all_metric_names = []
    all_baseline_metric_names = []

    if metric_labels is None:
        metric_labels = {}

    metric_labels = {**METRIC_PLOT_LABELS, **metric_labels}

    for metric_name, baseline_metric_name in metric_and_baseline_names:
        (all_metric_names.append if isinstance(metric_name, str) else all_metric_names.extend)(metric_name)

        if baseline_metric_name is not None:
            (
                all_baseline_metric_names.append
                if isinstance(baseline_metric_name, str)
                else all_baseline_metric_names.extend
            )(baseline_metric_name)

    for metric_name in all_metric_names:
        if metric_name not in metric_labels:
            metric_labels[metric_name] = METRIC_RENAMES.get(metric_name, metric_name)

    for baseline_metric_name in all_baseline_metric_names:
        if baseline_metric_name not in metric_labels:
            metric_labels[baseline_metric_name] = METRIC_RENAMES.get(baseline_metric_name, baseline_metric_name)

    if metric_group_names is not None and isinstance(metric_group_names, str):
        metric_group_names = [metric_group_names] * len(metric_and_baseline_names)

    # if baseline_metric_names is None:
    #     baseline_metric_names = []

    if model_plot_styles is None:
        model_plot_styles = defaultdict(dict)

    all_model_names = set(model_plot_styles.keys()) | set(MODEL_PLOT_STYLES.keys())
    for model_name in all_model_names:
        model_plot_styles[model_name] = {
            **MODEL_PLOT_STYLES.get(model_name, {}),
            **model_plot_styles.get(model_name, {}),
        }

    if metric_plot_styles is None:
        metric_plot_styles = defaultdict(dict)

    all_metric_style_names = set(metric_plot_styles.keys()) | set(METRIC_PLOT_STYLES.keys())
    for metric_name in all_metric_style_names:
        metric_plot_styles[metric_name] = {
            **METRIC_PLOT_STYLES.get(metric_name, {}),
            **metric_plot_styles.get(metric_name, {}),
        }

    if global_plot_style is None:
        global_plot_style = {}

    global_plot_style = {**GLOBAL_PLOT_STYLE, **global_plot_style}

    if baseline_x_outside_increment is None:
        baseline_x_outside_increment = baseline_x_inside_increment

    if baseline_annotation_nudges is None:
        baseline_annotation_nudges = {}

    if fontfamily is None:
        fontfamily = plt.rcParams["font.family"]

    x_positions = np.arange(n_metrics) + 0.5

    if ax is None:
        if should_show is None:
            should_show = True
        fig, ax = plt.subplots(figsize=(width + (legend_width * int(legend_outside)), height))

    if should_show is None:
        should_show = False

    ax.set_ylim(ylim)
    ax.set_xlim(x_positions[0] - baseline_x_outside_increment, x_positions[-1] + baseline_x_outside_increment)

    models_in_legend = set()

    for m, (metric_names, baseline_metric_names) in enumerate(metric_and_baseline_names):
        current_group_x_positions = x_positions[n_metrics_by_group_cumsum[m] : n_metrics_by_group_cumsum[m + 1]]

        if isinstance(metric_names, str):
            metric_names = [metric_names]

        if isinstance(baseline_metric_names, str):
            baseline_metric_names = [baseline_metric_names]

        for i, model in enumerate(models):
            model_dict = model_dicts[model]
            metric_values = [model_dict.get(metric_name, np.nan) for metric_name in metric_names]
            if np.any(np.isnan(metric_values)):
                logger.warning(f"Missing data for model {model} in metrics {metric_names}.")
                continue

            metric_errors = None
            if isinstance(metric_values[0], (list, tuple)):
                metric_errors = [mv[1] for mv in metric_values]
                metric_values = [mv[0] for mv in metric_values]

            if data_scaler is not None:
                metric_values = [data_scaler(mv) for mv in metric_values]

            combined_plot_style = {
                **global_plot_style,
                **model_plot_styles.get(model, {}),
            }
            if model in models_in_legend:
                label = "_" + model
            else:
                label = model
                models_in_legend.add(model)

            ax.plot(
                current_group_x_positions,
                metric_values,
                label=label,
                **combined_plot_style,
            )
            if metric_errors is not None and show_error_bars:
                ax.errorbar(
                    current_group_x_positions,
                    metric_values,
                    yerr=metric_errors,
                    fmt="none",
                    capsize=5,
                    elinewidth=2,
                    **combined_plot_style,
                )

            for baseline_metric_name in baseline_metric_names:
                baseline_metric_value = model_dict.get(baseline_metric_name, np.nan)
                if isinstance(baseline_metric_value, (list, tuple)):
                    # Ignore baseline std/sem for plotting purposes
                    baseline_metric_value = baseline_metric_value[0]

                if np.isnan(baseline_metric_value):
                    logger.warning(f"Missing data for model {model} in baseline metric {baseline_metric_name}.")
                    continue
                metric_plot_style = {
                    **combined_plot_style,
                    **metric_plot_styles.get(baseline_metric_name, {}),
                }
                metric_plot_style.pop("marker", None)
                metric_plot_style.pop("markersize", None)
                # if len(current_group_x_positions) > 1:
                #     metric_plot_style["xmin"] = current_group_x_positions[0]
                #     metric_plot_style["xmax"] = current_group_x_positions[-1]
                if m == 0:
                    metric_plot_style["xmin"] = current_group_x_positions[0] - baseline_x_outside_increment
                    metric_plot_style["xmax"] = current_group_x_positions[-1] + baseline_x_inside_increment
                else:
                    metric_plot_style["xmin"] = current_group_x_positions[0] - baseline_x_inside_increment
                    metric_plot_style["xmax"] = current_group_x_positions[-1] + baseline_x_outside_increment

                ax.hlines(
                    y=baseline_metric_value,
                    zorder=-1,
                    **metric_plot_style,
                )

        baseline_tick_locations = []

        if annotate_baselines:
            for baseline_metric_name in baseline_metric_names:
                baseline_metric_values = [model_dicts[model].get(baseline_metric_name, np.nan) for model in models]
                if isinstance(baseline_metric_values[0], (list, tuple)):
                    # Ignore baseline std/sem for plotting purposes
                    baseline_metric_values = [mv[0] for mv in baseline_metric_values]

                baseline_value = baseline_y_loc_func(baseline_metric_values)
                baseline_tick_locations.append(baseline_value)
                if np.isnan(baseline_value):
                    logger.warning(f"Missing data for baseline metric {baseline_metric_name}.")
                    continue

                if m == 0 and len(metric_and_baseline_names) > 1:
                    x = ax.get_xlim()[0] + 0.025
                    ha = "left"
                else:
                    x = ax.get_xlim()[1] - 0.025
                    ha = "right"

                ax.text(
                    x,
                    baseline_value + baseline_annotation_nudges.get(baseline_metric_name, baseline_default_nudge),
                    metric_labels.get(baseline_metric_name, baseline_metric_name),
                    ha=ha,
                    va=baseline_va,
                    fontsize=fontsize - text_font_inc,
                    fontweight=textfontweight,
                    rotation="horizontal",
                    fontfamily=fontfamily,
                )

        metric_group_x_nudge = 0
        if nudge_metric_groups:
            if isinstance(xtick_nudges, (float, int)):
                metric_group_x_nudge = xtick_nudges
            elif len(xtick_nudges) == len(metric_and_baseline_names):
                metric_group_x_nudge = xtick_nudges[m]

        if metric_group_names_from_keywords:
            for keyword, group_name in METRIC_GROUP_NAME_KEYWORDS.items():
                if all(keyword in metric for metric in metric_names):
                    ax.text(
                        np.mean(current_group_x_positions) + metric_group_x_nudge,
                        ax.get_ylim()[1] + 0.02,
                        group_name,
                        ha="center",
                        va="bottom",
                        fontsize=fontsize - text_font_inc,
                        fontweight=textfontweight,
                        fontfamily=fontfamily,
                    )
                    break

            else:
                raise ValueError(
                    f"`metric_group_names_from_keywords=True` but could not infer a group name from metrics: {metric_names}"
                )

        elif metric_group_names is not None:
            ax.text(
                np.mean(current_group_x_positions),
                ax.get_ylim()[1] + 0.02,
                metric_group_names[m],
                ha="center",
                va="bottom",
                fontsize=fontsize - text_font_inc,
                fontweight=textfontweight,
                fontfamily=fontfamily,
            )

    if annotate_panel is not None:
        ax.text(
            *annotate_panel_position,
            annotate_panel,
            ha="left",
            va="top",
            fontsize=fontsize + annotate_font_inc,
            fontweight="bold",
            fontfamily=fontfamily,
            transform=ax.transAxes,
        )

    if isinstance(xtick_nudges, (int, float)):
        xtick_nudges = np.ones_like(x_positions) * xtick_nudges
    else:
        xtick_nudges = np.array(xtick_nudges)

    ax.set_xticks(x_positions + xtick_nudges)
    ax.set_xticklabels(
        [metric_labels.get(m, m) for m in all_metric_names],
        fontsize=fontsize + font_inc,
        fontfamily=fontfamily,
        fontweight=textfontweight,
    )
    if yticks is not None:
        ax.set_yticks(yticks)

    ax.set_ylabel(ylabel, fontsize=fontsize + font_inc, fontfamily=fontfamily, fontweight=textfontweight)
    ax.tick_params(axis="both", labelsize=fontsize - text_font_inc, labelfontfamily=fontfamily)
    # plt.xticks(rotation=45)

    if show_legend:
        legend_kwargs = dict(
            prop=dict(family=fontfamily, size=fontsize - text_font_inc if legend_fontsize is None else legend_fontsize),
        )
        if legend_outside:
            legend_kwargs["bbox_to_anchor"] = (1.05, 1)
            legend_kwargs["loc"] = "upper left"
        elif legend_loc is not None:
            legend_kwargs["loc"] = legend_loc

        if add_legend_baseline_entries:
            handles, labels = ax.get_legend_handles_labels()
            for baseline_metric_name in set(all_baseline_metric_names):
                handles.append(
                    matplotlib.lines.Line2D([0], [0], color="black", **METRIC_PLOT_STYLES.get(baseline_metric_name, {}))
                )
                labels.append(METRIC_RENAMES.get(baseline_metric_name, baseline_metric_name))

            legend_kwargs["handles"] = handles
            legend_kwargs["labels"] = labels

        ax.legend(**legend_kwargs)

    if title is not None:
        ax.set_title(title, fontsize=fontsize + font_inc, fontfamily=fontfamily, fontweight=textfontweight)

    # if baseline_tick_locations:
    #     baseline_tick_locations.sort()
    #     ax2 = ax.twinx()
    #     ax2.set_yticks(baseline_tick_locations)
    #     ax2.set_yticklabels([""] * len(baseline_tick_locations))

    if should_show:
        plt.tight_layout()
        plt.show()


FIGURE_TEMPLATE = r"""\begin{{figure}}[!htb]
% \vspace{{-0.225in}}
\centering
\includegraphics[width=\linewidth]{{figures/{save_path}}}
\caption{{ {{\bf FIGURE TITLE.}} FIGURE DESCRIPTION.}}
\label{{fig:{label_name}}}
% \vspace{{-0.2in}}
\end{{figure}}
"""
WRAPFIGURE_TEMPLATE = r"""\begin{{wrapfigure}}{{r}}{{0.5\linewidth}}
\vspace{{-.3in}}
\begin{{spacing}}{{1.0}}
\centering
\includegraphics[width=0.95\linewidth]{{figures/{save_path}}}
\caption{{ {{\bf FIGURE TITLE.}} FIGURE DESCRIPTION.}}
\label{{fig:{label_name}}}
\end{{spacing}}
% \vspace{{-.25in}}
\end{{wrapfigure}}"""


SAVE_PATH_PREFIX = Path("./figures").absolute()


def save_plot(
    save_path: str | Path,
    bbox_inches="tight",
    should_print=False,
    transparent=False,
):
    if save_path is None:
        return

    if isinstance(save_path, str):
        save_path = Path(save_path)

    if should_print:
        print("Figure:\n")
        print(
            FIGURE_TEMPLATE.format(save_path=save_path, label_name=save_path.stem.replace("/", "-").replace("_", "-"))
        )
        print("\nWrapfigure:\n")
        print(
            WRAPFIGURE_TEMPLATE.format(
                save_path=save_path, label_name=save_path.stem.replace("/", "-").replace("_", "-")
            )
        )
        print("")

    if not save_path.is_relative_to(SAVE_PATH_PREFIX):
        save_path = SAVE_PATH_PREFIX / save_path

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        save_path,
        bbox_inches=bbox_inches,
        facecolor=plt.gcf().get_facecolor(),
        edgecolor="none",
        transparent=transparent,
    )
