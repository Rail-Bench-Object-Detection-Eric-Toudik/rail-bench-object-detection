import json
import re
from pathlib import Path


def extract_json_from_text(text: str) -> dict:
    """
    Extracts the JSON/dict block after 'Result from run_object_eval.py:'.
    Works with pasted console output that contains text before/after the JSON.
    """
    marker = "Result from run_object_eval.py:"
    if marker in text:
        text = text.split(marker, 1)[1]

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in input text.")

    # Find matching closing brace
    brace_count = 0
    end = None

    for i, char in enumerate(text[start:], start=start):
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                end = i + 1
                break

    if end is None:
        raise ValueError("JSON object appears to be incomplete.")

    json_text = text[start:end]

    # In case Python-style booleans/None appear
    json_text = (
        json_text
        .replace("True", "true")
        .replace("False", "false")
        .replace("None", "null")
    )

    return json.loads(json_text)


def fmt(value, decimals: int = 4) -> str:
    """Format floats compactly for markdown tables."""
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def make_metric_table(title: str, metrics: dict, keys: list[str]) -> str:
    lines = [
        f"### {title}",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]

    for key in keys:
        if key in metrics:
            lines.append(f"| {key} | {fmt(metrics[key])} |")

    return "\n".join(lines)


def make_ap_by_class_table(results: dict, decimals: int = 4) -> str:
    ap_by_class = results.get("AP_by_class", {})

    if not ap_by_class:
        return ""

    # Preserve class order from the first IoU threshold
    first_iou = next(iter(ap_by_class))
    classes = list(ap_by_class[first_iou].keys())

    lines = [
        "### AP by Class per IoU Threshold",
        "",
        "| IoU | " + " | ".join(classes) + " |",
        "|---:|" + "|".join(["---:"] * len(classes)) + "|",
    ]

    for iou_raw, class_values in ap_by_class.items():
        iou = round(float(iou_raw), 2)
        row = [f"{iou:.2f}"]

        for cls in classes:
            value = class_values.get(cls)
            row.append("—" if value is None else fmt(value, decimals))

        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def evaluation_to_markdown(
    results: dict,
    title: str = "Evaluation Results",
    decimals: int = 4,
) -> str:
    ap_keys = [
        "AP",
        "AP50",
        "AP75",
        "APeasy",
        "APmoderate",
        "APhard",
    ]

    recall_keys = [
        "AReasy",
        "ARmoderate",
        "ARhard",
        "AR1",
        "AR10",
        "AR100",
        "AR200",
    ]

    sections = [
        f"## {title}",
        "",
        make_metric_table("Overall AP Metrics", results, ap_keys),
        "",
        make_metric_table("Recall Metrics", results, recall_keys),
        "",
        make_ap_by_class_table(results, decimals),
    ]

    return "\n".join(section for section in sections if section.strip())


def convert_eval_text_to_markdown(
    input_text: str,
    title: str = "Evaluation Results",
) -> str:
    results = extract_json_from_text(input_text)
    return evaluation_to_markdown(results, title=title)


if __name__ == "__main__":
    input_path = Path("eval_output.txt")
    output_path = Path("eval_output.md")

    raw_text = input_path.read_text(encoding="utf-8")

    markdown = convert_eval_text_to_markdown(
        raw_text,
        title="Evaluation Results: EdgeCrafter + SAHI Inference",
    )

    output_path.write_text(markdown, encoding="utf-8")

    print(markdown)