#!/usr/bin/env python3
"""
CLI-Wrapper, der den externen RAIL-BENCH-Format-Check aufruft
und danach optional die Object-Evaluation startet.

Verhalten:
- Bietet die Option --pred_path (Standard: outputs/pred_test.json)
- Übersetzt die Option zu --pred_file für das externen Skript
- Optional: --is_railbench_test wird an das externen Skript weitergereicht
- Führt das externes Skript im Verzeichnis seiner Datei aus, damit relative Pfade dort aufgelöst werden

Diese Datei stellt jetzt die Funktion `run_railbench_format_check` zur Verfügung,
so dass andere Skripte sie per Import verwenden können. Der CLI-Entrypoint
ruft weiterhin `main()` und verwendet die Funktion intern.

Zusätzlich enthält die Datei jetzt `prepare_and_run_object_eval`,
welches ein Projektverzeichnis unter Benchmarks/RAILBENCH_Object/data/
erzeugt, GT- und Prediction-Dateien dort ablegt und das externe
`run_object_eval.py` startet.
"""
from __future__ import annotations
import argparse
import sys
import os
import json
from pathlib import Path
import subprocess
from typing import Optional, Union
import shutil

from jedi.api import project

TEST_PIPELINE = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if TEST_PIPELINE:
    DEFAULT_GT_PATH = Path("data") / "pipeline_test" / "annotations_pipeline_test.json"
else:
    DEFAULT_GT_PATH = Path("data") / "coco_annotations" / "objects" / "annotations_val.json"
DEFAULT_PRED_TEST = Path("outputs") / "pred_test.json"


def run_railbench_format_check(
    pred_path: Optional[Union[str, Path]] = None,
    is_railbench_test: bool = False,
    show_external_help: bool = False,
    project_root: Optional[Union[str, Path]] = None,
) -> int:
    """
    Run the external RAIL-BENCH formatting check on a prediction file.

    Parameters
    - pred_path: path to prediction JSON file. If None, uses the default
      outputs/pred_test.json under project_root.
    - is_railbench_test: whether to pass the `--is_railbench_test` flag to
      the external checker.
    - show_external_help: if True, only show the external script's help and
      return its exit code.
    - project_root: optional override for the project root. If None, the
      directory two levels above this file is used.

    Returns
    - exit code (int) from running the external script.

    Raises
    - FileNotFoundError: if the external `check_formatting.py` cannot be found.
    """
    proj_root = Path(project_root) if project_root is not None else PROJECT_ROOT

    external_script = (
        proj_root
        / "external"
        / "RAIL-BENCH_toolkit"
        / "Benchmarks"
        / "RAILBENCH_Object"
        / "check_formatting.py"
    )

    if not external_script.exists():
        raise FileNotFoundError(f"externes Skript nicht gefunden: {external_script}")

    external_cwd = external_script.parent

    # Ensure the external toolkit root is on PYTHONPATH so imports like
    # `utils.helpers` work when the external script runs.
    external_toolkit_root = external_script.parent.parent.parent
    env = os.environ.copy()
    prev_pp = env.get("PYTHONPATH", "")
    if prev_pp:
        env["PYTHONPATH"] = str(external_toolkit_root) + os.pathsep + prev_pp
    else:
        env["PYTHONPATH"] = str(external_toolkit_root)

    # Resolve prediction path
    if pred_path is None:
        pred_path = proj_root / DEFAULT_PRED_TEST
    pred_path = Path(pred_path) if not isinstance(pred_path, Path) else pred_path
    pred_path = pred_path if pred_path.is_absolute() else (proj_root / pred_path).resolve()

    if show_external_help:
        cmd = [sys.executable, str(external_script), "-h"]
        result = subprocess.run(cmd, cwd=external_cwd, env=env)
        return result.returncode

    if not pred_path.exists():
        # warn but continue so the external script can give more details
        print(f"Warnung: Vorhergesagte Datei nicht gefunden: {pred_path}", file=sys.stderr)
    else:
        # Early, user-friendly check: the external script expects a list of predictions.
        try:
            with open(pred_path, "r", encoding="utf-8") as f:
                predictions = json.load(f)
            if not isinstance(predictions, list):
                raise ValueError(
                    "Prediction file must be a JSON list. "
                    f"Got {type(predictions).__name__} in {pred_path}. "
                    "Did you accidentally pass a GT annotation file?"
                )
        except json.JSONDecodeError as e:
            raise ValueError(f"Prediction file is not valid JSON: {pred_path}\n{e}") from e

    cmd = [sys.executable, str(external_script), "--pred_file", str(pred_path)]
    if is_railbench_test:
        cmd.append("--is_railbench_test")

    result = subprocess.run(cmd, cwd=external_cwd, env=env)
    return result.returncode


def prepare_and_run_object_eval(
    gt_path: Optional[Union[str, Path]] = None,
    pred_path: Optional[Union[str, Path]] = None,
    split: str = "val",
    detector_name: str = "my_detector",
    project_name: str = "my_project",
    overwrite: bool = False,
    project_root: Optional[Union[str, Path]] = None,
) -> int:
    """
    Prepare a RAIL-BENCH project folder and run the object evaluation.

    Steps performed:
    - Create Benchmarks/RAILBENCH_Object/data/<project_name>/annotations and detectors/<detector_name>
    - Copy GT file to annotations/annotations_<split>.json
    - Copy prediction file to detectors/<detector_name>/pred_<split>.json
    - Execute run_object_eval.py with --project <project_name> (and --overwrite if set)

    Parameters:
    - gt_path: path to ground-truth JSON. If None, defaults to data/coco_annotations/objects/annotations_val.json
    - pred_path: path to prediction JSON. If None, defaults to outputs/pred_test.json
    - split: dataset split name (e.g. 'val' or 'test') used in filenames
    - detector_name: subfolder name under detectors/ for this model
    - project_name: name of the project folder created under Benchmarks/RAILBENCH_Object/data/
    - overwrite: whether to overwrite existing files during copy and pass --overwrite to the evaluator
    - project_root: optional project root override

    Returns:
    - exit code from the external run_object_eval.py process
    """
    proj_root = Path(project_root) if project_root is not None else PROJECT_ROOT

    # Resolve defaults
    if gt_path is None:
        gt_path = proj_root / DEFAULT_GT_PATH
    if pred_path is None:
        pred_path = proj_root / DEFAULT_PRED_TEST

    gt_path = Path(gt_path)
    pred_path = Path(pred_path)

    # Validate source files exist before copying
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    # External bench root
    bench_root = (
        proj_root
        / "external"
        / "RAIL-BENCH_toolkit"
        / "Benchmarks"
        / "RAILBENCH_Object"
    )

    if not bench_root.exists():
        raise FileNotFoundError(f"Benchmarks root not found: {bench_root}")

    data_project_dir = bench_root / "data" / project_name
    annotations_dir = data_project_dir / "annotations"
    detectors_dir = data_project_dir / "detectors" / detector_name

    # Remove project dir if it exists
    if data_project_dir.exists():
        shutil.rmtree(data_project_dir)
    os.makedirs(data_project_dir, exist_ok=True)

    # Create directories
    annotations_dir.mkdir(parents=True, exist_ok=True)
    detectors_dir.mkdir(parents=True, exist_ok=True)

    # Destination filenames
    gt_dest = annotations_dir / f"annotations_{split}.json"
    pred_dest = detectors_dir / f"pred_{split}.json"

    # Copy files (respect overwrite flag)
    if gt_dest.exists() and not overwrite:
        print(f"GT destination {gt_dest} already exists and overwrite=False. Skipping copy.")
    else:
        shutil.copy2(gt_path, gt_dest)
        print(f"Copied GT {gt_path} -> {gt_dest}")

    if pred_dest.exists() and not overwrite:
        print(f"Prediction destination {pred_dest} already exists and overwrite=False. Skipping copy.")
    else:
        shutil.copy2(pred_path, pred_dest)
        print(f"Copied pred {pred_path} -> {pred_dest}")

    # Prepare environment and call run_object_eval.py
    external_script = bench_root / "run_object_eval.py"
    if not external_script.exists():
        raise FileNotFoundError(f"externes run_object_eval.py nicht gefunden: {external_script}")

    # The toolkit root (contains 'utils' package) is two levels up from
    # RAILBENCH_Object (i.e., .../RAIL-BENCH_toolkit)
    external_toolkit_root = bench_root.parent.parent
    env = os.environ.copy()
    prev_pp = env.get("PYTHONPATH", "")
    if prev_pp:
        env["PYTHONPATH"] = str(external_toolkit_root) + os.pathsep + prev_pp
    else:
        env["PYTHONPATH"] = str(external_toolkit_root)

    cmd = [sys.executable, str(external_script), "--project", project_name]
    if overwrite:
        cmd.append("--overwrite")

    result = subprocess.run(cmd, cwd=bench_root, env=env)

    result_path = data_project_dir / "results" / detector_name / "eval_val.json"

    with open(result_path, "r", encoding="utf-8") as f:
            print(f"Result from run_object_eval.py: {f.read()}")

    return result.returncode


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="calculate_railbench_benchmark",
        description="Wrapper to run RAIL-BENCH formatting checks on prediction files.",
    )
    parser.add_argument(
        "--pred_path",
        type=Path,
        default=(PROJECT_ROOT / DEFAULT_PRED_TEST),
        help=f"Path to prediction file (default: {PROJECT_ROOT / DEFAULT_PRED_TEST})",
    )
    parser.add_argument(
        "--is_railbench_test",
        action="store_true",
        help="Pass this flag if the predictions are for the RAIL-BENCH Object test set.",
    )
    parser.add_argument(
        "--show_external_help",
        action="store_true",
        help="If set, show the help of the external check_formatting.py instead of running it.",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Dataset split used for evaluation filenames (default: val).",
    )
    parser.add_argument(
        "--detector_name",
        default="my_detector",
        help="Detector subfolder name for RAIL-BENCH eval data (default: my_detector).",
    )
    parser.add_argument(
        "--project_name",
        default="my_project",
        help="Project folder name under RAILBENCH_Object/data (default: my_project).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing copied GT/pred files and pass --overwrite to run_object_eval.py.",
    )

    args = parser.parse_args(argv)

    try:
        format_rc = run_railbench_format_check(
            pred_path=args.pred_path,
            is_railbench_test=args.is_railbench_test,
            show_external_help=args.show_external_help,
            project_root=PROJECT_ROOT,
        )

        if args.show_external_help:
            return format_rc

        if format_rc != 0:
            print(
                f"Format check failed with exit code {format_rc}. "
                "Skipping object evaluation.",
                file=sys.stderr,
            )
            return format_rc

        return prepare_and_run_object_eval(
            gt_path=PROJECT_ROOT / DEFAULT_GT_PATH,
            pred_path=args.pred_path,
            split=args.split,
            detector_name=args.detector_name,
            project_name=args.project_name,
            overwrite=args.overwrite,
            project_root=PROJECT_ROOT,
        )
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
