"""
CLI Wrapper that  runs the Railbench formatting check and the object evaluation.


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


class BenchmarkCalculator:
    def __init__(self, test_pipeline, project_root=None):
        if project_root is None:
            self.project_root = Path(__file__).resolve().parents[1]
        if test_pipeline:
            self.gt_path = self.project_root / "data" / "pipeline_test" / "annotations_pipeline_test.json"
        else:
            self.gt_path = self.project_root / "data" /  "coco_annotations" / "objects" / "annotations_val.json"

        self.pred_path = self.project_root / "outputs" / "pred_test.json"
        self._check_paths()


    def _check_paths(self):
        if not self.gt_path.exists():
            raise FileNotFoundError(f"Ground-truth file not found: {self.gt_path}")
        if not self.pred_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {self.pred_path}")

    def run_railbench_format_check(self,
        is_railbench_test: bool = False,
    ) -> int:
        """
        Run the external RAIL-BENCH formatting check on a prediction file.

        Parameters
        - is_railbench_test: whether to pass the `--is_railbench_test` flag to
          the external checker.
        Returns
        - exit code (int) from running the external script.
        """

        external_script = (
            self.project_root
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

        try:
            with open(self.pred_path, "r", encoding="utf-8") as f:
                predictions = json.load(f)
            if not isinstance(predictions, list):
                raise ValueError(
                    "Prediction file must be a JSON list. "
                    f"Got {type(predictions).__name__} in {self.pred_path}. "
                )
        except json.JSONDecodeError as e:
            raise ValueError(f"Prediction file is not valid JSON: {self.pred_path}\n{e}") from e

        cmd = [sys.executable, str(external_script), "--pred_file", str(self.pred_path)]
        if is_railbench_test:
            cmd.append("--is_railbench_test")

        result = subprocess.run(cmd, cwd=external_cwd, env=env)
        if result.returncode != 0:
            print(f"Format check failed with exit code {result.returncode}")

        return result.returncode


    def prepare_and_run_object_eval(self,
        split: str = "val",
        detector_name: str = "my_detector",
        project_name: str = "my_project",
    ) -> int:
        """
        Prepare a RAIL-BENCH project folder and run the object evaluation.

        Parameters:
        - split: dataset split name (e.g. 'val' or 'test') used in filenames
        - detector_name: subfolder name under detectors/ for this model
        - project_name: name of the project folder created under Benchmarks/RAILBENCH_Object/data/
        - overwrite: whether to overwrite existing files during copy and pass --overwrite to the evaluator
        Returns:
        - exit code from the external run_object_eval.py process
        """


        # External bench root
        bench_root = (
            self.project_root
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

        # Copy files)
        shutil.copy2(self.gt_path, gt_dest)
        print(f"Copied GT {self.gt_path} -> {gt_dest}")

        shutil.copy2(self.pred_path, pred_dest)
        print(f"Copied pred {self.pred_path} -> {pred_dest}")

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

        cmd = [sys.executable, str(external_script), "--project", project_name, "--overwrite"]

        result = subprocess.run(cmd, cwd=bench_root, env=env)

        result_path = data_project_dir / "results" / detector_name / "eval_val.json"


        print(f"-- Result of the RAIL-BENCH Object evaluation (see {result_path}) : --")
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
        help=f"Path to prediction file",
    )
    parser.add_argument(
        "--is_railbench_test",
        action="store_true",
        help="Pass this flag if the predictions are for the RAIL-BENCH Object test set.",
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
        "--test_pipeline",
        action="store_true",
        help="Use the pipeline test GT and pred files instead of the default val set.",
    )

    args = parser.parse_args(argv)
    benchmark_calculator = BenchmarkCalculator(test_pipeline=False)

    format_rc = benchmark_calculator.run_railbench_format_check(
        is_railbench_test=args.is_railbench_test)

    if format_rc != 0:
        return format_rc

    return benchmark_calculator.prepare_and_run_object_eval(
        split=args.split,
        detector_name=args.detector_name,
        project_name=args.project_name,
    )
if __name__ == "__main__":
    raise SystemExit(main())
