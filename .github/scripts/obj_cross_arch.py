"""Fork-only OBJ validation; never substitute this for the full upstream CI."""

import json
import os
import platform
import struct
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

BASE_SHA = "dc8490c825c382d2d58b5bd87ecacc74790ead27"
REVISION_SHA = "ba44a70bb545f1b1ce09d6a2572548a48d63c393"

# Each phase uses a fresh process. Import the intended package before pytest adds
# the test directory to sys.path, and verify that no installed wheel is tested.
WORKER = """
import json, pathlib, sys
source = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source))
import trimesh
actual = pathlib.Path(trimesh.__file__).resolve()
assert actual == source / 'trimesh' / '__init__.py', actual
print(json.dumps({'imported_trimesh': str(actual)}), flush=True)
import pytest
raise SystemExit(pytest.main(sys.argv[2:]))
"""


def run_phase(root, label, source, tests, selection, expected):
    output = root / "results"
    xml = output / f"{label}.xml"
    command = [
        sys.executable,
        "-c",
        WORKER,
        str(source),
        str(tests / "tests/test_obj.py"),
        "-v",
        "--tb=short",
        f"--junitxml={xml}",
    ]
    if selection:
        command.extend(["-k", selection])
    started = time.monotonic()
    result = {"phase": label, "expected": expected, "verified": False}
    try:
        with (output / f"{label}.log").open("w") as log:
            process = subprocess.run(
                command,
                cwd=source,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=900,
                check=False,
            )
        result["exit_code"] = process.returncode
        cases = list(ET.parse(xml).iter("testcase"))
        failures = [case for case in cases if case.find("failure") is not None]
        errors = [case for case in cases if case.find("error") is not None]
        skipped = [case for case in cases if case.find("skipped") is not None]
        observed = {
            "tests": len(cases),
            "failures": len(failures),
            "errors": len(errors),
            "skipped": len(skipped),
        }
        result["observed"] = observed
        result["failed_tests"] = [case.get("name") for case in failures]
        result["verified"] = observed == expected and process.returncode == (
            1 if expected["failures"] else 0
        )
        if expected["failures"]:
            # Expected red is only accepted for the known regression cases and
            # assertion failures, never dependency/collection/import errors.
            known = all(
                case.get("name", "").startswith("test_obj_interleaved_relative_indices")
                and (
                    "AssertionError" in case.find("failure").get("message", "")
                    or "DID NOT RAISE" in case.find("failure").get("message", "")
                )
                for case in failures
            )
            passing = [case.get("name") for case in cases if case not in failures]
            result["known_regressions_only"] = known
            result["verified"] &= known and passing == [
                "test_obj_interleaved_relative_indices_mixed_faces[f\\t1 2 3]"
            ]
    except (subprocess.TimeoutExpired, OSError, ET.ParseError) as error:
        result["error"] = str(error)
    result["seconds"] = time.monotonic() - started
    (output / f"{label}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)  # noqa: T201
    return result


def main():
    root = Path(sys.argv[1]).resolve()
    target = sys.argv[2]
    output = root / "results"
    output.mkdir(parents=True, exist_ok=True)
    environment = {
        "target": target,
        "machine": platform.machine(),
        "python": sys.version,
        "numpy": np.__version__,
        "pytest": pytest.__version__,
        "byteorder": sys.byteorder,
        "pointer_bits": struct.calcsize("P") * 8,
        "intp_bits": np.dtype(np.intp).itemsize * 8,
        "base_sha": BASE_SHA,
        "revision_sha": REVISION_SHA,
        "thread_limits": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    print(json.dumps(environment, indent=2), flush=True)  # noqa: T201
    if target != "native":
        actual_arch = subprocess.check_output(
            ["dpkg", "--print-architecture"], text=True
        ).strip()
        assert actual_arch == target, (actual_arch, target)
        endian, bits = {"s390x": ("big", 64), "i386": ("little", 32)}[target]
        assert environment["byteorder"] == endian, environment
        assert environment["pointer_bits"] == environment["intp_bits"] == bits, (
            environment
        )

    base, revision = root / "base", root / "revision"
    phases = [
        ("base-original", base, base, "", 38, 0),
        ("base-regressions", base, revision, "interleaved_relative_indices", 39, 38),
        ("revision-all-obj", revision, revision, "", 77, 0),
    ]
    results = [
        run_phase(
            root,
            label,
            source,
            tests,
            selection,
            {"tests": count, "failures": failures, "errors": 0, "skipped": 0},
        )
        for label, source, tests, selection, count, failures in phases
    ]
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = [
        f"## OBJ comparison: {target}",
        "",
        f"Base `{BASE_SHA}`; revision `{REVISION_SHA}`.",
        "",
        "| Phase | Observed | Matches expected |",
        "| --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            f"| {result['phase']} | {result.get('observed', result.get('error'))} "
            + f"| {result['verified']} |"
        )
    lines.extend(
        [
            "",
            "Baseline regression failures are intentional and checked for assertion failures.",
            "This runs only OBJ tests, with one numerical-library thread.",
            "The workflow configures QEMU for architecture-specific containers;",
            "i386 can also use the x86-64 host's native 32-bit compatibility support.",
            "It does not complete or replace the full upstream architecture suite.",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return 0 if all(result["verified"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
