#!/usr/bin/env python3
"""
Code quality checks for the job application sorter project.
"""

import subprocess
import sys
import re
from tqdm import tqdm


def run_command(command, description):
    """Run a command and return success status."""
    try:
        subprocess.run(
            command, shell=True, capture_output=True, text=True, check=True
        )
        return True, None
    except subprocess.CalledProcessError as e:
        error_info = {
            "description": description,
            "exit_code": e.returncode,
            "stdout": e.stdout,
            "stderr": e.stderr,
        }
        return False, error_info


def count_flake8_errors(stdout_output):
    """Count the number of flake8 error lines."""
    if not stdout_output:
        return 0
    # Count lines that contain error codes (like E302, E501, etc.)
    error_lines = [
        line
        for line in stdout_output.split("\n")
        if line.strip()
        and ":" in line
        and any(code in line for code in ["E", "W", "F"])
    ]
    return len(error_lines)


def extract_pytest_info(stdout_output, stderr_output):
    """
    Extract test failure count and coverage percentage from pytest output.
    """
    combined_output = (
        stdout_output + "\n" + stderr_output
        if stderr_output
        else stdout_output
    )

    # Extract test results
    test_failed_match = re.search(r"(\d+) failed", combined_output)
    test_passed_match = re.search(r"(\d+) passed", combined_output)
    test_count_match = re.search(r"collected (\d+) items?", combined_output)

    failed_count = int(test_failed_match.group(1)) if test_failed_match else 0
    passed_count = int(test_passed_match.group(1)) if test_passed_match else 0
    total_count = (
        int(test_count_match.group(1))
        if test_count_match
        else (failed_count + passed_count)
    )

    # Extract coverage percentage
    coverage_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", combined_output)
    coverage_percent = int(coverage_match.group(1)) if coverage_match else 0

    return failed_count, total_count, coverage_percent


def main():
    """Run all code quality checks."""

    # Define checks
    checks = [
        (
            "find . -name '*.py' -exec sed -i '' 's/[[:space:]]*$//' {} +",
            "Strip trailing whitespace",
        ),
        ("black . -l 79", "Black formatting"),
        ("flake8 src", "Flake8 linting"),
        (
            "python -m pytest --cov=src --cov-report=term-missing "
            "--cov-fail-under=80 --quiet -n 2 --cov-context=test",
            "Pytest with coverage",
        ),
    ]

    # Run each check with progress bar
    failures = []

    for command, description in tqdm(
        checks, desc="Code quality checks", unit="check"
    ):
        success, error_info = run_command(command, description)
        if not success:
            failures.append(error_info)

    # Final summary
    print(f"\n{'='*50}")
    print("FINAL SUMMARY")
    print("=" * 50)

    if not failures:
        print("🎉 All checks passed!")
        return 0
    else:
        print(f"💥 {len(failures)} check(s) failed:")
        for i, failure in enumerate(failures, 1):
            description = failure["description"]

            if description == "Flake8 linting":
                error_count = count_flake8_errors(failure["stdout"])
                print(f"\n{i}. {description} - {error_count} errors found")
            elif failure["description"] == "Pytest with coverage":
                failed_count, total_count, coverage_percent = (
                    extract_pytest_info(failure["stdout"], failure["stderr"])
                )
                print(
                    f"\n{i}. {description} - {failed_count}/{total_count} "
                    f"tests failed, {coverage_percent}% coverage"
                )
            else:
                print(
                    f"\n{i}. {description} (exit code: {failure['exit_code']})"
                )

        print(f"\n❌ {len(failures)} out of {len(checks)} checks failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
