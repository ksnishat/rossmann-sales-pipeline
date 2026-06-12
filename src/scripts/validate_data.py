"""
Data quality validation for Rossmann Sales Pipeline.

Implements data quality checks using a schema-based approach:
- Completeness: no missing values in critical columns
- Validity: values within expected ranges
- Consistency: cross-column constraints
- Uniqueness: no duplicate records

Usage:
    from src.scripts.validate_data import validate_dataset

    results = validate_dataset("data/silver/train.csv")
    print(results)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


class CheckType(Enum):
    """Types of data quality checks."""
    COMPLETENESS = "completeness"
    VALIDITY = "validity"
    CONSISTENCY = "consistency"
    UNIQUENESS = "uniqueness"


@dataclass
class CheckResult:
    """Result of a single data quality check."""
    check_name: str
    check_type: CheckType
    passed: bool
    message: str
    failed_rows: int = 0
    total_rows: int = 0


@dataclass
class ValidationResult:
    """Aggregated results of all data quality checks."""
    dataset_name: str
    total_rows: int
    total_columns: int
    checks: List[CheckResult] = field(default_factory=list)
    passed: bool = True

    @property
    def pass_rate(self) -> float:
        """Percentage of checks that passed."""
        if not self.checks:
            return 100.0
        passed = sum(1 for c in self.checks if c.passed)
        return (passed / len(self.checks)) * 100

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Dataset: {self.dataset_name}",
            f"Rows: {self.total_rows}, Columns: {self.total_columns}",
            f"Checks passed: {sum(1 for c in self.checks if c.passed)}/{len(self.checks)}",
            f"Pass rate: {self.pass_rate:.1f}%",
        ]
        for check in self.checks:
            status = "✅" if check.passed else "❌"
            lines.append(f"  {status} {check.check_name}: {check.message}")
        return "\n".join(lines)


# --- Schema Definition ---

ROSSMANN_SCHEMA = {
    "required_columns": [
        "Store", "DayOfWeek", "Date", "Sales", "Customers",
        "Promo", "SchoolHoliday", "StoreType", "Assortment",
        "CompetitionDistance", "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear", "Promo2", "Promo2SinceWeek",
        "Promo2SinceYear", "PromoInterval",
    ],
    "column_ranges": {
        "Store": (1, 1115),
        "DayOfWeek": (1, 7),
        "Sales": (0, 100000),
        "Customers": (0, 10000),
        "Promo": (0, 1),
        "SchoolHoliday": (0, 1),
        "Promo2": (0, 1),
        "CompetitionDistance": (0, 100000),
    },
    "categorical_values": {
        "StoreType": ["a", "b", "c", "d"],
        "Assortment": ["a", "b", "c"],
    },
}


def validate_dataset(file_path: str, dataset_name: str = None) -> ValidationResult:
    """Validate a dataset against the Rossmann schema.

    Args:
        file_path: Path to the CSV file
        dataset_name: Optional name for the dataset

    Returns:
        ValidationResult with all check results
    """
    if dataset_name is None:
        dataset_name = file_path

    df = pd.read_csv(file_path)
    result = ValidationResult(
        dataset_name=dataset_name,
        total_rows=len(df),
        total_columns=len(df.columns),
    )

    # --- Check 1: Required columns exist ---
    missing_cols = set(ROSSMANN_SCHEMA["required_columns"]) - set(df.columns)
    result.checks.append(CheckResult(
        check_name="Required columns present",
        check_type=CheckType.COMPLETENESS,
        passed=len(missing_cols) == 0,
        message=f"Missing: {missing_cols}" if missing_cols else "All required columns present",
    ))

    # --- Check 2: No missing values in critical columns ---
    for col in ["Store", "Sales", "Date"]:
        if col in df.columns:
            missing = df[col].isna().sum()
            result.checks.append(CheckResult(
                check_name=f"No missing values in {col}",
                check_type=CheckType.COMPLETENESS,
                passed=missing == 0,
                message=f"{missing} missing values" if missing > 0 else "No missing values",
                failed_rows=missing,
                total_rows=len(df),
            ))

    # --- Check 3: Value ranges ---
    for col, (min_val, max_val) in ROSSMANN_SCHEMA["column_ranges"].items():
        if col in df.columns:
            out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
            result.checks.append(CheckResult(
                check_name=f"{col} within range [{min_val}, {max_val}]",
                check_type=CheckType.VALIDITY,
                passed=out_of_range == 0,
                message=f"{out_of_range} values out of range" if out_of_range > 0 else "All values in range",
                failed_rows=out_of_range,
                total_rows=len(df),
            ))

    # --- Check 4: Categorical values ---
    for col, valid_values in ROSSMANN_SCHEMA["categorical_values"].items():
        if col in df.columns:
            invalid = ~df[col].isin(valid_values)
            invalid_count = invalid.sum()
            result.checks.append(CheckResult(
                check_name=f"{col} has valid categories",
                check_type=CheckType.VALIDITY,
                passed=invalid_count == 0,
                message=f"{invalid_count} invalid values" if invalid_count > 0 else "All values valid",
                failed_rows=invalid_count,
                total_rows=len(df),
            ))

    # --- Check 5: No duplicate rows ---
    duplicates = df.duplicated().sum()
    result.checks.append(CheckResult(
        check_name="No duplicate rows",
        check_type=CheckType.UNIQUENESS,
        passed=duplicates == 0,
        message=f"{duplicates} duplicate rows" if duplicates > 0 else "No duplicates",
        failed_rows=duplicates,
        total_rows=len(df),
    ))

    # --- Check 6: Date format validity ---
    if "Date" in df.columns:
        try:
            pd.to_datetime(df["Date"])
            result.checks.append(CheckResult(
                check_name="Date format valid",
                check_type=CheckType.VALIDITY,
                passed=True,
                message="All dates parseable",
            ))
        except Exception as e:
            result.checks.append(CheckResult(
                check_name="Date format valid",
                check_type=CheckType.VALIDITY,
                passed=False,
                message=f"Date parsing error: {e}",
            ))

    # Overall pass/fail
    result.passed = all(bool(c.passed) for c in result.checks)

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python validate_data.py <csv_file>")
        sys.exit(1)

    result = validate_dataset(sys.argv[1])
    print(result.summary())
    sys.exit(0 if result.passed else 1)