"""Tests for the data quality validation module."""
import pytest
import pandas as pd
import tempfile
from pathlib import Path
from src.scripts.validate_data import (
    validate_dataset,
    ValidationResult,
    CheckResult,
    CheckType,
    ROSSMANN_SCHEMA,
)


@pytest.fixture
def valid_dataframe():
    """Create a valid Rossmann-style DataFrame."""
    return pd.DataFrame({
        "Store": [1, 2, 3],
        "DayOfWeek": [1, 2, 3],
        "Date": ["2015-01-01", "2015-01-02", "2015-01-03"],
        "Sales": [5000, 6000, 7000],
        "Customers": [500, 600, 700],
        "Promo": [0, 1, 0],
        "SchoolHoliday": [0, 0, 1],
        "StoreType": ["a", "b", "c"],
        "Assortment": ["a", "b", "c"],
        "CompetitionDistance": [1000.0, 2000.0, 3000.0],
        "CompetitionOpenSinceMonth": [1, 2, 3],
        "CompetitionOpenSinceYear": [2013, 2013, 2013],
        "Promo2": [0, 0, 0],
        "Promo2SinceWeek": [0, 0, 0],
        "Promo2SinceYear": [0, 0, 0],
        "PromoInterval": ["", "", ""],
    })


@pytest.fixture
def invalid_dataframe():
    """Create an invalid DataFrame with various issues."""
    return pd.DataFrame({
        "Store": [1, 2, 3],
        "DayOfWeek": [1, 2, 8],  # Invalid: 8 > 7
        "Date": ["2015-01-01", "2015-01-02", "invalid"],  # Invalid date
        "Sales": [5000, 6000, -100],  # Invalid: negative
        "Customers": [500, 600, 700],
        "Promo": [0, 1, 0],
        "SchoolHoliday": [0, 0, 1],
        "StoreType": ["a", "b", "x"],  # Invalid: 'x' not in valid values
        "Assortment": ["a", "b", "c"],
        "CompetitionDistance": [1000.0, 2000.0, 3000.0],
        "CompetitionOpenSinceMonth": [1, 2, 3],
        "CompetitionOpenSinceYear": [2013, 2013, 2013],
        "Promo2": [0, 0, 0],
        "Promo2SinceWeek": [0, 0, 0],
        "Promo2SinceYear": [0, 0, 0],
        "PromoInterval": ["", "", ""],
    })


@pytest.fixture
def valid_csv(tmp_path, valid_dataframe):
    """Write a valid DataFrame to CSV."""
    filepath = tmp_path / "valid.csv"
    valid_dataframe.to_csv(filepath, index=False)
    return filepath


@pytest.fixture
def invalid_csv(tmp_path, invalid_dataframe):
    """Write an invalid DataFrame to CSV."""
    filepath = tmp_path / "invalid.csv"
    invalid_dataframe.to_csv(filepath, index=False)
    return filepath


class TestValidationResult:
    """Test the ValidationResult dataclass."""

    def test_pass_rate_all_passed(self):
        """Pass rate should be 100% when all checks pass."""
        result = ValidationResult(
            dataset_name="test",
            total_rows=100,
            total_columns=5,
            checks=[
                CheckResult("check1", CheckType.COMPLETENESS, True, "OK"),
                CheckResult("check2", CheckType.VALIDITY, True, "OK"),
            ],
        )
        assert result.pass_rate == 100.0
        assert result.passed is True

    def test_pass_rate_some_failed(self):
        """Pass rate should reflect failed checks."""
        result = ValidationResult(
            dataset_name="test",
            total_rows=100,
            total_columns=5,
            checks=[
                CheckResult("check1", CheckType.COMPLETENESS, True, "OK"),
                CheckResult("check2", CheckType.VALIDITY, False, "Failed"),
                CheckResult("check3", CheckType.VALIDITY, False, "Failed"),
            ],
        )
        # Manually compute passed (as validate_dataset does)
        result.passed = all(bool(c.passed) for c in result.checks)
        assert result.pass_rate == pytest.approx(33.33, abs=0.1)
        assert result.passed is False

    def test_summary_output(self):
        """Summary should contain key information."""
        result = ValidationResult(
            dataset_name="test_dataset",
            total_rows=100,
            total_columns=5,
            checks=[
                CheckResult("check1", CheckType.COMPLETENESS, True, "All good"),
            ],
        )
        summary = result.summary()
        assert "test_dataset" in summary
        assert "100" in summary
        assert "✅" in summary


class TestValidateDataset:
    """Test the validate_dataset function."""

    def test_valid_dataset_passes(self, valid_csv):
        """Valid dataset should pass all checks."""
        result = validate_dataset(str(valid_csv), "valid_test")
        assert result.passed is True
        assert result.total_rows == 3
        assert result.total_columns == 16

    def test_invalid_dataset_fails(self, invalid_csv):
        """Invalid dataset should fail some checks."""
        result = validate_dataset(str(invalid_csv), "invalid_test")
        assert result.passed is False
        assert result.pass_rate < 100.0

    def test_missing_columns_detected(self, tmp_path):
        """Missing required columns should be detected."""
        df = pd.DataFrame({"Store": [1], "Sales": [5000]})
        filepath = tmp_path / "missing_cols.csv"
        df.to_csv(filepath, index=False)

        result = validate_dataset(str(filepath))
        assert result.passed is False
        # Should have a check about missing columns
        missing_check = [c for c in result.checks if "Required columns" in c.check_name]
        assert len(missing_check) == 1
        assert missing_check[0].passed is False

    def test_duplicate_rows_detected(self, tmp_path, valid_dataframe):
        """Duplicate rows should be detected."""
        df = pd.concat([valid_dataframe, valid_dataframe], ignore_index=True)
        filepath = tmp_path / "duplicates.csv"
        df.to_csv(filepath, index=False)

        result = validate_dataset(str(filepath))
        dup_check = [c for c in result.checks if "duplicate" in c.check_name.lower()]
        assert len(dup_check) == 1
        assert bool(dup_check[0].passed) is False
        assert dup_check[0].failed_rows == 3  # 3 duplicate rows

    def test_value_range_violations(self, tmp_path, valid_dataframe):
        """Out-of-range values should be detected."""
        df = valid_dataframe.copy()
        df.loc[0, "Sales"] = -100  # Invalid: negative sales
        df.loc[1, "DayOfWeek"] = 8  # Invalid: > 7

        filepath = tmp_path / "range_violation.csv"
        df.to_csv(filepath, index=False)

        result = validate_dataset(str(filepath))
        range_checks = [c for c in result.checks if "range" in c.check_name.lower()]
        assert len(range_checks) >= 2
        assert any(not c.passed for c in range_checks)


class TestSchemaDefinition:
    """Test the schema definition."""

    def test_required_columns_complete(self):
        """Schema should have all expected required columns."""
        expected = {"Store", "DayOfWeek", "Date", "Sales", "Customers",
                    "Promo", "SchoolHoliday", "StoreType", "Assortment",
                    "CompetitionDistance"}
        assert expected.issubset(set(ROSSMANN_SCHEMA["required_columns"]))

    def test_column_ranges_valid(self):
        """Column ranges should have valid min < max."""
        for col, (min_val, max_val) in ROSSMANN_SCHEMA["column_ranges"].items():
            assert min_val < max_val, f"Invalid range for {col}: {min_val} >= {max_val}"

    def test_categorical_values_valid(self):
        """Categorical values should be non-empty lists."""
        for col, values in ROSSMANN_SCHEMA["categorical_values"].items():
            assert isinstance(values, list)
            assert len(values) > 0
