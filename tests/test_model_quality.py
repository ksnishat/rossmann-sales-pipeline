"""Model-quality regression tests.

The previous forecaster passed every functional test while being useless for the
question it was built to answer: asked for sales per store, it returned the same
figure for 443 of 1,115 stores. Nothing failed, because nothing asserted that
different stores produce different predictions.

These tests assert the property that actually matters — the model discriminates
between stores — against the promoted artifact.
"""

import warnings

import pytest

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def model_and_columns():
    joblib = pytest.importorskip("joblib")
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    mp = root / "models" / "production_model.pkl"
    cp = root / "models" / "model_columns.pkl"
    if not mp.exists() or not cp.exists():
        pytest.skip("promoted model artifacts not present")
    return joblib.load(mp), list(joblib.load(cp))


def _predict(model, columns, store):
    pd = pytest.importorskip("pandas")
    row = {
        "Store": store,
        "DayOfWeek": 4,
        "Promo": 1,
        "SchoolHoliday": 0,
        "StoreType": "a",
        "Assortment": "a",
        "CompetitionDistance": 1270.0,
        "Promo2": 1,
    }
    df = pd.DataFrame([row])
    df["Open"] = 1
    featured = pd.get_dummies(df).reindex(columns=columns, fill_value=0)
    return float(model.predict(featured)[0])


class TestStoreDiscrimination:
    def test_different_stores_produce_different_predictions(self, model_and_columns):
        """The bug: stores 1..400 all returned the same value.

        A model that ignores Store is not a store-level forecaster, however
        good its aggregate MAE looks.
        """
        model, columns = model_and_columns
        preds = {s: _predict(model, columns, s) for s in (1, 20, 100, 300, 700, 1000)}
        distinct = len({round(v, 2) for v in preds.values()})
        assert distinct >= 4, (
            f"only {distinct} distinct predictions across 6 stores: {preds}. "
            "The model is not using the Store feature."
        )

    def test_prediction_spread_is_the_right_order_of_magnitude(self, model_and_columns):
        """Real mean sales vary by ~EUR 19,000 across all stores.

        The threshold is measured, not guessed. With StoreType/Assortment held
        at "a" and the same competitor distance, stores 1-200 span EUR 2,608
        (min 4,905, max 7,512) under the current model. The previous forest
        spanned EUR 0 across that range and EUR 684 across all 1,115 stores, so
        anything above ~EUR 2,000 is evidence the model is separating stores
        rather than returning the population mean.
        """
        model, columns = model_and_columns
        preds = [_predict(model, columns, s) for s in range(1, 201)]
        spread = max(preds) - min(preds)
        assert spread > 2000, (
            f"prediction spread across stores 1-200 is only EUR {spread:.0f}; "
            "the model has collapsed toward the population mean"
        )

    def test_predictions_are_positive_and_plausible(self, model_and_columns):
        model, columns = model_and_columns
        preds = [_predict(model, columns, s) for s in (1, 500, 1115)]
        assert all(100 < p < 30000 for p in preds), preds
