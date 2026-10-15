"""Build the ``train_cleaned`` feature table the training script reads.

``train_model.py`` reads ``SELECT * FROM train_cleaned``, but nothing in the
repository created that table: the DVC pipeline writes CSV files while training
reads Postgres, so a fresh clone could not retrain. This script closes that gap.

It joins the raw ``train`` and ``store`` tables and applies the same filters the
model was trained with:

* ``Open = 1``  -- closed stores have no sales to predict.
* ``Sales > 0`` -- zero-sales rows are closed-store artefacts, not real demand.

Run after ``load_data.py``::

    python src/scripts/build_feature_store.py
"""

import os

import pandas as pd
from sqlalchemy import create_engine

DEFAULT_DB_URL = "postgresql://rossmann:changeme123@localhost:5433/rossmann"

# Columns the model expects, in the order it was trained on. Kept explicit so a
# schema change fails loudly here rather than silently shifting feature order.
FEATURE_COLUMNS = [
    "Store",
    "DayOfWeek",
    "Date",
    "Sales",
    "Customers",
    "Open",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "Promo2",
]


def build_feature_store(db_url: str | None = None, table: str = "train_cleaned") -> int:
    """Create ``train_cleaned`` from ``train`` joined with ``store``.

    Returns the number of rows written.
    """
    engine = create_engine(db_url or os.getenv("DATABASE_URL", DEFAULT_DB_URL))

    query = """
        SELECT t."Store", t."DayOfWeek", t."Date", t."Sales", t."Customers",
               t."Open", t."Promo", t."StateHoliday", t."SchoolHoliday",
               s."StoreType", s."Assortment", s."CompetitionDistance", s."Promo2"
        FROM train t
        JOIN store s ON t."Store" = s."Store"
        WHERE t."Open" = 1 AND t."Sales" > 0
    """
    df = pd.read_sql(query, engine)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"feature store is missing expected columns: {missing}")

    df = df[FEATURE_COLUMNS]

    # CompetitionDistance is NULL for stores with no nearby competitor. Random
    # Forest rejects NaN outright ("Input X contains NaN"), so it must be filled.
    # A large sentinel is the honest encoding: "no competitor within range" is
    # not the same as "competitor at distance 0", which is what a 0 fill would
    # imply and would distort the feature.
    if df["CompetitionDistance"].isna().any():
        n_null = int(df["CompetitionDistance"].isna().sum())
        sentinel = float(df["CompetitionDistance"].max()) * 2
        df["CompetitionDistance"] = df["CompetitionDistance"].fillna(sentinel)
        print(
            f"Filled {n_null} missing CompetitionDistance values with {sentinel:.0f} "
            "(no competitor within range)"
        )

    df.to_sql(table, engine, if_exists="replace", index=False, chunksize=10000)
    print(f"Built '{table}' with {len(df)} rows and {len(df.columns)} columns")
    return len(df)


if __name__ == "__main__":
    build_feature_store()
