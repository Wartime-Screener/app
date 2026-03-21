"""
Universe loader for managing ticker universes.

Each universe is a CSV file in config/universes/ with columns:
ticker, company_name, segment, sub_segment.
"""

import logging
from pathlib import Path

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
UNIVERSES_DIR = PROJECT_ROOT / "config" / "universes"


def list_universes() -> list[str]:
    """
    List all available universe names (without .csv extension).

    Returns:
        Sorted list of universe names, e.g. ['airlines', 'coal', 'oil_and_gas'].
    """
    if not UNIVERSES_DIR.exists():
        logger.warning(f"Universes directory not found: {UNIVERSES_DIR}")
        return []
    return sorted(p.stem for p in UNIVERSES_DIR.glob("*.csv"))


def load_universe(name: str) -> pd.DataFrame:
    """
    Load a single universe CSV by name.

    Args:
        name: Universe name (e.g. 'oil_and_gas'). The .csv extension is optional.

    Returns:
        DataFrame with columns: ticker, company_name, segment, sub_segment.
        Returns empty DataFrame if the file does not exist.
    """
    clean_name = name.replace(".csv", "")
    csv_path = UNIVERSES_DIR / f"{clean_name}.csv"

    if not csv_path.exists():
        logger.error(f"Universe file not found: {csv_path}")
        return pd.DataFrame(columns=["ticker", "company_name", "segment", "sub_segment"])

    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        df["ticker"] = df["ticker"].str.strip().str.upper()
        logger.info(f"Loaded universe '{clean_name}' with {len(df)} tickers")
        return df
    except Exception as e:
        logger.error(f"Error loading universe '{clean_name}': {e}")
        return pd.DataFrame(columns=["ticker", "company_name", "segment", "sub_segment"])


@st.cache_data(ttl=300)
def load_all_universes() -> pd.DataFrame:
    """
    Load all universe CSVs and combine them with a 'universe' column.

    Returns:
        Combined DataFrame with an added 'universe' column indicating the source file.
    """
    frames = []
    for name in list_universes():
        df = load_universe(name)
        if not df.empty:
            df["universe"] = name
            frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=["ticker", "company_name", "segment", "sub_segment", "universe"]
        )

    combined = pd.concat(frames, ignore_index=True)
    # Drop duplicates by ticker, keep first occurrence
    combined = combined.drop_duplicates(subset="ticker", keep="first")
    logger.info(f"Loaded {len(combined)} total tickers across {len(frames)} universes")
    return combined
