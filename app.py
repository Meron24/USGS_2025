# app.py
# Streamlit app for manipulating, cleaning, analyzing, and exporting USGS water-quality CSVs

import io
import json
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="USGS Water Quality Toolkit", layout="wide")

# ----------------------------
# Constants & Column Definitions
# ----------------------------
DISPLAY_NAME = "USGS Water Quality Toolkit"
NUMCOL = "__RMV"  # internal numeric mirror for ResultMeasureValue
DT_HELPER = "__dt"

# Canonical column names we care about (case-insensitive mapping supported)
EXPECTED_COLS = [
    "OrganizationIdentifier",
    "OrganizationFormalName",
    "ActivityIdentifier",
    "ActivityTypeCode",
    "ActivityMediaName",
    "ActivityMediaSubdivisionName",
    "ActivityStartDate",
    "ActivityStartTime/Time",
    "ActivityStartTime/TimeZoneCode",
    "ActivityConductingOrganizationText",
    "MonitoringLocationIdentifier",
    "HydrologicCondition",
    "HydrologicEvent",
    "SampleCollectionMethod/MethodIdentifier",
    "SampleCollectionMethod/MethodIdentifierContext",
    "SampleCollectionMethod/MethodName",
    "SampleCollectionEquipmentName",
    "ResultIdentifier",
    "CharacteristicName",
    "ResultSampleFractionText",
    "ResultMeasureValue",
    "ResultMeasure/MeasureUnitCode",
    "ResultStatusIdentifier",
    "ResultValueTypeName",
    "USGSPCode",
    "ProviderName",
    # Optional geo columns (if present in uploaded data)
    "LatitudeMeasure",
    "LongitudeMeasure",
    "ActivityStartDateTime",
]

# Group-by candidates for quick analysis
DEFAULT_GROUPS = ["CharacteristicName", "ResultSampleFractionText", "MonitoringLocationIdentifier"]

# ----------------------------
# Utilities
# ----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy with standardized column names to match EXPECTED_COLS (case-insensitive).
    If a close match exists with different case, we rename it to canonical.
    """
    df = df.copy()
    lower_to_canon = {c.lower(): c for c in EXPECTED_COLS}
    new_cols = {}
    for c in df.columns:
        canon = lower_to_canon.get(c.lower())
        new_cols[c] = canon if canon else c
    df = df.rename(columns=new_cols)
    return df


def move_geo_next_to_location(df: pd.DataFrame) -> pd.DataFrame:
    """
    If LatitudeMeasure/LongitudeMeasure exist, place them immediately after MonitoringLocationIdentifier.
    """
    cols = list(df.columns)
    if "MonitoringLocationIdentifier" not in cols:
        return df
    left = []
    right = []
    after_inserted = False
    for c in cols:
        if c == "MonitoringLocationIdentifier":
            left.append(c)
            if "LatitudeMeasure" in cols:
                left.append("LatitudeMeasure")
            if "LongitudeMeasure" in cols:
                left.append("LongitudeMeasure")
            after_inserted = True
        else:
            if after_inserted and c in ("LatitudeMeasure", "LongitudeMeasure"):
                continue
            right.append(c)
    return df[left + right]


def coerce_result_value_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a numeric mirror column NUMCOL from ResultMeasureValue.
    Treat strings like '00000000' as missing.
    Accepts scientific-notation strings like '6e-05'.
    """
    df = df.copy()
    if "ResultMeasureValue" not in df.columns:
        return df

    s = df["ResultMeasureValue"].astype(str).str.strip()

    # Mark known bad tokens as NaN
    bad_tokens = {"", "nan", "none", "null", "na", "00000000"}
    s = s.mask(s.str.lower().isin(bad_tokens))

    # Coerce to numeric
    df[NUMCOL] = pd.to_numeric(s, errors="coerce")
    return df


def build_datetime_helper(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a helper datetime column for time-series plots.
    Prefer ActivityStartDateTime; else parse ActivityStartDate.
    """
    df = df.copy()
    if "ActivityStartDateTime" in df.columns:
        df[DT_HELPER] = pd.to_datetime(df["ActivityStartDateTime"], errors="coerce", utc=False)
    elif "ActivityStartDate" in df.columns:
        df[DT_HELPER] = pd.to_datetime(df["ActivityStartDate"], errors="coerce", utc=False, infer_datetime_format=True)
    return df


def drop_non_usgs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep rows that clearly come from USGS. This looks at OrganizationIdentifier or ProviderName if available.
    """
    df = df.copy()
    cols = set(df.columns)
    mask_parts = []

    if "OrganizationIdentifier" in cols:
        mask_parts.append(df["OrganizationIdentifier"].astype(str).str.upper().str.startswith("USGS"))
    if "ProviderName" in cols:
        mask_parts.append(df["ProviderName"].astype(str).str.upper().eq("NWIS"))

    if mask_parts:
        mask_any = mask_parts[0]
        for m in mask_parts[1:]:
            mask_any = mask_any | m
        df = df[mask_any].copy()

    return df


def summarize_aggregations(df: pd.DataFrame, value_col: str, group_dims: list) -> pd.DataFrame:
    """
    Robust aggregation that only touches the numeric mirror column.
    - Drops NaN in the numeric column
    - Allows group keys to be missing in the DataFrame (silently skip)
    """
    dims = [c for c in group_dims if c in df.columns]
    if not dims or value_col not in df.columns:
        return pd.DataFrame()

    g = df[dims + [value_col]].dropna(subset=[value_col])
    if g.empty:
        return pd.DataFrame()

    out = (
        g.groupby(dims, dropna=False)[value_col]
        .agg(count="count", mean="mean", median="median", min="min", max="max")
        .reset_index()
        .sort_values("count", ascending=False)
    )
    return out


def to_csv_download(df: pd.DataFrame, filename: str) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def describe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light profile of key columns if they exist.
    """
    cols = [c for c in EXPECTED_COLS if c in df.columns]
    head = df[cols].head(50) if cols else df.head(50)
    return head


# ----------------------------
# Streamlit UI
# ----------------------------
def sidebar_about():
    with st.sidebar.expander("ℹ️ About", expanded=False):
        st.markdown(
            textwrap.dedent(
                f"""
                **{DISPLAY_NAME}**

                Upload a USGS/WQX-like CSV, clean/filter, explore quick stats, and download a cleaned/filtered CSV.
                - Automatically standardizes column names (case-insensitive)
                - Optional: drop non-USGS rows
                - Numeric-safe analysis using a mirror of **ResultMeasureValue**
                - Quick aggregations / histograms / boxplots / time series
                """
            )
        )


def sidebar_controls():
    st.sidebar.header("1) Upload CSV")
    file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"], key="upl_csv")
    st.sidebar.divider()

    st.sidebar.header("2) Cleaning Options")
    drop_usgs_only = st.sidebar.checkbox("Keep only USGS/NWIS rows", value=True, key="opt_usgs_only")
    reorder_geo = st.sidebar.checkbox(
        "Place Latitude/Longitude after MonitoringLocationIdentifier", value=True, key="opt_reorder_geo"
    )
    st.sidebar.divider()

    st.sidebar.header("3) Filters")
    return file, drop_usgs_only, reorder_geo


def render_data_tab(df: pd.DataFrame):
    st.subheader("Data Preview")
    st.dataframe(describe_dataframe(df), use_container_width=True)

    st.subheader("Column Summary")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows", f"{len(df):,}")
    with c2:
        st.metric("Columns", f"{df.shape[1]:,}")
    with c3:
        st.metric(
            "Non-null in ResultMeasureValue",
            f"{df['ResultMeasureValue'].notna().sum():,}" if "ResultMeasureValue" in df.columns else "n/a",
        )

    with st.expander("Show all columns"):
        st.write(list(df.columns))


def render_downloads(df_clean: pd.DataFrame, df_filtered: pd.DataFrame):
    st.subheader("Download")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Download CLEANED CSV",
            data=to_csv_download(df_clean, "cleaned.csv"),
            file_name="cleaned_usgs_water_quality.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_clean",
        )
    with c2:
        st.download_button(
            "⬇️ Download FILTERED CSV",
            data=to_csv_download(df_filtered, "filtered.csv"),
            file_name="filtered_usgs_water_quality.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_filtered",
        )


def render_filters(df: pd.DataFrame, key_prefix: str = "filters") -> pd.DataFrame:
    """
    Simple filter UI for common columns. Returns the filtered DataFrame.
    Each widget has a unique key using key_prefix to avoid StreamlitDuplicateElementId.
    """
    dfF = df.copy()

    with st.expander("🔎 Filters", expanded=False):
        cols = dfF.columns

        # CharacteristicName
        if "CharacteristicName" in cols:
            vals = sorted(dfF["CharacteristicName"].dropna().astype(str).unique().tolist())
            chosen = st.multiselect(
                "CharacteristicName", vals, default=[], key=f"{key_prefix}_charname"
            )
            if chosen:
                dfF = dfF[dfF["CharacteristicName"].astype(str).isin(chosen)]

        # ResultSampleFractionText
        if "ResultSampleFractionText" in cols:
            vals = sorted(dfF["ResultSampleFractionText"].dropna().astype(str).unique().tolist())
            chosen = st.multiselect(
                "ResultSampleFractionText", vals, default=[], key=f"{key_prefix}_fraction"
            )
            if chosen:
                dfF = dfF[dfF["ResultSampleFractionText"].astype(str).isin(chosen)]

        # Unit filter
        if "ResultMeasure/MeasureUnitCode" in cols:
            vals = sorted(dfF["ResultMeasure/MeasureUnitCode"].dropna().astype(str).unique().tolist())
            chosen = st.multiselect(
                "ResultMeasure/MeasureUnitCode", vals, default=[], key=f"{key_prefix}_unit"
            )
            if chosen:
                dfF = dfF[dfF["ResultMeasure/MeasureUnitCode"].astype(str).isin(chosen)]

        # Site filter
        if "MonitoringLocationIdentifier" in cols:
            top_sites = dfF["MonitoringLocationIdentifier"].value_counts().head(200).index.tolist()
            chosen = st.multiselect(
                "MonitoringLocationIdentifier (Top 200)", top_sites, default=[], key=f"{key_prefix}_site"
            )
            if chosen:
                dfF = dfF[dfF["MonitoringLocationIdentifier"].isin(chosen)]

        # Date range
        if DT_HELPER in cols and dfF[DT_HELPER].notna().any():
            min_dt = pd.to_datetime(dfF[DT_HELPER]).min()
            max_dt = pd.to_datetime(dfF[DT_HELPER]).max()
            st.write("Date range filter (based on ActivityStartDate/ActivityStartDateTime):")
            d1, d2 = st.columns(2)
            with d1:
                start_date = st.date_input(
                    "Start date", value=min_dt.date() if pd.notna(min_dt) else None, key=f"{key_prefix}_start"
                )
            with d2:
                end_date = st.date_input(
                    "End date", value=max_dt.date() if pd.notna(max_dt) else None, key=f"{key_prefix}_end"
                )

            if start_date and end_date:
                m1 = pd.to_datetime(start_date)
                m2 = pd.to_datetime(end_date) + pd.Timedelta(days=1)  # inclusive end
                mask = (dfF[DT_HELPER] >= m1) & (dfF[DT_HELPER] < m2)
                dfF = dfF[mask]

    st.success(f"Filtered rows: {len(dfF):,}", icon="✅")
    return dfF


def render_analysis_tab(dfA: pd.DataFrame):
    st.subheader("Quick Aggregations")

    dims = st.multiselect(
        "Group by (choose 1–3):",
        options=[c for c in DEFAULT_GROUPS if c in dfA.columns],
        default=[c for c in DEFAULT_GROUPS if c in dfA.columns][:2],
        max_selections=3,
        key="analysis_dims",
    )

    agg_tbl = summarize_aggregations(dfA, NUMCOL, dims)
    if agg_tbl.empty:
        st.info("No aggregation available with current data/filters.")
    else:
        st.dataframe(agg_tbl, use_container_width=True)

    st.subheader("Charts")

    # Histogram of ResultMeasureValue (numeric mirror)
    if NUMCOL in dfA.columns and dfA[NUMCOL].notna().any():
        v = dfA[NUMCOL].dropna()
        if st.checkbox("Histogram of ResultMeasureValue", value=True, key="hist_checkbox"):
            fig = px.histogram(
                v.to_frame(name="ResultMeasureValue"),
                x="ResultMeasureValue",
                nbins=50,
                title="Histogram of ResultMeasureValue",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Boxplot by CharacteristicName (top N)
    if {"CharacteristicName", NUMCOL}.issubset(dfA.columns):
        if st.checkbox("Boxplot by CharacteristicName (Top 10)", value=False, key="box_checkbox"):
            sub = dfA.dropna(subset=[NUMCOL, "CharacteristicName"]).copy()
            if not sub.empty:
                top = sub["CharacteristicName"].value_counts().head(10).index
                sub = sub[sub["CharacteristicName"].isin(top)]
                if not sub.empty:
                    fig = px.box(
                        sub,
                        x="CharacteristicName",
                        y=NUMCOL,
                        points="outliers",
                        title="ResultMeasureValue by CharacteristicName (Top 10)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # Monthly median time-series
    if DT_HELPER in dfA.columns and NUMCOL in dfA.columns:
        if st.checkbox("Time series (monthly median of ResultMeasureValue)", value=False, key="ts_checkbox"):
            ts = dfA[[DT_HELPER, NUMCOL]].dropna().copy()
            if not ts.empty:
                ts["_period"] = pd.to_datetime(ts[DT_HELPER], errors="coerce").dt.to_period("M").dt.to_timestamp()
                grp = ts.groupby("_period")[NUMCOL].median().reset_index()
                if not grp.empty:
                    fig = px.line(
                        grp, x="_period", y=NUMCOL, markers=True, title="Monthly Median of ResultMeasureValue"
                    )
                    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title(DISPLAY_NAME)
    sidebar_about()

    file, drop_usgs_only, reorder_geo = sidebar_controls()

    st.markdown("### Workflow\n1) Upload → 2) Clean → 3) Filter → 4) Analyze → 5) Download")

    if not file:
        st.info("Upload a CSV to begin.")
        return

    # Load CSV
    try:
        df_raw = pd.read_csv(file, dtype=str, low_memory=False)
    except Exception:
        file.seek(0)
        df_raw = pd.read_csv(file, low_memory=False)  # try default dtypes
    df_raw = normalize_columns(df_raw)

    # CLEAN
    df_clean = df_raw.copy()

    # Optional: keep only USGS/NWIS rows
    if drop_usgs_only:
        df_clean = drop_non_usgs(df_clean)

    # Numeric mirror for ResultMeasureValue
    df_clean = coerce_result_value_numeric(df_clean)

    # Date helper
    df_clean = build_datetime_helper(df_clean)

    # Optional: reorder geo columns next to MonitoringLocationIdentifier
    if reorder_geo:
        df_clean = move_geo_next_to_location(df_clean)

    # TABS
    tab1, tab2, tab3 = st.tabs(["Data", "Analysis", "Export"])

    with tab1:
        render_data_tab(df_clean)
        st.caption("Note: Internal numeric mirror column used for analysis: `__RMV` (not exported).")

    with tab2:
        # Filters for the Analysis tab (unique widget keys)
        df_filtered_analysis = render_filters(df_clean, key_prefix="analysis_filters")
        render_analysis_tab(df_filtered_analysis)

    with tab3:
        # Separate filter set for the Export tab (unique widget keys)
        df_filtered_export = render_filters(df_clean, key_prefix="export_filters")

        # Do not export internal helper columns
        drop_cols_on_export = [NUMCOL, DT_HELPER]
        export_clean = df_clean.drop(columns=[c for c in drop_cols_on_export if c in df_clean.columns], errors="ignore")
        export_filtered = df_filtered_export.drop(
            columns=[c for c in drop_cols_on_export if c in df_filtered_export.columns], errors="ignore"
        )

        render_downloads(export_clean, export_filtered)

        st.markdown("#### Notes")
        st.markdown("- Cleaned CSV preserves your original columns, except we may reorder Latitude/Longitude.")
        st.markdown(
            "- Analysis uses a safe numeric mirror of `ResultMeasureValue` (`__RMV`) to avoid dtype errors (e.g., `'00000000'` or scientific-notation strings)."
        )


if __name__ == "__main__":
    main()
