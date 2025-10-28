import io
import hashlib
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------
# Constants & Config
# ----------------------------

st.set_page_config(page_title="USGS Water Quality Toolkit", layout="wide")

EXPECTED_COLUMNS = [
    "OrganizationIdentifier", "OrganizationFormalName", "ActivityIdentifier",
    "ActivityTypeCode", "ActivityMediaName", "ActivityMediaSubdivisionName",
    "ActivityStartDate", "ActivityStartTime/Time", "ActivityStartTime/TimeZoneCode",
    "ActivityConductingOrganizationText", "MonitoringLocationIdentifier",
    "HydrologicCondition", "HydrologicEvent",
    "SampleCollectionMethod/MethodIdentifier", "SampleCollectionMethod/MethodIdentifierContext",
    "SampleCollectionMethod/MethodName", "SampleCollectionEquipmentName",
    "ResultIdentifier", "CharacteristicName", "ResultSampleFractionText",
    "ResultMeasureValue", "ResultMeasure/MeasureUnitCode", "ResultStatusIdentifier",
    "ResultValueTypeName", "USGSPCode", "ProviderName"
]

# Common US time zone abbreviations -> IANA names (extend as needed)
TZ_ABBREV_TO_IANA = {
    "UTC": "UTC",
    "GMT": "UTC",
    "EST": "America/New_York",
    "EDT": "America/New_York",
    "CST": "America/Chicago",
    "CDT": "America/Chicago",
    "MST": "America/Denver",
    "MDT": "America/Denver",
    "PST": "America/Los_Angeles",
    "PDT": "America/Los_Angeles",
    # Some WQP exports use numeric UTC offsets like "-05:00"; pandas can localize with these too.
}

# Defaults for "invalid" ResultStatusIdentifier
DEFAULT_INVALID_STATUSES = ["Rejected", "Invalid"]

# SessionState keys
SS_KEYS = {
    "invalid_statuses": "invalid_statuses",
    "filters": "filters",
    "text_search_cols": "text_search_cols",
}

# ----------------------------
# Small Utilities
# ----------------------------

def safe_colname(col: str) -> str:
    """Create an internal-safe column name (no spaces/slashes; lowercase)."""
    return (
        col.strip()
           .replace("/", "_")
           .replace(" ", "_")
           .replace("-", "_")
           .replace("\n", "_")
           .lower()
    )

def build_colmap(df: pd.DataFrame) -> Dict[str, str]:
    """Original -> safe column mapping."""
    return {c: safe_colname(c) for c in df.columns}

def reverse_map(d: Dict[str, str]) -> Dict[str, str]:
    """Safe -> original reverse mapping."""
    return {v: k for k, v in d.items()}

def to_bytes_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def stable_hash_row(row: pd.Series) -> str:
    m = hashlib.md5()
    m.update("|".join([str(x) if pd.notna(x) else "" for x in row]).encode("utf-8"))
    return m.hexdigest()

def has_cols(df: pd.DataFrame, cols: List[str]) -> bool:
    return all(c in df.columns for c in cols)

def show_df(df: pd.DataFrame, height: int = 320):
    """Unified dataframe display that uses the new Streamlit width API."""
    st.dataframe(df, width="stretch", height=height)

def warn_missing(*cols: str):
    missing = [c for c in cols if c not in st.session_state["df_original"].columns]
    if missing:
        st.info(f"Note: missing column(s): {', '.join(missing)}")

# ----------------------------
# Cached Core Steps
# ----------------------------

@st.cache_data(show_spinner=False)
def read_csv(file: io.BytesIO) -> pd.DataFrame:
    """Robust CSV reader for up to ~200MB; compatible across pandas versions."""
    base_kwargs = dict(
        encoding="utf-8",
        na_values=["", "NA", "NaN"],
        on_bad_lines="skip"
    )

    # Try fast C engine first (works with low_memory)
    try:
        return pd.read_csv(file, low_memory=False, **base_kwargs)
    except Exception:
        # Fallback to Python engine (no low_memory there)
        try:
            return pd.read_csv(file, engine="python", **base_kwargs)
        except Exception:
            # Last resort: be very permissive about encodings
            try:
                return pd.read_csv(file, engine="python", encoding_errors="ignore")
            except TypeError:
                # Older pandas without encoding_errors
                return pd.read_csv(file, engine="python")

@st.cache_data(show_spinner=False)
def schema_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Schema summary: column, present?, dtype, non-null count, % missing."""
    info = []
    present_set = set(df.columns)
    for c in EXPECTED_COLUMNS:
        present = c in present_set
        dtype = str(df[c].dtype) if present else "-"
        nonnull = int(df[c].notna().sum()) if present else 0
        total = len(df) if present else 0
        pct_missing = float(100 * (1 - nonnull / total)) if present and total > 0 else 100.0
        info.append({
            "Column": c,
            "Present": present,
            "Dtype": dtype,
            "Non-Null Count": nonnull,
            "% Missing": round(pct_missing, 2),
        })
    # Also add any extra columns that were not listed but are present
    for c in df.columns:
        if c not in EXPECTED_COLUMNS:
            nonnull = int(df[c].notna().sum())
            total = len(df)
            pct_missing = float(100 * (1 - nonnull / total)) if total > 0 else 100.0
            info.append({
                "Column": c,
                "Present": True,
                "Dtype": str(df[c].dtype),
                "Non-Null Count": nonnull,
                "% Missing": round(pct_missing, 2),
            })
    return pd.DataFrame(info)

@st.cache_data(show_spinner=False)
def infer_types(df: pd.DataFrame) -> pd.DataFrame:
    """Infer and cast types; create ActivityStartDateTime robustly without pandas internals."""
    df = df.copy()

    # Trim whitespace in strings
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].map(lambda x: x.strip() if isinstance(x, str) else x)

    # Parse ActivityStartDate to date (no tz)
    if "ActivityStartDate" in df.columns:
        ds = pd.to_datetime(df["ActivityStartDate"], errors="coerce")
        df["ActivityStartDate"] = ds.dt.date

    # Cast numeric
    if "ResultMeasureValue" in df.columns:
        df["ResultMeasureValue"] = pd.to_numeric(df["ResultMeasureValue"], errors="coerce")

    # Build ActivityStartDateTime if date AND time exist
    if "ActivityStartDate" in df.columns and "ActivityStartTime/Time" in df.columns:
        # Parse time strings to "%H:%M:%S"; missing -> "00:00:00"
        ts = pd.to_datetime(df["ActivityStartTime/Time"], errors="coerce")
        time_str = ts.dt.strftime("%H:%M:%S").fillna("00:00:00")

        # Construct a full datetime from date + time
        date_str = pd.to_datetime(df["ActivityStartDate"], errors="coerce").dt.strftime("%Y-%m-%d")
        dt_naive = pd.to_datetime(date_str + " " + time_str, errors="coerce")

        # Timezone resolution: single uniform code -> use mapping; else UTC
        tz = "UTC"
        if "ActivityStartTime/TimeZoneCode" in df.columns:
            tz_codes = df["ActivityStartTime/TimeZoneCode"].astype(str).str.strip().str.upper()
            uniq = tz_codes.dropna().unique()
            if len(uniq) == 1:
                code = uniq[0]
                if code in TZ_ABBREV_TO_IANA:
                    tz = TZ_ABBREV_TO_IANA[code]
                elif code.startswith(("+", "-")):
                    tz = code  # pandas can localize fixed offsets like "+03:00"
                else:
                    tz = "UTC"

        try:
            df["ActivityStartDateTime"] = dt_naive.dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
        except Exception:
            df["ActivityStartDateTime"] = dt_naive.dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")

    # Drop all-empty columns
    empty_cols = [c for c in df.columns if df[c].notna().sum() == 0]
    if empty_cols:
        df = df.drop(columns=empty_cols)

    return df

@st.cache_data(show_spinner=False)
def clean_df(
    df: pd.DataFrame,
    invalid_statuses: List[str]
) -> pd.DataFrame:
    """Full cleaning, including optional invalid status filtering and de-dup."""
    df = infer_types(df)

    # Drop rows with "invalid" statuses (if column exists)
    if "ResultStatusIdentifier" in df.columns and invalid_statuses:
        mask = df["ResultStatusIdentifier"].isin(invalid_statuses)
        df = df.loc[~mask].copy()

    # Exact duplicate rows removal
    df = df.drop_duplicates().reset_index(drop=True)

    # Internal column standardization (safe names), but keep mapping for export
    colmap = build_colmap(df)
    df = df.rename(columns=colmap)
    df.attrs["colmap"] = colmap  # store mapping for later
    return df

@st.cache_data(show_spinner=False)
def build_kpis(df: pd.DataFrame) -> Dict[str, Optional[int]]:
    kpis = {
        "Rows": len(df),
        "Columns": len(df.columns),
        "Unique Monitoring Locations": None,
        "Unique Characteristics": None,
        "Date Range Min": None,
        "Date Range Max": None,
    }
    if "MonitoringLocationIdentifier" in df.columns:
        kpis["Unique Monitoring Locations"] = int(df["MonitoringLocationIdentifier"].nunique())
    if "CharacteristicName" in df.columns:
        kpis["Unique Characteristics"] = int(df["CharacteristicName"].nunique())
    if "ActivityStartDate" in df.columns and df["ActivityStartDate"].notna().any():
        dates = pd.to_datetime(df["ActivityStartDate"], errors="coerce")
        if dates.notna().any():
            kpis["Date Range Min"] = str(dates.min().date())
            kpis["Date Range Max"] = str(dates.max().date())
    return kpis

# ----------------------------
# Filtering & Analysis
# ----------------------------

def apply_filters(
    df_original_cols: pd.DataFrame,
    filters: Dict
) -> pd.DataFrame:
    """Apply multi-selects, date range, and text search over ORIGINAL column names."""
    df = df_original_cols.copy()

    # Date range
    if "ActivityStartDate" in df.columns and filters.get("date_range"):
        start_date, end_date = filters["date_range"]
        if start_date and end_date:
            d = pd.to_datetime(df["ActivityStartDate"], errors="coerce")
            mask = (d >= pd.to_datetime(start_date)) & (d <= pd.to_datetime(end_date))
            df = df.loc[mask]

    # Multi-select filters
    for key in [
        "CharacteristicName", "ResultSampleFractionText",
        "ResultMeasure/MeasureUnitCode", "MonitoringLocationIdentifier", "ProviderName"
    ]:
        vals = filters.get(key)
        if key in df.columns and vals:
            df = df[df[key].isin(vals)]

    # Text search
    query = filters.get("text_query")
    search_cols = filters.get("text_cols", [])
    if query and search_cols:
        regex = filters.get("text_is_regex", False)
        q = str(query)
        mask = pd.Series(False, index=df.index)
        for c in search_cols:
            if c in df.columns:
                s = df[c].astype(str)
                try:
                    if regex:
                        m = s.str.contains(q, regex=True, case=False, na=False)
                    else:
                        m = s.str.contains(q, regex=False, case=False, na=False)
                    mask = mask | m
                except Exception:
                    m = s.str.contains(q, regex=False, case=False, na=False)
                    mask = mask | m
        df = df[mask]

    return df.copy()

def summarize_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame()
    return df.groupby(col).size().reset_index(name="count").sort_values("count", ascending=False)

def summarize_aggregations(
    df: pd.DataFrame,
    value_col: str,
    group_dims: List[str]
) -> pd.DataFrame:
    if value_col not in df.columns:
        return pd.DataFrame()
    dims = [c for c in group_dims if c in df.columns]
    if not dims:
        return pd.DataFrame()
    g = df.groupby(dims)[value_col].agg(["count", "mean", "median", "min", "max"]).reset_index()
    return g.sort_values("count", ascending=False)

# ----------------------------
# Workup / Feature Engineering
# ----------------------------

def build_workup(
    df_clean_safe: pd.DataFrame
) -> pd.DataFrame:
    """Create engineered columns and order them after original metadata."""
    df = df_clean_safe.copy()
    colmap = df.attrs.get("colmap", {})

    # Uppercase unit code if present (safe name)
    unit_safe = safe_colname("ResultMeasure/MeasureUnitCode")
    if unit_safe in df.columns:
        df[unit_safe] = df[unit_safe].astype(str).str.upper()

    # Build record_id from available keys (SAFE names)
    key_candidates = [
        "MonitoringLocationIdentifier",
        "ActivityIdentifier",
        "CharacteristicName",
        "ResultIdentifier",
    ]
    keys_present = [safe_colname(k) for k in key_candidates if safe_colname(k) in df.columns]
    if keys_present:
        df["record_id"] = df[keys_present].apply(stable_hash_row, axis=1)
    else:
        df["record_id"] = df.apply(stable_hash_row, axis=1)

    # Column ordering: original ID/metadata -> result columns -> engineered
    id_meta = [
        "OrganizationIdentifier", "OrganizationFormalName", "ActivityIdentifier",
        "ActivityTypeCode", "ActivityMediaName", "ActivityMediaSubdivisionName",
        "ActivityStartDate", "ActivityStartTime/Time", "ActivityStartTime/TimeZoneCode",
        "ActivityConductingOrganizationText", "MonitoringLocationIdentifier",
        "HydrologicCondition", "HydrologicEvent",
        "SampleCollectionMethod/MethodIdentifier", "SampleCollectionMethod/MethodIdentifierContext",
        "SampleCollectionMethod/MethodName", "SampleCollectionEquipmentName",
        "ResultIdentifier", "CharacteristicName", "ResultSampleFractionText",
        "ResultStatusIdentifier", "ResultValueTypeName", "USGSPCode", "ProviderName",
        "ActivityStartDateTime",
    ]
    result_cols = [
        "ResultMeasureValue", "ResultMeasure/MeasureUnitCode",
    ]
    engineered = ["record_id"]

    id_meta_safe = [safe_colname(c) for c in id_meta if safe_colname(c) in df.columns]
    result_safe = [safe_colname(c) for c in result_cols if safe_colname(c) in df.columns]
    engineered_safe = [c for c in engineered if c in df.columns]
    remaining = [c for c in df.columns if c not in set(id_meta_safe + result_safe + engineered_safe)]

    ordered = id_meta_safe + result_safe + remaining + engineered_safe
    df = df.loc[:, ordered]

    df.attrs["colmap"] = colmap
    return df

def revert_to_original_cols(df_safe: pd.DataFrame) -> pd.DataFrame:
    """Map safe internal names back to original names for export."""
    colmap = df_safe.attrs.get("colmap", {})
    rmap = reverse_map(colmap) if colmap else {}
    new_cols = [rmap.get(c, c) for c in df_safe.columns]  # keep engineered (e.g., record_id) as-is
    out = df_safe.copy()
    out.columns = new_cols
    return out

# ----------------------------
# App
# ----------------------------

def kpi_card(label: str, value):
    st.metric(label, value if value is not None else "—")

def main():
    st.title("USGS Water Quality Data — Filter • Clean • Analyze • Download")

    st.sidebar.header("Upload CSV")
    file = st.sidebar.file_uploader(
        "Upload a .csv file (≤ ~200MB)",
        type=["csv"],
        accept_multiple_files=False,
        help="Reads with UTF-8; treats blank/NA/NaN as missing; skips bad lines."
    )

    # Initialize session state
    if SS_KEYS["invalid_statuses"] not in st.session_state:
        st.session_state[SS_KEYS["invalid_statuses"]] = DEFAULT_INVALID_STATUSES.copy()
    if SS_KEYS["filters"] not in st.session_state:
        st.session_state[SS_KEYS["filters"]] = {}
    if SS_KEYS["text_search_cols"] not in st.session_state:
        st.session_state[SS_KEYS["text_search_cols"]] = []

    if not file:
        st.markdown("↖️ Upload a CSV to get started.")
        st.caption("The app will infer types, tolerate missing columns, and summarize schema.")
        return

    # Read file
    try:
        df_raw = read_csv(file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    # Keep an untouched copy with original columns for filtering & analysis
    df_original = df_raw.copy()
    st.session_state["df_original"] = df_original

    # Tabs
    tab_overview, tab_clean, tab_explore, tab_analysis, tab_downloads = st.tabs(
        ["Overview", "Cleaning", "Filter & Explore", "Analysis", "Downloads"]
    )

    # ---------------- Overview ----------------
    with tab_overview:
        st.subheader("Quick Overview")

        kpi_cols = st.columns(5)
        kpis = build_kpis(df_original)
        with kpi_cols[0]: kpi_card("Rows", kpis["Rows"])
        with kpi_cols[1]: kpi_card("Columns", kpis["Columns"])
        with kpi_cols[2]: kpi_card("Unique Locations", kpis["Unique Monitoring Locations"])
        with kpi_cols[3]: kpi_card("Unique Characteristics", kpis["Unique Characteristics"])
        with kpi_cols[4]:
            if kpis["Date Range Min"] and kpis["Date Range Max"]:
                kpi_card("Date Range", f"{kpis['Date Range Min']} → {kpis['Date Range Max']}")
            else:
                kpi_card("Date Range", "—")

        st.markdown("### Data Dictionary (present/missing, dtypes, % missing)")
        show_df(schema_summary(df_original), 320)

        with st.expander("Peek at data (first 200 rows)", expanded=False):
            show_df(df_original.head(200), 320)

    # ---------------- Cleaning ----------------
    with tab_clean:
        st.subheader("Cleaning Options")

        st.write("Toggle cleaning options; a preview will appear below. Cleaning includes:")
        st.markdown(
            "- Type casting for dates/times → `ActivityStartDateTime` (timezone-aware best-effort)\n"
            "- Numeric casting for `ResultMeasureValue`\n"
            "- Column name standardization internally (original names restored on export)\n"
            "- Dropping all-empty columns; trimming whitespace; de-duplicating\n"
            "- Optional row-drop by `ResultStatusIdentifier`"
        )

        # Invalid statuses editor
        if "ResultStatusIdentifier" in df_original.columns:
            unique_statuses = sorted([s for s in df_original["ResultStatusIdentifier"].dropna().astype(str).unique()])
            st.write("**Drop rows with these ResultStatusIdentifier values**")
            invalid_selected = st.multiselect(
                "Select status values to drop",
                options=unique_statuses,
                default=[s for s in st.session_state[SS_KEYS["invalid_statuses"]] if s in unique_statuses],
                help="You can also type custom values below, one per line."
            )
            custom_invalid = st.text_area("Custom status values (one per line)", "", height=80)
            custom_list = [x.strip() for x in custom_invalid.splitlines() if x.strip()]
            invalid_statuses = sorted(set(invalid_selected + custom_list))
        else:
            st.info("`ResultStatusIdentifier` column not found; skip invalid-status filtering.")
            invalid_statuses = []

        st.session_state[SS_KEYS["invalid_statuses"]] = invalid_statuses

        # Run cleaning
        df_clean_safe = clean_df(df_original, invalid_statuses)  # safe colnames inside
        # For preview in human-readable form, revert to original names if possible
        df_clean_preview = revert_to_original_cols(df_clean_safe)

        st.markdown("### Cleaned Preview (first 200 rows)")
        show_df(df_clean_preview.head(200), 320)

        with st.expander("Notes on internal columns"):
            st.caption(
                "Internally, columns are lower-cased with spaces/slashes replaced. "
                "For downloads, original column names are restored. "
                "Engineered columns like `record_id` remain in lowercase."
            )

    # ---------------- Filter & Explore ----------------
    with tab_explore:
        st.subheader("Filter & Explore")
        df = df_original  # all filters are applied to ORIGINAL column names

        # Build filter widgets
        fc1, fc2, fc3, fc4, fc5 = st.columns(5)
        def opts(col): return sorted(df[col].dropna().astype(str).unique().tolist()) if col in df.columns else []

        with fc1:
            sel_char = st.multiselect("CharacteristicName", options=opts("CharacteristicName"))
        with fc2:
            sel_frac = st.multiselect("ResultSampleFractionText", options=opts("ResultSampleFractionText"))
        with fc3:
            sel_unit = st.multiselect("ResultMeasure/MeasureUnitCode", options=opts("ResultMeasure/MeasureUnitCode"))
        with fc4:
            sel_loc = st.multiselect("MonitoringLocationIdentifier", options=opts("MonitoringLocationIdentifier"))
        with fc5:
            sel_provider = st.multiselect("ProviderName", options=opts("ProviderName"))

        # Date range
        if "ActivityStartDate" in df.columns and df["ActivityStartDate"].notna().any():
            dmin = pd.to_datetime(df["ActivityStartDate"], errors="coerce").min()
            dmax = pd.to_datetime(df["ActivityStartDate"], errors="coerce").max()
            # Guard for empty/NaT
            if pd.isna(dmin) or pd.isna(dmax):
                date_range = None
            else:
                date_range = st.date_input("ActivityStartDate range", value=(dmin.date(), dmax.date()))
        else:
            date_range = None
            warn_missing("ActivityStartDate")

        # Text search
        st.markdown("**Text search**")
        text_cols = st.multiselect(
            "Apply search to columns",
            options=[c for c in df.columns],
            default=st.session_state[SS_KEYS["text_search_cols"]],
        )
        st.session_state[SS_KEYS["text_search_cols"]] = text_cols
        tcol1, tcol2 = st.columns([3,1])
        with tcol1:
            q = st.text_input("Search query (plain or regex)", "")
        with tcol2:
            is_regex = st.checkbox("Regex", value=False)

        # Assemble filters & apply
        filt = {
            "CharacteristicName": sel_char,
            "ResultSampleFractionText": sel_frac,
            "ResultMeasure/MeasureUnitCode": sel_unit,
            "MonitoringLocationIdentifier": sel_loc,
            "ProviderName": sel_provider,
            "date_range": date_range,
            "text_query": q,
            "text_cols": text_cols,
            "text_is_regex": is_regex,
        }
        st.session_state[SS_KEYS["filters"]] = filt

        df_filtered = apply_filters(df_original, filt)
        st.success(f"Filtered rows: {len(df_filtered)} / {len(df_original)}")

        with st.expander("Preview filtered data (first 200 rows)", expanded=True):
            show_df(df_filtered.head(200), 320)

    # ---------------- Analysis ----------------
    with tab_analysis:
        st.subheader("Analysis")

        # Use the filtered subset for analysis
        dfA = apply_filters(df_original, st.session_state[SS_KEYS["filters"]])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Counts by CharacteristicName**")
            show_df(summarize_counts(dfA, "CharacteristicName"), 280)
        with c2:
            st.markdown("**Counts by MonitoringLocationIdentifier**")
            show_df(summarize_counts(dfA, "MonitoringLocationIdentifier"), 280)

        st.markdown("---")
        st.markdown("### Aggregations for ResultMeasureValue")
        dims = st.multiselect(
            "Group by (choose 1–3)",
            options=[c for c in ["CharacteristicName", "ResultSampleFractionText", "MonitoringLocationIdentifier", "ProviderName"] if c in dfA.columns],
            default=[c for c in ["CharacteristicName", "ResultSampleFractionText"] if c in dfA.columns]
        )
        agg_tbl = summarize_aggregations(dfA, "ResultMeasureValue", dims)
        if agg_tbl.empty:
            warn_missing("ResultMeasureValue")
            st.info("No aggregation available with current data/filters.")
        else:
            show_df(agg_tbl, 340)

        st.markdown("---")
        st.markdown("### Visuals (optional)")
        v1, v2, v3 = st.columns(3)

        with v1:
            if "ResultMeasureValue" in dfA.columns and dfA["ResultMeasureValue"].notna().any():
                if st.checkbox("Histogram: ResultMeasureValue", value=True):
                    fig = px.histogram(dfA, x="ResultMeasureValue", nbins=50, title="Histogram of ResultMeasureValue")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Histogram unavailable (missing/empty ResultMeasureValue).")

        with v2:
            # Boxplot by CharacteristicName (top N categories)
            if has_cols(dfA, ["ResultMeasureValue", "CharacteristicName"]):
                if st.checkbox("Boxplot by CharacteristicName (top N)", value=False):
                    counts = dfA["CharacteristicName"].value_counts().head(10).index.tolist()
                    sub = dfA[dfA["CharacteristicName"].isin(counts)]
                    if not sub.empty:
                        fig = px.box(sub, x="CharacteristicName", y="ResultMeasureValue", points="outliers",
                                     title="ResultMeasureValue by CharacteristicName (Top N)")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No categories available for boxplot after filters.")
            else:
                st.info("Boxplot unavailable (needs CharacteristicName & ResultMeasureValue).")

        with v3:
            # Time series median by month if datetime exists
            dt_col = None
            for c in ["ActivityStartDateTime"]:
                if c in dfA.columns:
                    dt_col = c
                    break
            if dt_col is None and "ActivityStartDate" in dfA.columns:
                dfA = dfA.copy()
                dfA["__dt"] = pd.to_datetime(dfA["ActivityStartDate"], errors="coerce")
                dt_col = "__dt"

            if dt_col and "ResultMeasureValue" in dfA.columns:
                if st.checkbox("Time series (median by month)", value=False):
                    ts = dfA[[dt_col, "ResultMeasureValue"]].dropna()
                    if not ts.empty:
                        ts["_period"] = pd.to_datetime(ts[dt_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
                        grp = ts.groupby("_period")["ResultMeasureValue"].median().reset_index()
                        fig = px.line(grp, x="_period", y="ResultMeasureValue", markers=True,
                                      title="Monthly Median of ResultMeasureValue")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Time series empty after filters.")
            else:
                st.info("Time series unavailable (needs date/time and ResultMeasureValue).")

    # ---------------- Downloads ----------------
    with tab_downloads:
        st.subheader("Downloads (CSV, UTF-8, no index)")

        # Build filtered (original columns), cleaned (safe then reverted), workup (safe then reverted)
        df_filtered = apply_filters(df_original, st.session_state[SS_KEYS["filters"]])

        # Cleaned (safe internal), then revert to original names for export
        df_clean_safe = clean_df(df_original, st.session_state[SS_KEYS["invalid_statuses"]])
        df_clean_export = revert_to_original_cols(df_clean_safe)

        # Workup from cleaned safe
        df_workup_safe = build_workup(df_clean_safe)
        df_workup_export = revert_to_original_cols(df_workup_safe)

        # Ensure non-empty checks for each
        cA, cB, cC = st.columns(3)
        with cA:
            st.write("**Filtered subset**")
            if len(df_filtered) == 0:
                st.warning("No rows after current filters.")
            st.download_button(
                "Download filtered CSV",
                data=to_bytes_csv(df_filtered),
                file_name="filtered.csv",
                mime="text/csv",
                disabled=(len(df_filtered) == 0)
            )
        with cB:
            st.write("**Cleaned dataset**")
            if len(df_clean_export) == 0:
                st.warning("Cleaned dataset is empty.")
            st.download_button(
                "Download cleaned CSV",
                data=to_bytes_csv(df_clean_export),
                file_name="cleaned.csv",
                mime="text/csv",
                disabled=(len(df_clean_export) == 0)
            )
        with cC:
            st.write("**Workup (engineered)**")
            if len(df_workup_export) == 0:
                st.warning("Workup dataset is empty.")
            st.download_button(
                "Download workup CSV",
                data=to_bytes_csv(df_workup_export),
                file_name="workup.csv",
                mime="text/csv",
                disabled=(len(df_workup_export) == 0)
            )

        st.markdown("---")
        st.caption(
            "Downloads restore original column names where available; engineered columns like `record_id` remain. "
            "If some requested operation requires a missing column, the app adds a gentle warning and continues."
        )

    # Helpful notes
    with st.sidebar.expander("Tips"):
        st.caption(
            "- Use **Filter & Explore** to narrow the dataset before charts.\n"
            "- In **Cleaning**, edit the invalid statuses list to drop sketchy rows.\n"
            "- **Analysis** charts automatically handle missing columns or empty subsets."
        )

if __name__ == "__main__":
    main()
