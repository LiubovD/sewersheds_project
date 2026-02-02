"""
Tabular-only pipeline (no polygons):
- 2020 Decennial PL 94-171 at BLOCK level: group(P2) + group(P12)
- ACS 5-year at BLOCK GROUP level: group(B27010)

Then:
- Create race/ethnicity categories from P2 via variable LABEL matching
- Create age bins from P12 (sex-by-age) via LABEL matching
- Compute statewide + county totals (PL + ACS insurance)
- Save enriched datasets + summaries

Requires: pandas, requests
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

# -----------------------------
# Config
# -----------------------------
STATE_FIPS = "44"   # Rhode Island
PL_YEAR = 2020
ACS_YEAR = 2023     # change to 2022 if you want 2018–2022 ACS 5-year
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")

OUTDIR = Path("data_out")
OUTDIR.mkdir(exist_ok=True)

PL_ENDPOINT  = f"https://api.census.gov/data/{PL_YEAR}/dec/pl"
ACS_ENDPOINT = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"


# -----------------------------
# HTTP helpers
# -----------------------------
def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=10,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": "RIDOH-WWTF-CensusPull/1.1"})
    return s

SESSION = make_session()

def census_get(endpoint: str, params: dict, timeout: int = 180) -> pd.DataFrame:
    if CENSUS_API_KEY:
        params = dict(params)
        params["key"] = CENSUS_API_KEY
    r = SESSION.get(endpoint, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return pd.DataFrame(data[1:], columns=data[0])

def get_variables(endpoint: str) -> pd.DataFrame:
    """Download endpoint variables metadata and return as DataFrame(var, label, concept, predicateType)."""
    url = endpoint.rstrip("/") + "/variables.json"
    r = SESSION.get(url, timeout=180)
    r.raise_for_status()
    vars_json = r.json().get("variables", {})
    rows = []
    for var, meta in vars_json.items():
        rows.append({
            "var": var,
            "label": meta.get("label", ""),
            "concept": meta.get("concept", ""),
            "predicateType": meta.get("predicateType", ""),
        })
    return pd.DataFrame(rows)

def to_int(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
    return df


# -----------------------------
# 1) Fetch PL blocks (P2 + P12), tract-by-tract
# -----------------------------
def get_ri_tract_list_from_pl() -> List[Tuple[str, str, str]]:
    df = census_get(
        PL_ENDPOINT,
        params={"get": "state,county,tract", "for": "tract:*", "in": f"state:{STATE_FIPS}"},
        timeout=180,
    ).sort_values(["county", "tract"])
    return list(df.itertuples(index=False, name=None))

def fetch_pl_blocks_p2_p12_ri() -> pd.DataFrame:
    tracts = get_ri_tract_list_from_pl()
    parts: List[pd.DataFrame] = []

    for (state, county, tract) in tracts:
        p2 = census_get(
            PL_ENDPOINT,
            params={"get": "group(P2)", "for": "block:*", "in": f"state:{state} county:{county} tract:{tract}"},
            timeout=240,
        )
        p12 = census_get(
            PL_ENDPOINT,
            params={"get": "group(P12)", "for": "block:*", "in": f"state:{state} county:{county} tract:{tract}"},
            timeout=240,
        )

        keys = ["state", "county", "tract", "block"]
        merged = p2.merge(p12, on=keys, how="outer")
        merged["GEOID"] = (
            merged["state"].astype(str).str.zfill(2)
            + merged["county"].astype(str).str.zfill(3)
            + merged["tract"].astype(str).str.zfill(6)
            + merged["block"].astype(str).str.zfill(4)
        )
        parts.append(merged)
        time.sleep(0.15)

    df = pd.concat(parts, ignore_index=True)

    # Keep GEOID first
    geo_cols = ["GEOID", "state", "county", "tract", "block"]
    df = df[geo_cols + [c for c in df.columns if c not in geo_cols]]

    df.to_parquet(OUTDIR / "pl2020_ri_blocks_p2_p12_raw.parquet", index=False)
    return df


# -----------------------------
# 2) Fetch ACS BG (B27010)
# -----------------------------
def fetch_acs_b27010_blockgroups_ri() -> pd.DataFrame:
    df = census_get(
        ACS_ENDPOINT,
        params={"get": "NAME,group(B27010)", "for": "block group:*", "in": f"state:{STATE_FIPS} county:* tract:*"},
        timeout=240,
    )
    df["GEOID"] = (
        df["state"].astype(str).str.zfill(2)
        + df["county"].astype(str).str.zfill(3)
        + df["tract"].astype(str).str.zfill(6)
        + df["block group"].astype(str).str.zfill(1)
    )

    geo_cols = ["GEOID", "NAME", "state", "county", "tract", "block group"]
    df = df[geo_cols + [c for c in df.columns if c not in geo_cols]]

    df.to_parquet(OUTDIR / f"acs{ACS_YEAR}_ri_blockgroups_b27010_raw.parquet", index=False)
    return df


# -----------------------------
# 3) Build race/ethnicity categories from PL P2 (label-driven)
# -----------------------------
def pick_p2_vars(pl_vars: pd.DataFrame) -> Dict[str, str]:
    """
    Return mapping of desired categories -> P2 variable name (the _N form).
    We match on labels, not hardcode IDs, so it stays robust.
    """
    # Focus only on numeric count vars from P2
    p2 = pl_vars[pl_vars["concept"].fillna("").str.contains("P2", na=False)].copy()
    p2 = p2[p2["predicateType"].eq("int")]

    def find_one(pattern: str) -> str:
        m = p2[p2["label"].str.contains(pattern, flags=re.IGNORECASE, regex=True, na=False)]
        if len(m) == 0:
            raise RuntimeError(f"Could not find P2 var matching pattern: {pattern}")
        # Prefer _N vars (count) if present
        m2 = m[m["var"].str.endswith("N")]
        row = (m2.iloc[0] if len(m2) else m.iloc[0])
        return row["var"]

    # Total population (P2 table total)
    total = find_one(r"^Total:?$|Total$")

    # Hispanic or Latino (any race)
    hisp = find_one(r"Hispanic or Latino$")

    # Non-Hispanic race single categories (Not Hispanic or Latino: <race alone>)
    white = find_one(r"Not Hispanic or Latino.*White alone$")
    black = find_one(r"Not Hispanic or Latino.*Black or African American alone$")
    aian  = find_one(r"Not Hispanic or Latino.*American Indian and Alaska Native alone$")
    asian = find_one(r"Not Hispanic or Latino.*Asian alone$")
    nhpi  = find_one(r"Not Hispanic or Latino.*Native Hawaiian and Other Pacific Islander alone$")
    other = find_one(r"Not Hispanic or Latino.*Some Other Race alone$")
    multi = find_one(r"Not Hispanic or Latino.*Two or More Races$")

    return {
        "p2_total": total,
        "hispanic_any_race": hisp,
        "white_nh": white,
        "black_nh": black,
        "aian_nh": aian,
        "asian_nh": asian,
        "nhpi_nh": nhpi,
        "other_race_nh": other,
        "multiple_race_nh": multi,
    }

def add_race_ethnicity_categories(pl_blocks: pd.DataFrame, p2_map: Dict[str, str]) -> pd.DataFrame:
    need = list(p2_map.values())
    pl_blocks = to_int(pl_blocks, need)

    pl_blocks["pop_total"] = pl_blocks[p2_map["p2_total"]]
    pl_blocks["race_hispanic_latino_any"] = pl_blocks[p2_map["hispanic_any_race"]]
    pl_blocks["race_aian_nh"] = pl_blocks[p2_map["aian_nh"]]
    pl_blocks["race_asian_nh"] = pl_blocks[p2_map["asian_nh"]]
    pl_blocks["race_black_nh"] = pl_blocks[p2_map["black_nh"]]
    pl_blocks["race_nhpi_nh"] = pl_blocks[p2_map["nhpi_nh"]]
    pl_blocks["race_white_nh"] = pl_blocks[p2_map["white_nh"]]
    pl_blocks["race_other_nh"] = pl_blocks[p2_map["other_race_nh"]]
    pl_blocks["race_multiple_nh"] = pl_blocks[p2_map["multiple_race_nh"]]

    # Optional check: do these sum to total? (Hispanic includes any race, so NOT additive with NH groups.)
    # If you need "Non-Hispanic total", compute:
    # pl_blocks["non_hisp_total"] = pl_blocks["pop_total"] - pl_blocks["race_hispanic_latino_any"]

    return pl_blocks


# -----------------------------
# 4) Build age bins from PL P12 (sex-by-age), label-driven
# -----------------------------
def build_p12_agegroup_var_map(pl_vars: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    Map age-group label -> {'male': var, 'female': var} using P12 labels.

    We will pick P12 vars where labels include:
    - "Male:" or "Female:"
    - an age group phrase (e.g., "Under 5 years", "5 to 9 years", etc.)
    """
    p12 = pl_vars[pl_vars["concept"].fillna("").str.contains("P12", na=False)].copy()
    p12 = p12[p12["predicateType"].eq("int")]

    age_map: Dict[str, Dict[str, str]] = {}

    for _, row in p12.iterrows():
        var = row["var"]
        label = row["label"] or ""

        # Only count vars
        if not var.endswith("N"):
            continue

        # Match sex
        sex = None
        if re.search(r"\bMale:", label, flags=re.IGNORECASE):
            sex = "male"
        elif re.search(r"\bFemale:", label, flags=re.IGNORECASE):
            sex = "female"
        else:
            continue

        # Extract the age-group text after "Male:" / "Female:"
        m = re.split(r"Male:|Female:", label, flags=re.IGNORECASE)
        if len(m) < 2:
            continue
        age_text = m[1].strip()

        # Skip totals (like "Total population")
        if re.fullmatch(r"Total:?$", age_text, flags=re.IGNORECASE):
            continue

        # Normalize age_text
        age_text = re.sub(r"\s+", " ", age_text)

        age_map.setdefault(age_text, {})[sex] = var

    # Keep only age groups that have both sexes
    age_map = {k: v for k, v in age_map.items() if "male" in v and "female" in v}
    if not age_map:
        raise RuntimeError("Could not build age-group map from P12 labels; check endpoint variables.json.")
    return age_map

def add_age_bins(pl_blocks: pd.DataFrame, age_map: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """
    Create:
    - age_* single age-group totals (sex-summed) for all P12 age groups
    - then create collapsed bins:
        pediatric detail: <1, 1–4, 5–9, 10–14, 15–17
        main bins: 0–4, 5–11, 12–17, 18–24, 25–44, 45–64, 65+
    """
    # Create a column per P12 age-group label (sex-summed)
    # Name safely
    def safe_col(s: str) -> str:
        s2 = s.lower()
        s2 = re.sub(r"[^a-z0-9]+", "_", s2).strip("_")
        return f"agegrp_{s2}"

    # Convert needed columns to int
    needed_vars = []
    for v in age_map.values():
        needed_vars.extend([v["male"], v["female"]])
    pl_blocks = to_int(pl_blocks, needed_vars)

    agegrp_cols = {}
    for age_label, sexes in age_map.items():
        col = safe_col(age_label)
        pl_blocks[col] = pl_blocks[sexes["male"]] + pl_blocks[sexes["female"]]
        agegrp_cols[age_label] = col

    # Helper to sum matching age group labels
    def sum_groups(patterns: List[str], outcol: str) -> None:
        cols = []
        for pat in patterns:
            for age_label, col in agegrp_cols.items():
                if re.search(pat, age_label, flags=re.IGNORECASE):
                    cols.append(col)
        if not cols:
            raise RuntimeError(f"Age bin '{outcol}' matched no P12 age groups. Patterns: {patterns}")
        pl_blocks[outcol] = pl_blocks[cols].sum(axis=1).astype("int64")

    # Pediatric detail bins (these depend on P12 label wording; patterns cover common phrasings)
    sum_groups([r"Under 1 year", r"Under 1 year(s)?"], "age_u1")
    sum_groups([r"1 year", r"2 year", r"3 year", r"4 year"], "age_1_4")
    sum_groups([r"5 to 9 years"], "age_5_9")
    sum_groups([r"10 to 14 years"], "age_10_14")
    # 15–17 may be split (15–17) or (15–17) style; handle both
    sum_groups([r"15 to 17 years", r"15 to 19 years"], "age_15_17_or_15_19_note")

    # If P12 uses 15–19, we’ll split later when possible; otherwise we keep as is.
    # For your project, you likely want <18 precisely; if your table gives 15–17, great.
    # If it only gives 15–19, we’ll keep 15–19 as the best available bin.
    # We’ll build the main bins using label-based patterns directly for stability.

    # Main bins (more stable across label variants)
    sum_groups([r"Under 5 years"], "age_0_4")
    # 5–11 = 5–9 + 10–14 (partial) is messy; better: 5–9 plus 10–14 minus 12–14 if needed.
    # But P12 usually provides 5–9 and 10–14, 15–17 etc. We'll do a best-effort stable set:

    # 5–11: 5–9 + (10–14 minus 12–14) isn't possible without 1-year ages.
    # So we implement a stable alternative: 5–9 + (10–14 * 2/5) is NOT acceptable.
    # Instead we offer bins that are fully supported by P12 standard groupings:
    # 0–4, 5–9, 10–14, 15–17/15–19, 18–24, 25–44, 45–64, 65+
    # And we also create 5–11 and 12–17 only if the table provides those exact groups.

    # Preferred: exact 12–17 group if present
    try:
        sum_groups([r"12 to 14 years", r"15 to 17 years"], "age_12_17")
        sum_groups([r"5 to 9 years", r"10 to 11 years"], "age_5_11")  # rare
    except RuntimeError:
        # If exact groups aren't present, keep the stable bins only.
        pl_blocks["age_12_17"] = 0
        pl_blocks["age_5_11"] = 0

    sum_groups([r"18 to 24 years"], "age_18_24")
    sum_groups([r"25 to 44 years"], "age_25_44")
    sum_groups([r"45 to 64 years"], "age_45_64")
    sum_groups([r"65 to 74 years", r"75 to 84 years", r"85 years and over"], "age_65_plus")

    # Save stable bins that you can always use
    # If you prefer exactly 5–11 and 12–17, tell me and I’ll adjust to a different source table.
    return pl_blocks


# -----------------------------
# 5) Insurance totals from ACS B27010 (label-driven)
# -----------------------------
def add_insurance_totals(acs_bg: pd.DataFrame, acs_vars: pd.DataFrame) -> pd.DataFrame:
    """
    Create:
    - ins_total
    - ins_uninsured_total (sum of all cells labeled "No health insurance coverage")
    - ins_insured_total = total - uninsured
    - pct_uninsured

    Uses label matching on B27010 variables to find all uninsured line items.
    """
    # Total var
    total_var = acs_vars[
        (acs_vars["concept"].fillna("").str.contains("B27010", na=False)) &
        (acs_vars["label"].fillna("").str.fullmatch(r"Estimate!!Total", case=False, na=False))
    ]
    if len(total_var) == 0:
        # fallback: common total var name
        total_name = "B27010_001E"
    else:
        total_name = total_var.iloc[0]["var"]

    # All uninsured estimate vars
    uninsured = acs_vars[
        (acs_vars["concept"].fillna("").str.contains("B27010", na=False)) &
        (acs_vars["label"].fillna("").str.contains("No health insurance coverage", case=False, na=False)) &
        (acs_vars["var"].str.endswith("E"))
    ]["var"].tolist()

    if not uninsured:
        raise RuntimeError("Could not find any B27010 uninsured variables by label. Check ACS variables metadata.")

    # Convert to ints
    acs_bg = to_int(acs_bg, [total_name] + uninsured)

    acs_bg["ins_total"] = acs_bg[total_name]
    acs_bg["ins_uninsured_total"] = acs_bg[uninsured].sum(axis=1).astype("int64")
    acs_bg["ins_insured_total"] = (acs_bg["ins_total"] - acs_bg["ins_uninsured_total"]).clip(lower=0).astype("int64")
    acs_bg["pct_uninsured"] = (acs_bg["ins_uninsured_total"] / acs_bg["ins_total"].replace(0, pd.NA)).astype("float")

    return acs_bg


# -----------------------------
# 6) Summaries: state + county totals
# -----------------------------
def summarize_pl(pl_blocks: pd.DataFrame) -> None:
    cols = [
        "pop_total",
        "race_hispanic_latino_any",
        "race_aian_nh","race_asian_nh","race_black_nh","race_nhpi_nh","race_white_nh","race_other_nh","race_multiple_nh",
        "age_u1","age_1_4","age_5_9","age_10_14","age_15_17_or_15_19_note",
        "age_0_4","age_18_24","age_25_44","age_45_64","age_65_plus",
        "age_5_11","age_12_17"
    ]
    cols = [c for c in cols if c in pl_blocks.columns]

    # Statewide totals
    st = pl_blocks[cols].sum(numeric_only=True).to_frame().T
    st.insert(0, "geography", "Rhode Island (statewide)")
    st.to_csv(OUTDIR / "summary_pl2020_statewide.csv", index=False)

    # County totals
    ct = pl_blocks.groupby(["state","county"], as_index=False)[cols].sum(numeric_only=True)
    ct["geography"] = "County FIPS " + ct["county"].astype(str).str.zfill(3)
    ct = ct[["geography","state","county"] + cols]
    ct.to_csv(OUTDIR / "summary_pl2020_counties.csv", index=False)

def summarize_acs(acs_bg: pd.DataFrame) -> None:
    cols = ["ins_total","ins_insured_total","ins_uninsured_total"]
    cols = [c for c in cols if c in acs_bg.columns]

    # Statewide
    st = acs_bg[cols].sum(numeric_only=True).to_frame().T
    st.insert(0, "geography", "Rhode Island (statewide)")
    st["pct_uninsured"] = st["ins_uninsured_total"] / st["ins_total"].replace(0, pd.NA)
    st.to_csv(OUTDIR / f"summary_acs{ACS_YEAR}_insurance_statewide.csv", index=False)

    # Counties
    ct = acs_bg.groupby(["state","county"], as_index=False)[cols].sum(numeric_only=True)
    ct["pct_uninsured"] = ct["ins_uninsured_total"] / ct["ins_total"].replace(0, pd.NA)
    ct["geography"] = "County FIPS " + ct["county"].astype(str).str.zfill(3)
    ct = ct[["geography","state","county"] + cols + ["pct_uninsured"]]
    ct.to_csv(OUTDIR / f"summary_acs{ACS_YEAR}_insurance_counties.csv", index=False)


# -----------------------------
# Main
# -----------------------------
def main():
    print("Downloading variables metadata...")
    pl_vars = get_variables(PL_ENDPOINT)
    acs_vars = get_variables(ACS_ENDPOINT)
    print("  done.")

    print("\nFetching 2020 PL block-level P2 + P12 for Rhode Island (tract-by-tract)...")
    pl_blocks = fetch_pl_blocks_p2_p12_ri()
    print(f"  fetched {len(pl_blocks):,} block rows.")

    print("\nCreating race/ethnicity categories from P2 (label-driven)...")
    p2_map = pick_p2_vars(pl_vars)
    pl_blocks = add_race_ethnicity_categories(pl_blocks, p2_map)
    print("  done.")

    print("\nCreating age bins from P12 (label-driven)...")
    age_map = build_p12_agegroup_var_map(pl_vars)
    pl_blocks = add_age_bins(pl_blocks, age_map)
    print("  done.")

    # Save enriched PL
    pl_out = OUTDIR / "pl2020_ri_blocks_enriched.parquet"
    pl_blocks.to_parquet(pl_out, index=False)
    print(f"\nSaved enriched PL blocks -> {pl_out}")

    print(f"\nFetching ACS {ACS_YEAR} 5-year B27010 for Rhode Island block groups...")
    acs_bg = fetch_acs_b27010_blockgroups_ri()
    print(f"  fetched {len(acs_bg):,} block group rows.")

    print("\nCreating insurance totals (insured/uninsured) from B27010 (label-driven)...")
    acs_bg = add_insurance_totals(acs_bg, acs_vars)
    acs_out = OUTDIR / f"acs{ACS_YEAR}_ri_blockgroups_b27010_enriched.parquet"
    acs_bg.to_parquet(acs_out, index=False)
    print(f"Saved enriched ACS BG -> {acs_out}")

    print("\nWriting statewide + county summaries...")
    summarize_pl(pl_blocks)
    summarize_acs(acs_bg)
    print("  done.")

    print("\nAll outputs are in:", OUTDIR.resolve())
    print("\nFiles created:")
    print(" - pl2020_ri_blocks_p2_p12_raw.parquet")
    print(" - pl2020_ri_blocks_enriched.parquet")
    print(f" - acs{ACS_YEAR}_ri_blockgroups_b27010_raw.parquet")
    print(f" - acs{ACS_YEAR}_ri_blockgroups_b27010_enriched.parquet")
    print(" - summary_pl2020_statewide.csv")
    print(" - summary_pl2020_counties.csv")
    print(f" - summary_acs{ACS_YEAR}_insurance_statewide.csv")
    print(f" - summary_acs{ACS_YEAR}_insurance_counties.csv")


if __name__ == "__main__":
    main()
