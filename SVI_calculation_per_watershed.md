📌 Sewershed Social Vulnerability Index (SVI) Methodology

This repository documents the process used to derive a population-weighted Social Vulnerability Index (SVI) for wastewater sewersheds in Rhode Island using the CDC/ATSDR SVI 2022 tract-level data.

The goal was to translate tract-level SVI metrics into a meaningful, defensible sewershed-level vulnerability indicator that reflects the social conditions of the population served by each wastewater catchment area.

📂 Data Sources
Dataset	Source	Notes
CDC/ATSDR SVI (2022)	Centers for Disease Control and Prevention	Tract-level RPL_THEMES (percentile rank), SPL_THEMES (sum of theme rankings), population (E_TOTPOP), and area (AREA_SQMI)
Sewershed boundaries	Local utility / agency	Wastewater treatment plant catchments
Coordinate system	NAD 1983 (2011) StatePlane Rhode Island FIPS 3800 (US Feet)	Used for all spatial processing
🔧 Processing Workflow
1) Data Preparation

The CDC/ATSDR SVI 2022 census tract dataset for Rhode Island was downloaded and prepared alongside sewershed boundary polygons representing wastewater service areas.

The following key variables were retained:

FIPS — census tract identifier
RPL_THEMES — overall SVI percentile rank (0–1)
SPL_THEMES — sum of theme-specific rankings (non-percentile; typically ranges ~0–16)
E_TOTPOP — estimated tract population
AREA_SQMI — tract area
2) Coordinate System Standardization

Both datasets were projected to:

NAD 1983 (2011) StatePlane Rhode Island FIPS 3800 (US Feet)

This minimizes spatial distortion and ensures accurate area-based calculations.

3) Spatial Overlay (Intersect)

An Intersect operation was performed between:

SVI census tracts
Sewershed polygons

The output represents the portion of each tract contained within each sewershed.

4) Calculate Overlap Area

For each intersected feature:

AREA_OVLP_SQMI = shape_area (sq ft) / 27,878,400

This converts area to square miles, consistent with CDC SVI data.

5) Allocate Population to Overlap Areas

Population was proportionally allocated assuming uniform distribution within each tract:

POP_ALLOC = E_TOTPOP × (AREA_OVLP_SQMI / AREA_SQMI)

This estimates the number of residents from each tract within each sewershed.

6) Compute Weighted SVI Contributions
Overall Percentile-Based SVI (Primary Metric)
W_RPL = POP_ALLOC × RPL_THEMES
Summed Vulnerability Score (SPL-Based Metric)
W_SPL = POP_ALLOC × SPL_THEMES
7) Aggregate to Sewershed Level

Using summary statistics grouped by sewershed:

SUM_POP_ALLOC = Σ POP_ALLOC
SUM_W_RPL     = Σ W_RPL
SUM_W_SPL     = Σ W_SPL

Final sewershed-level metrics:

RPL_SEW = SUM_W_RPL / SUM_POP_ALLOC
SPL_SEW = SUM_W_SPL / SUM_POP_ALLOC
8) Optional Normalization of SPL

Because SPL_THEMES is not a percentile, it may be normalized for interpretability:

SPL_SEW_NORM = SPL_SEW / max(SPL_SEW)

This rescales SPL values to a 0–1 range, enabling comparison with percentile-based metrics.

9) Join Results Back to Sewersheds

Aggregated results were joined back to the original sewershed polygons using a common ID, enabling:

Mapping
Visualization
Reporting
Spatial analysis
10) Quality Control (Population Validation)

Estimated population totals were compared to known sewershed populations:

PCT_ERROR = ((SUM_POP_ALLOC − pop_seward) / pop_seward) × 100

This step helps identify:

Boundary misalignment
Data inconsistencies
Limitations of the uniform population assumption
✅ Interpretation of Results
RPL-Based Metric (Recommended)
RPL_SEW ≈ 0 → very low social vulnerability
RPL_SEW ≈ 1 → very high social vulnerability

Example:

“A sewershed with RPL_SEW = 0.72 serves a population more socially vulnerable than approximately 72% of U.S. communities.”

SPL-Based Metric (Supplementary)
SPL_SEW reflects the aggregate magnitude of vulnerability
Not a percentile and should not be interpreted as a rank
Higher values indicate greater cumulative vulnerability burden

If normalized:

SPL_SEW_NORM (0–1) provides a relative comparison scale
⚠️ Key Methodological Notes
RPL_THEMES is the preferred metric for reporting because it represents a true percentile ranking
SPL_THEMES is a composite score, useful for understanding magnitude but not relative rank
These measures are not interchangeable and should be interpreted accordingly
⚠️ Limitations
Population allocation assumes uniform distribution within census tracts
Tracts with missing values (e.g., -999) were excluded
Sewer infrastructure boundaries may not perfectly align with census geography
SPL normalization depends on dataset-specific ranges
🧾 Summary

This methodology produces a population-weighted, spatially explicit measure of social vulnerability at the sewershed scale, enabling:

Public health prioritization
Infrastructure planning
Environmental justice analysis

The use of both RPL (percentile) and SPL (magnitude) metrics provides complementary perspectives on community vulnerability.
