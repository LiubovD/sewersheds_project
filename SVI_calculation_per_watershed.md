📌 Sewershed Social Vulnerability Index (SVI) Methodology

This repository documents the process used to derive a population-weighted Social Vulnerability Index (SVI) for wastewater sewersheds in Rhode Island using the CDC/ATSDR SVI 2022 tract-level data.

The goal was to translate tract-level SVI scores into a meaningful, defensible sewershed-level vulnerability metric that reflects the social conditions of the people served by each wastewater catchment area.

📂 Data Sources
Dataset	Source	Notes
CDC/ATSDR SVI (2022)	CDC/ATSDR	Tract-level RPL_THEMES, population (E_TOTPOP), and area (AREA_SQMI)
Sewershed boundaries	Local utility / agency	Wastewater treatment plant catchments
Coordinate system	NAD 1983 (2011) StatePlane Rhode Island FIPS 3800 (US Feet)	Used for all spatial processing
🔧 Processing Workflow
1) Data Preparation

Downloaded the CDC/ATSDR SVI 2022 tract dataset for Rhode Island.

Assembled wastewater sewershed polygons representing service catchment areas.

Retained key SVI fields:

FIPS — tract identifier

RPL_THEMES — overall SVI percentile rank (0–1)

E_TOTPOP — estimated tract population

AREA_SQMI — tract area

2) Coordinate System Standardization

To ensure accurate area calculations, both layers were projected to:

NAD 1983 (2011) StatePlane Rhode Island FIPS 3800 (US Feet)


This projection minimizes spatial distortion across Rhode Island and supports valid area-based calculations.

3) Spatial Overlay (Intersect)

An Intersect operation was performed between:

SVI census tracts

Sewershed polygons

The result was a new feature class where each record represents the portion of a tract that lies within a specific sewershed.

4) Calculate Overlap Area

For each intersected fragment, overlap area was computed in square survey miles:

AREA_OVLP_SQMI = shape_area (sq ft) / 27,878,400


This unit matches the tract area units provided in the SVI dataset.

5) Allocate Population to Overlap Pieces

Assuming uniform population distribution within each tract, population was proportionally allocated to each fragment:

POP_ALLOC = E_TOTPOP × (AREA_OVLP_SQMI / AREA_SQMI)


This estimates how many people from each tract reside within each sewershed.

6) Compute Weighted SVI Contribution

Each overlap fragment’s contribution to sewershed vulnerability was calculated as:

W_RPL = POP_ALLOC × RPL_THEMES


This represents the “vulnerability carried by people” in each fragment.

7) Aggregate to Sewershed Level

Using Summary Statistics, values were summed by sewershed ID:

SUM_POP_ALLOC = Σ POP_ALLOC

SUM_W_RPL = Σ W_RPL

Final sewershed SVI percentile was computed as:

RPL_SEW = SUM_W_RPL / SUM_POP_ALLOC


This yields a population-weighted SVI score (0–1) for each sewershed.

8) Join Results Back to Sewersheds

The summarized table was joined back to the original sewershed polygons using a common sewershed ID, enabling:

Mapping

Visualization

Reporting

Further analysis

9) Quality Control (Population Check)

To assess the validity of population allocation, estimated population (SUM_POP_ALLOC) was compared to a provided sewershed population (pop_seward).

Percent error was calculated as:

PCT_ERROR = ((SUM_POP_ALLOC − pop_seward) / pop_seward) × 100


This step helps identify:

boundary mismatches

problematic tracts

areas where uniform population assumptions may break down

✅ Interpretation of Results

RPL_SEW ≈ 0 → very low social vulnerability

RPL_SEW ≈ 1 → very high social vulnerability

Values represent national percentile ranks based on CDC SVI.

Example language for users:

“A sewershed with RPL_SEW = 0.72 serves a population more socially vulnerable than approximately 72% of U.S. communities.”

⚠️ Limitations

Population allocation assumes uniform distribution within tracts.

Tracts with RPL_THEMES = -999 were treated as No Data and excluded.
