# 📌 Sewershed Social Vulnerability Index (SVI) Methodology

This repository documents the process used to derive a population-weighted Social Vulnerability Index (SVI) for wastewater sewersheds in Rhode Island using the CDC/ATSDR SVI 2022 tract-level data.

The goal was to translate tract-level SVI metrics into a meaningful, defensible sewershed-level vulnerability indicator that reflects the social conditions of the population served by each wastewater catchment area.

---

## 📂 Data Sources

| Dataset | Source | Notes |
|--------|--------|------|
| CDC/ATSDR SVI (2022) | CDC/ATSDR | Tract-level RPL_THEMES (percentile rank), SPL_THEMES (sum of theme rankings), population (E_TOTPOP), and area (AREA_SQMI) |
| Sewershed boundaries | Local utility / agency | Wastewater treatment plant catchments |
| Coordinate system | NAD 1983 (2011) StatePlane Rhode Island FIPS 3800 (US Feet) | Used for all spatial processing |

---

## 🔧 Processing Workflow

### 1) Data Preparation

The CDC/ATSDR SVI 2022 census tract dataset for Rhode Island was downloaded and prepared alongside sewershed boundary polygons representing wastewater service areas.

The following key variables were retained:

- FIPS — census tract identifier  
- RPL_THEMES — overall SVI percentile rank (0–1)  
- SPL_THEMES — sum of theme-specific rankings (non-percentile; typically ranges ~0–16)  
- E_TOTPOP — estimated tract population  
- AREA_SQMI — tract area  

---

### 2) Coordinate System Standardization

Both datasets were projected to:

NAD 1983 (2011) StatePlane Rhode Island FIPS 3800 (US Feet)

---

### 3) Spatial Overlay (Intersect)

An Intersect operation was performed between:

- SVI census tracts  
- Sewershed polygons  

---

### 4) Calculate Overlap Area

AREA_OVLP_SQMI = shape_area (sq ft) / 27,878,400

---

### 5) Allocate Population

POP_ALLOC = E_TOTPOP × (AREA_OVLP_SQMI / AREA_SQMI)

---

### 6) Weighted Contributions

W_RPL = POP_ALLOC × RPL_THEMES  
W_SPL = POP_ALLOC × SPL_THEMES  

---

### 7) Aggregation

SUM_POP_ALLOC = Σ POP_ALLOC  
SUM_W_RPL = Σ W_RPL  
SUM_W_SPL = Σ W_SPL  

RPL_SEW = SUM_W_RPL / SUM_POP_ALLOC  
SPL_SEW = SUM_W_SPL / SUM_POP_ALLOC  

---

### 8) Optional Normalization

SPL_SEW_NORM = SPL_SEW / max(SPL_SEW)

---

### 9) Quality Control

PCT_ERROR = ((SUM_POP_ALLOC − pop_seward) / pop_seward) × 100

---

## ✅ Interpretation

RPL_SEW → percentile (0–1)  
SPL_SEW → magnitude (not percentile)

---

## ⚠️ Notes

- RPL is preferred for reporting  
- SPL reflects cumulative vulnerability  
- Not interchangeable  

---

## ⚠️ Limitations

- Uniform population assumption  
- Missing data excluded  
- Boundary mismatch possible  
