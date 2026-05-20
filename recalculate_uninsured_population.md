# Sewershed-Level Population and Uninsured Population Methodology

This repository documents the workflow used to estimate:

- Total population served by each wastewater sewershed
- Estimated number of uninsured residents
- Estimated uninsured percentage

using census tract demographic data and spatial allocation methods.

---

# Data Sources

| Dataset | Source | Notes |
|---|---|---|
| Census tract demographic data | U.S. Census / ACS | Includes total population and uninsured population counts |
| Sewershed boundaries | Local utility / agency | Wastewater treatment plant service areas |
| Coordinate system | NAD 1983 (2011) StatePlane Rhode Island FIPS 3800 (US Feet) | Used for all spatial processing |

---

# Variables Used

| Variable | Description |
|---|---|
| `FIPS` | Census tract identifier |
| `E_TOTPOP` | Estimated total tract population |
| `E_UNINSUR` | Estimated count of uninsured persons |
| `AREA_SQMI` | Census tract area in square miles |

> **Note:** Count variables were used for uninsured estimation instead of percentage variables.

---

# Processing Workflow

## 1. Data Preparation

Prepare the census tract demographic dataset and sewershed polygon dataset.

Retain the following variables:

```text
FIPS
E_TOTPOP
E_UNINSUR
AREA_SQMI
```

---

## 2. Coordinate System Standardization

Project both datasets into:

```text
NAD 1983 (2011) StatePlane Rhode Island FIPS 3800 (US Feet)
```

This ensures accurate spatial overlay and area calculations.

---

## 3. Spatial Overlay (Intersect)

Perform an **Intersect** operation between:

```text
Census tract polygons
Sewershed polygons
```

The intersect output should contain one record for each tract–sewershed overlap area.

Retain:

```text
FIPS
Sewershed ID or Name
E_TOTPOP
E_UNINSUR
AREA_SQMI
```

---

## 4. Calculate Overlap Area

Calculate overlap area in square miles:

```text
AREA_OVLP_SQMI = shape_area_sqft / 27,878,400
```

where:

```text
27,878,400 = square feet in one square mile
```

---

## 5. Allocate Total Population

Estimate the portion of tract population within each overlap area:

```text
POP_ALLOC = E_TOTPOP × (AREA_OVLP_SQMI / AREA_SQMI)
```

This assumes that population is uniformly distributed across each census tract.

---

## 6. Allocate Uninsured Population

Estimate the portion of uninsured residents within each overlap area:

```text
UNINSUR_ALLOC = E_UNINSUR × (AREA_OVLP_SQMI / AREA_SQMI)
```

This uses the same area-weighted allocation approach applied to total population.

---

## 7. Aggregate by Sewershed

Group intersected records by sewershed.

Calculate:

```text
SUM_POP_ALLOC = Σ POP_ALLOC
```

```text
SUM_UNINSUR_ALLOC = Σ UNINSUR_ALLOC
```

---

## 8. Calculate Sewershed Uninsured Percentage

Calculate the estimated uninsured rate for each sewershed:

```text
PCT_UNINSUR_SEW = (SUM_UNINSUR_ALLOC / SUM_POP_ALLOC) × 100
```

---

# Quality Control

## Population Comparison

If independent sewershed population estimates are available:

```text
PCT_ERROR = ((SUM_POP_ALLOC − POP_SEWERSHED) / POP_SEWERSHED) × 100
```

## Validation Checks

Verify:

```text
SUM_POP_ALLOC > 0
SUM_UNINSUR_ALLOC <= SUM_POP_ALLOC
PCT_UNINSUR_SEW between 0 and 100
```

Review records where:

```text
AREA_OVLP_SQMI > AREA_SQMI
Negative or null values occur
E_UNINSUR > E_TOTPOP
```

---

# Final Output Table

| Field | Description |
|---|---|
| Sewershed ID/Name | Wastewater catchment identifier |
| `SUM_POP_ALLOC` | Estimated total population served |
| `SUM_UNINSUR_ALLOC` | Estimated uninsured population |
| `PCT_UNINSUR_SEW` | Estimated uninsured percentage |
| `PCT_ERROR` | Optional validation metric |

---

# Interpretation

- `SUM_POP_ALLOC` → estimated population served by the sewershed
- `SUM_UNINSUR_ALLOC` → estimated uninsured population within the sewershed
- `PCT_UNINSUR_SEW` → estimated uninsured percentage of the sewershed population

---

# Limitations

- Assumes total population and uninsured residents are uniformly distributed within each census tract
- Boundary mismatch between census tracts and sewersheds may introduce uncertainty
- Localized demographic variation within tracts is not captured
- Missing or suppressed demographic values should be reviewed before aggregation
