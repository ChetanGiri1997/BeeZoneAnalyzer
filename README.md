# BeeZoneAnalyzer

A deterministic geospatial suitability engine for co-locating **bee farming** and **mandarin orange cultivation**. Given a farm boundary (KML), it ingests SRTM elevation data and Landsat/Sentinel-2 imagery to produce scored probability maps, ranked zone polygons, and KML overlays — no machine learning, no ground-truth labels required.

---

## Table of contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Installation](#installation)
- [Configuration](#configuration)
- [Suitability models](#suitability-models)
- [Running the analysis](#running-the-analysis)
- [Outputs](#outputs)
- [Example outputs](#example-outputs)
- [License](#license)

---

## Overview

BeeZoneAnalyzer answers a single question: **given a bee farm location, where within a 4 km radius are the best spots for bees to thrive — and can mandarin orange trees grow there too?**

It does this by:

1. Parsing a Google Earth KML polygon as the area of interest
2. Deriving terrain layers (slope, aspect, fog, solar exposure, wind exposure) from USGS SRTM DEM data
3. Computing vegetation indices (NDVI, NDWI, EVI, SAVI) from Landsat 8/9 or Sentinel-2 imagery — with a synthetic fallback if no imagery is available
4. Scoring each 100 m grid cell against two weighted ecological models
5. Exporting GeoTIFF suitability rasters, KML zone polygons, and JSON reports

The models are fully transparent and parameterised via `config.yaml`. No black-box ML — every score is traceable back to a formula.

---

## Pipeline

```
InitialFarm.kml
      │
      ▼
┌─────────────────┐     ┌───────────────────────┐
│   KML Parser    │────▶│   Search bounds (4km) │
└─────────────────┘     └───────────┬───────────┘
                                    │
              ┌─────────────────────┼──────────────────────┐
              ▼                     ▼                      ▼
   ┌──────────────────┐  ┌─────────────────────┐  ┌──────────────────┐
   │  Terrain Analyzer │  │ Satellite Processor │  │ Temperature Model│
   │  (USGS SRTM DEM) │  │ (Landsat / S2-L2A)  │  │ (lapse rate est) │
   │                  │  │                     │  └────────┬─────────┘
   │ • elevation      │  │ • NDVI              │           │
   │ • slope          │  │ • NDWI              │           │
   │ • aspect         │  │ • EVI               │           │
   │ • fog proxy      │  │ • SAVI              │           │
   │ • solar exposure │  │ • veg. stability    │           │
   │ • wind exposure  │  └──────────┬──────────┘           │
   │ • elev variation │             │                      │
   └────────┬─────────┘             │                      │
            └─────────────┬─────────┘──────────────────────┘
                          ▼
           ┌──────────────────────────────┐
           │   EcologicalSuitabilityModel │
           │                              │
           │  BeeSuitabilityScore (raster)│
           │  MandarinSuitabilityScore    │
           │  CombinedScore (50/50)       │
           └──────────────┬───────────────┘
                          │
              ┌───────────┼─────────────┐
              ▼           ▼             ▼
        GeoTIFF       Top-10         JSON
        rasters       KML zones      report
```

---

## Installation

Python 3.10+ is required. GDAL must be installed system-wide before the pip dependencies.

**Ubuntu/Debian:**
```bash
sudo apt-get install gdal-bin libgdal-dev python3-gdal
```

**Fedora/RHEL:**
```bash
sudo dnf install gdal gdal-devel python3-gdal
```

**macOS (Homebrew):**
```bash
brew install gdal
```

Then install the Python dependencies:
```bash
git clone https://github.com/ChetanGiri1997/BeeZoneAnalyzer.git
cd BeeZoneAnalyzer
pip install -r requirements.txt
```

If you have an NVIDIA GPU and want raster processing acceleration, uncomment the `torch` lines at the bottom of `requirements.txt` and ensure CUDA 12.x drivers are installed.

---

## Data requirements

Before running, place the following files in the project directory:

| File | Description | Source |
|------|-------------|--------|
| `InitialFarm.kml` | Farm boundary polygon (Google Earth KML) | Draw in Google Earth, export as KML |
| `USGS/n28_e081_1arc_v3.dt2` | SRTM 1-arc-second DEM tile | [USGS EarthExplorer](https://earthexplorer.usgs.gov/) |
| `USGS/LC0*_L2SP_*.tar` | Landsat 8/9 Level-2 scene (optional) | [USGS EarthExplorer](https://earthexplorer.usgs.gov/) |
| `sentinal2/S2*_MSIL2A_*.zip` | Sentinel-2 Level-2A scene (optional) | [Copernicus Browser](https://browser.dataspace.copernicus.eu/) |

The DEM tile filename encodes its geographic bounds (e.g. `n28_e081` = 28–29°N, 81–82°E). Download the tile(s) covering your farm location. If no satellite imagery is provided, the engine generates a synthetic NDVI layer derived from the elevation model — results will be lower fidelity but the pipeline will still run end-to-end.

---

## Configuration

All analysis parameters live in `config.yaml`. The key sections are:

### Search and grid parameters

```yaml
search_radius_km: 4.0      # Aerial radius from farm center to search
grid_resolution_m: 100     # Grid cell size for suitability scoring (100m × 100m)
```

### Bee farming criteria

```yaml
bee_farming:
  elevation:
    min: 500          # Hard lower bound (m) — bees won't establish below this
    max: 2500         # Hard upper bound (m)
    optimal_min: 1000 # Start of optimal plateau
    optimal_max: 1800 # End of optimal plateau
    weight: 0.20      # Contribution to final score (must sum to 1.0 across all criteria)
  temperature:
    min: 15           # Hard limits (°C)
    max: 35
    optimal_min: 20
    optimal_max: 28
    weight: 0.25
  vegetation_ndvi:
    min: 0.3          # Below this = too sparse for foraging
    optimal_min: 0.5  # Moderate-high vegetation preferred (diverse flowering plants)
    weight: 0.25
  slope:
    max: 15           # Hard upper limit (degrees)
    optimal_max: 10   # Flat-to-gentle preferred for hive placement
    weight: 0.15
  water_proximity_km:
    max: 2.0
    optimal_max: 1.0
    weight: 0.15
```

### Mandarin farming criteria

```yaml
mandarin_farming:
  elevation:
    min: 800
    max: 2000
    optimal_min: 1000
    optimal_max: 1500
    weight: 0.20
  temperature:
    min: 13
    max: 30
    optimal_min: 20
    optimal_max: 25
    weight: 0.25
  slope:
    min: 5            # Needs some slope for drainage
    max: 20
    optimal_min: 8
    optimal_max: 15
    weight: 0.20
  aspect:
    preferred: ["S", "SE", "SW"]   # South-facing for winter sun (northern hemisphere)
    weight: 0.15
  soil_moisture:
    min: 0.2
    max: 0.7
    optimal_min: 0.3
    optimal_max: 0.6
    weight: 0.20
```

### Output paths

```yaml
output:
  report_html: "analysis_report.html"
  results_json: "results.json"
  recommended_kml: "recommended_locations.kml"
  bee_farming_kml: "bee_farming_suitable.kml"
  mandarin_farming_kml: "mandarin_farming_suitable.kml"
  maps_dir: "suitability_maps"
```

---

## Suitability models

### Bee suitability score

Each 100 m grid cell receives a score in [0, 1] from the following weighted formula:

```
BeeSuitabilityScore =
    0.20 × NDVI_score
  + 0.15 × Elevation_optimality
  + 0.15 × SolarExposure
  + 0.15 × WindExposure_inverse       ← 1 − wind_exposure (bees prefer calm)
  + 0.10 × Temperature_optimality
  + 0.10 × Fog_inverse                ← 1 − fog_persistence
  + 0.10 × Vegetation_stability       ← spatial NDVI variance proxy for flowering diversity
  + 0.05 × Slope_optimality
```

### Mandarin suitability score

```
MandarinSuitabilityScore =
    0.25 × Elevation_optimality
  + 0.20 × Temperature_optimality
  + 0.15 × SolarExposure
  + 0.15 × Slope_optimality
  + 0.10 × NDVI_score
  + 0.10 × Fog_inverse
  + 0.05 × FrostRisk_inverse          ← 1 − fog_persistence (valley cold-sinks penalised)
```

### Combined co-location score

```
CombinedScore = 0.5 × BeeSuitabilityScore + 0.5 × MandarinSuitabilityScore
```

### Scoring functions

All component scores use a **triangular (tent) function** — linear ramp from 0 at the hard boundary to 1 at the optimal range, then flat across the optimal plateau, then linear decline back to 0 at the far hard boundary. For criteria with only an upper bound (slope for bees), the function is a one-sided ramp.

### Derived terrain layers

| Layer | Method |
|-------|--------|
| Slope | Gradient of DEM (numpy.gradient), converted to degrees |
| Aspect | arctan2 of DEM gradient, 0° = North |
| Fog persistence proxy | 0.4 × valley_depth + 0.3 × north_facing + 0.3 × low_elevation_norm |
| Solar exposure index | 0.4 × south_facing_score + 0.35 × optimal_slope_score + 0.25 × elevation_norm |
| Wind exposure | 0.5 × ridge_position (RTP) + 0.3 × slope_norm + 0.2 × elevation_norm |
| Elevation variation (1 km) | Range of elevation within 1 km radius — bee energy cost proxy |
| Vegetation stability | Spatial variance of NDVI within a 5-pixel window — flowering diversity proxy |

Temperature is estimated from the elevation lapse rate: `T = 25°C − 0.0065 × elevation_m`, adjusted by `+2°C × solar_exposure`.

### Classification thresholds

| Score range | Classification |
|-------------|----------------|
| ≥ 0.80 | Excellent |
| 0.60 – 0.80 | High |
| 0.40 – 0.60 | Moderate |
| < 0.40 | Low |

---

## Running the analysis

The recommended entry point is `run_analysis.py`, which validates inputs, converts the DEM if needed, and calls the full pipeline:

```bash
python run_analysis.py
```

For direct control over the pipeline:

```bash
python main_new.py --config config.yaml
```

To convert a raw SRTM `.hgt` or `.dt2` DEM tile to GeoTIFF manually:

```bash
python hgt_loader.py USGS/n28_e081_1arc_v3.dt2 USGS/n28_e081_1arc_v3.tif
```

To regenerate only the top-10 zone polygons from existing suitability rasters:

```bash
python generate_top10_zones.py
```

Expected runtime is 5–15 minutes depending on search radius, grid resolution, and whether a GPU is available.

---

## Outputs

After a successful run, the `suitability_maps/` directory contains:

| File | Description |
|------|-------------|
| `elevation.tif` | Cropped and clipped DEM |
| `slope.tif` | Slope in degrees |
| `aspect.tif` | Aspect in degrees (0 = North) |
| `fog_persistence_proxy.tif` | Fog proxy raster [0, 1] |
| `solar_exposure_index.tif` | Solar exposure raster [0, 1] |
| `wind_exposure.tif` | Wind exposure raster [0, 1] |
| `ndvi.tif` | NDVI from satellite imagery (or synthetic) |
| `bee_suitability.tif` | Bee suitability scores [0, 1] |
| `mandarin_suitability.tif` | Mandarin suitability scores [0, 1] |
| `combined_suitability.tif` | Combined co-location scores [0, 1] |
| `initialfarm_probability_report.json` | Full statistics and methodology metadata |
| `initialfarm_best_zones.kml` | Top 20% zone points, colour-coded by class |

The root directory also receives:

| File | Description |
|------|-------------|
| `bee_farming_suitable.kml` | Top 20 bee suitability point locations |
| `mandarin_farming_suitable.kml` | Top 20 mandarin suitability point locations |
| `recommended_locations.kml` | Combined comparison KML (both layers) |
| `initial_farm_analysis.kml` | Original farm polygon with suitability annotations |
| `results.json` | Machine-readable summary of all top locations |

All KML files can be opened directly in Google Earth Pro or QGIS.

---

## Example outputs

### Suitability report excerpt (`initialfarm_probability_report.json`)

```json
{
  "farm_name": "Bee Farm",
  "farm_center": { "lat": 28.776578, "lon": 81.480797 },
  "farm_area_hectares": 4.07,
  "bee_suitability": {
    "mean_score": 0.71,
    "classification": "High",
    "min_score": 0.44,
    "max_score": 0.89
  },
  "mandarin_suitability": {
    "mean_score": 0.63,
    "classification": "High",
    "min_score": 0.31,
    "max_score": 0.81
  },
  "combined_suitability": {
    "mean_score": 0.67,
    "classification": "High"
  }
}
```

### Top-10 zone output (from `generate_top10_zones.py`)

```
Top 10 Best Zones:
  1. Combined: 0.847, Bee: 0.881, Mandarin: 0.812, Area: 8.3 ha
  2. Combined: 0.831, Bee: 0.863, Mandarin: 0.798, Area: 5.1 ha
  3. Combined: 0.819, Bee: 0.845, Mandarin: 0.792, Area: 3.7 ha
  ...
```

### Viewing in QGIS

1. Open QGIS → Layer → Add Layer → Add Raster Layer → select `combined_suitability.tif`
2. In the layer properties, set the render type to Singleband pseudocolor with a green ramp
3. Add `InitialFarm.kml` as a vector overlay to compare suitability against your existing farm footprint

---

## License

MIT License — see `LICENSE` for details. Satellite data (Landsat, Sentinel-2) and DEM tiles are sourced from USGS and Copernicus under their respective open-data licences.
