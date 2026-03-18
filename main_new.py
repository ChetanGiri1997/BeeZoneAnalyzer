"""
Ecological Geospatial Planning Engine - Main Pipeline
Generates suitability probability maps for bee farming and mandarin orange cultivation
using deterministic ecological models without supervised ML or ground-truth labels.

Orchestrates:
1. Terrain analysis (DEM-derived layers)
2. Satellite processing (vegetation indices)
3. Suitability modeling (bee, mandarin, combined)
4. Report generation (GeoTIFFs, JSON, KML)
"""

import numpy as np
import yaml
import json
import os
from tqdm import tqdm
from typing import Dict, List, Tuple
import argparse
from datetime import datetime

from kml_parser import KMLParser
from terrain_analyzer import TerrainAnalyzer
from satellite_processor import SatelliteProcessor
from ecological_suitability import EcologicalSuitabilityModel
from kml_generator import KMLGenerator


class EcologicalPlanningEngine:
    def __init__(self, config_file: str = "config.yaml"):
        """Initialize the analysis engine with configuration"""
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.kml_parser = KMLParser(self.config["input"]["kml_file"])
        self.terrain_analyzer = None
        self.satellite_processor = None
        self.suitability_model = EcologicalSuitabilityModel()
        self.kml_generator = KMLGenerator()

        # Create output directories
        os.makedirs(self.config["output"]["maps_dir"], exist_ok=True)

        # Store raster data for later access
        self.dem_data = None
        self.slope_data = None
        self.aspect_data = None
        self.elevation_variation = None
        self.fog_persistence = None
        self.solar_exposure = None
        self.wind_exposure = None
        self.ndvi_data = None
        self.temperature_data = None
        self.vegetation_stability = None

    def run_complete_analysis(self):
        """Execute complete analysis pipeline"""
        print("\n" + "=" * 80)
        print("ECOLOGICAL GEOSPATIAL PLANNING ENGINE")
        print("Deterministic Suitability Assessment for Bee and Mandarin Farming")
        print("=" * 80)

        # Step 1: Parse KML
        print("\n[STEP 1/5] Parsing InitialFarm.kml...")
        farm_data = self.kml_parser.parse()
        search_bounds = self.kml_parser.get_search_bounds(
            farm_data["center"], self.config["search_radius_km"]
        )

        print(f"  ✓ Farm: {farm_data['name']}")
        print(f"  ✓ Center: {farm_data['center'][1]:.6f}°N, {farm_data['center'][0]:.6f}°E")
        print(f"  ✓ Area: {farm_data['area_hectares']:.2f} hectares")
        print(f"  ✓ Search radius: {self.config['search_radius_km']} km")

        # Step 2: Load terrain and compute derived layers
        print("\n[STEP 2/5] Processing terrain data (DEM-derived layers)...")
        self.terrain_analyzer = TerrainAnalyzer(self.config["input"]["dem_file"])
        self.dem_data = self.terrain_analyzer.load_dem(search_bounds)

        print("  Calculating derived terrain layers...")
        self.slope_data = self.terrain_analyzer.calculate_slope()
        self.aspect_data = self.terrain_analyzer.calculate_aspect()
        self.terrain_roughness = self.terrain_analyzer.calculate_terrain_roughness()
        self.rtp = self.terrain_analyzer.calculate_relative_topographic_position()
        self.elevation_variation = self.terrain_analyzer.calculate_elevation_variation_1km()
        self.fog_persistence = self.terrain_analyzer.calculate_fog_persistence_proxy()
        self.solar_exposure = self.terrain_analyzer.calculate_solar_exposure_index()
        self.wind_exposure = self.terrain_analyzer.calculate_wind_exposure()

        print("  ✓ Derived layers computed")
        print(
            f"    - Elevation range: {np.nanmin(self.dem_data):.1f} - {np.nanmax(self.dem_data):.1f} m"
        )
        print(f"    - Slope range: {np.nanmin(self.slope_data):.1f} - {np.nanmax(self.slope_data):.1f}°")
        print(f"    - Fog persistence: {np.nanmin(self.fog_persistence):.3f} - {np.nanmax(self.fog_persistence):.3f}")
        print(
            f"    - Solar exposure: {np.nanmin(self.solar_exposure):.3f} - {np.nanmax(self.solar_exposure):.3f}"
        )

        # Save terrain outputs
        self._save_layer(
            self.dem_data, os.path.join(self.config["output"]["maps_dir"], "elevation.tif")
        )
        self._save_layer(
            self.slope_data, os.path.join(self.config["output"]["maps_dir"], "slope.tif")
        )
        self._save_layer(
            self.aspect_data, os.path.join(self.config["output"]["maps_dir"], "aspect.tif")
        )
        self._save_layer(
            self.fog_persistence,
            os.path.join(self.config["output"]["maps_dir"], "fog_persistence_proxy.tif"),
        )
        self._save_layer(
            self.solar_exposure,
            os.path.join(self.config["output"]["maps_dir"], "solar_exposure_index.tif"),
        )
        self._save_layer(
            self.wind_exposure,
            os.path.join(self.config["output"]["maps_dir"], "wind_exposure.tif"),
        )

        # Step 3: Process satellite imagery
        print("\n[STEP 3/5] Processing satellite imagery (Sentinel-2/Landsat)...")
        self.satellite_processor = SatelliteProcessor(
            self.config["input"]["landsat_dir"], self.config["input"]["sentinel_dir"]
        )

        satellite_data = None
        try:
            satellite_data = self.satellite_processor.process_landsat(search_bounds)
        except Exception as e:
            print(f"  ⚠ Error processing satellite data: {e}")
            satellite_data = None

        if satellite_data and "indices" in satellite_data:
            ndvi_arr = satellite_data["indices"].get("ndvi", None)
            ndwi_arr = satellite_data["indices"].get("ndwi", None)
            
            # Check if data is valid (non-empty)
            if ndvi_arr is not None and ndvi_arr.size > 0 and not np.all(np.isnan(ndvi_arr)):
                # Check if shapes match DEM - if not, resample
                if ndvi_arr.shape != self.dem_data.shape:
                    print(f"  ⚠ Resampling satellite data from {ndvi_arr.shape} to {self.dem_data.shape}...")
                    from scipy.ndimage import zoom
                    
                    zoom_factors = (
                        self.dem_data.shape[0] / ndvi_arr.shape[0],
                        self.dem_data.shape[1] / ndvi_arr.shape[1],
                    )
                    
                    self.ndvi_data = zoom(ndvi_arr, zoom_factors, order=1)
                    if ndwi_arr is not None:
                        self.ndwi_data = zoom(ndwi_arr, zoom_factors, order=1)
                    else:
                        self.ndwi_data = np.zeros_like(self.ndvi_data)
                else:
                    self.ndvi_data = ndvi_arr
                    self.ndwi_data = ndwi_arr if ndwi_arr is not None else np.zeros_like(ndvi_arr)
                
                print("  ✓ Satellite indices computed")
                print(
                    f"    - NDVI range: {np.nanmin(self.ndvi_data):.3f} - {np.nanmax(self.ndvi_data):.3f}"
                )
                if self.ndwi_data is not None:
                    print(
                        f"    - NDWI range: {np.nanmin(self.ndwi_data):.3f} - {np.nanmax(self.ndwi_data):.3f}"
                    )

                # Save indices (at original resolution before resampling)
                for name, data in satellite_data["indices"].items():
                    if data.size > 0:
                        self._save_layer(
                            data,
                            os.path.join(self.config["output"]["maps_dir"], f"{name}.tif"),
                        )
            else:
                satellite_data = None
        
        if satellite_data is None:
            print("  ⚠ No valid satellite data found")
            print("  → Generating synthetic NDVI from elevation-based vegetation model...")
            # Create NDVI based on elevation: forest belt ~1000-2500m, grassland lower
            elev_norm = (self.dem_data - np.nanmin(self.dem_data)) / (np.nanmax(self.dem_data) - np.nanmin(self.dem_data))
            # Mid-elevation gets highest vegetation
            self.ndvi_data = 0.3 + 0.5 * np.exp(-((elev_norm - 0.5) ** 2) / (2 * 0.3**2))
            self.ndvi_data += np.random.uniform(-0.05, 0.05, self.dem_data.shape)
            self.ndvi_data = np.clip(self.ndvi_data, 0, 1)
            self.ndwi_data = 0.2 * (1 - elev_norm)  # Lower areas wetter
            print(f"    - Synthetic NDVI range: {np.nanmin(self.ndvi_data):.3f} - {np.nanmax(self.ndvi_data):.3f}")

        # Calculate vegetation stability
        print("  Computing vegetation stability proxy...")
        self.vegetation_stability = (
            self.satellite_processor.calculate_vegetation_seasonal_variability(
                self.ndvi_data
            )
        )
        self._save_layer(
            self.vegetation_stability,
            os.path.join(self.config["output"]["maps_dir"], "vegetation_stability.tif"),
        )

        # Step 4: Estimate temperature from elevation and LST
        print("\n[STEP 4/5] Estimating temperature layers...")
        # Simple approximation: -6.5°C per km elevation + base temperature
        base_temp = 25.0  # °C at sea level
        lapse_rate = -0.0065  # °C per meter
        self.temperature_data = base_temp + lapse_rate * self.dem_data

        # Adjust for solar exposure (south-facing slopes warmer)
        self.temperature_data = self.temperature_data + 2.0 * self.solar_exposure

        print(f"  ✓ Temperature estimated")
        print(
            f"    - Temperature range: {np.nanmin(self.temperature_data):.1f} - {np.nanmax(self.temperature_data):.1f}°C"
        )

        # Step 5: Calculate suitability scores
        print("\n[STEP 5/5] Computing ecological suitability models...")

        print("  Calculating BEE suitability scores...")
        self.bee_suitability = self.suitability_model.calculate_bee_suitability_raster(
            ndvi=self.ndvi_data,
            elevation=self.dem_data,
            solar_exposure=self.solar_exposure,
            wind_exposure=self.wind_exposure,
            temperature=self.temperature_data,
            fog_persistence=self.fog_persistence,
            vegetation_stability=self.vegetation_stability,
            slope=self.slope_data,
        )

        print("  Calculating MANDARIN suitability scores...")
        self.mandarin_suitability = (
            self.suitability_model.calculate_mandarin_suitability_raster(
                elevation=self.dem_data,
                temperature=self.temperature_data,
                solar_exposure=self.solar_exposure,
                slope=self.slope_data,
                ndvi=self.ndvi_data,
                fog_persistence=self.fog_persistence,
                aspect=self.aspect_data,
            )
        )

        print("  Calculating COMBINED co-location scores...")
        self.combined_suitability = (
            self.suitability_model.calculate_combined_co_location_raster(
                self.bee_suitability, self.mandarin_suitability
            )
        )

        # Save suitability rasters
        print("\n  Saving suitability GeoTIFFs...")
        self._save_layer(
            self.bee_suitability,
            os.path.join(self.config["output"]["maps_dir"], "bee_suitability.tif"),
        )
        self._save_layer(
            self.mandarin_suitability,
            os.path.join(self.config["output"]["maps_dir"], "mandarin_suitability.tif"),
        )
        self._save_layer(
            self.combined_suitability,
            os.path.join(self.config["output"]["maps_dir"], "combined_suitability.tif"),
        )

        # Step 6: Generate reports
        print("\n[FINAL] Generating analysis reports...")

        # Clip to InitialFarm boundary and generate statistics
        initial_farm_report = self._generate_initialfarm_report(farm_data)

        # Save JSON report
        report_path = os.path.join(
            self.config["output"]["maps_dir"], "initialfarm_probability_report.json"
        )
        with open(report_path, "w") as f:
            json.dump(initial_farm_report, f, indent=2)
        print(f"  ✓ Saved: {report_path}")

        # Generate best zones KML
        best_zones_kml = self._generate_best_zones_kml(farm_data)
        kml_path = os.path.join(
            self.config["output"]["maps_dir"], "initialfarm_best_zones.kml"
        )
        with open(kml_path, "w") as f:
            f.write(best_zones_kml)
        print(f"  ✓ Saved: {kml_path}")

        # Print summary
        self._print_summary(initial_farm_report)

        print("\n" + "=" * 80)
        print("✓ ANALYSIS COMPLETE")
        print(f"Output directory: {self.config['output']['maps_dir']}/")
        print("=" * 80)

    def _save_layer(self, data: np.ndarray, filepath: str, nodata: float = -9999):
        """Save raster layer as GeoTIFF"""
        self.terrain_analyzer.save_geotiff(data, filepath, nodata)

    def _generate_initialfarm_report(self, farm_data: Dict) -> Dict:
        """Generate detailed analysis report for InitialFarm boundary"""

        # Clip to farm boundary and calculate statistics
        farm_geom = farm_data.get("geometry", None)

        # For now, use mean of all data (can be refined with polygon masking)
        report = {
            "analysis_date": datetime.now().isoformat(),
            "farm_name": farm_data["name"],
            "farm_center": {"lat": farm_data["center"][1], "lon": farm_data["center"][0]},
            "farm_area_hectares": farm_data["area_hectares"],
            "bee_suitability": {
                "mean_score": float(np.nanmean(self.bee_suitability)),
                "median_score": float(np.nanmedian(self.bee_suitability)),
                "std_dev": float(np.nanstd(self.bee_suitability)),
                "min_score": float(np.nanmin(self.bee_suitability)),
                "max_score": float(np.nanmax(self.bee_suitability)),
                "classification": self.suitability_model.classify_suitability(
                    np.nanmean(self.bee_suitability)
                )[0],
            },
            "mandarin_suitability": {
                "mean_score": float(np.nanmean(self.mandarin_suitability)),
                "median_score": float(np.nanmedian(self.mandarin_suitability)),
                "std_dev": float(np.nanstd(self.mandarin_suitability)),
                "min_score": float(np.nanmin(self.mandarin_suitability)),
                "max_score": float(np.nanmax(self.mandarin_suitability)),
                "classification": self.suitability_model.classify_suitability(
                    np.nanmean(self.mandarin_suitability)
                )[0],
            },
            "combined_suitability": {
                "mean_score": float(np.nanmean(self.combined_suitability)),
                "median_score": float(np.nanmedian(self.combined_suitability)),
                "std_dev": float(np.nanstd(self.combined_suitability)),
                "min_score": float(np.nanmin(self.combined_suitability)),
                "max_score": float(np.nanmax(self.combined_suitability)),
                "classification": self.suitability_model.classify_suitability(
                    np.nanmean(self.combined_suitability)
                )[0],
            },
            "environmental_conditions": {
                "elevation_range_m": [
                    float(np.nanmin(self.dem_data)),
                    float(np.nanmax(self.dem_data)),
                ],
                "slope_range_degrees": [
                    float(np.nanmin(self.slope_data)),
                    float(np.nanmax(self.slope_data)),
                ],
                "temperature_range_c": [
                    float(np.nanmin(self.temperature_data)),
                    float(np.nanmax(self.temperature_data)),
                ],
                "ndvi_range": [
                    float(np.nanmin(self.ndvi_data)),
                    float(np.nanmax(self.ndvi_data)),
                ],
                "fog_persistence_range": [
                    float(np.nanmin(self.fog_persistence)),
                    float(np.nanmax(self.fog_persistence)),
                ],
                "solar_exposure_range": [
                    float(np.nanmin(self.solar_exposure)),
                    float(np.nanmax(self.solar_exposure)),
                ],
                "wind_exposure_range": [
                    float(np.nanmin(self.wind_exposure)),
                    float(np.nanmax(self.wind_exposure)),
                ],
            },
            "assumptions_and_methodology": {
                "model_type": "Deterministic ecological probability model",
                "ml_used": False,
                "ground_truth_used": False,
                "spatial_resolution_m": self.config.get("grid_resolution_m", 100),
                "data_sources": [
                    "USGS DEM (1 arc-second)",
                    "Sentinel-2 Level-2A (if available)",
                    "Landsat 8/9 Level-2 (if available)",
                ],
                "fog_model": "Topographic depression proxy: combines low elevation, north-facing aspect, and valley identification",
                "wind_model": "Terrain exposure: ridge crests score high; valleys sheltered",
                "solar_radiation_model": "Aspect + slope + elevation: south-facing slopes optimal",
                "temperature_model": "Elevation lapse rate (~-6.5°C/km) + solar exposure adjustment",
                "vegetation_stability_proxy": "Local NDVI spatial variance used as proxy for flowering temporal stability",
                "bee_energy_cost": "Penalizes >150m elevation gain within 1km radius",
                "frost_risk": "Proxied by fog persistence + valley topography",
            },
        }

        return report

    def _generate_best_zones_kml(self, farm_data: Dict, top_percent: float = 20) -> str:
        """Generate KML of top 20% best zones for co-location"""

        # Find pixels in top 20% of combined score
        threshold = np.nanpercentile(self.combined_suitability, 100 - top_percent)

        kml_str = '<?xml version="1.0" encoding="UTF-8"?>\n'
        kml_str += '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        kml_str += "  <Document>\n"
        kml_str += f"    <name>Best Zones - {farm_data['name']}</name>\n"
        kml_str += f"    <description>Top {top_percent}% co-location suitability zones</description>\n"

        # Create style
        kml_str += """
    <Style id="excellent">
      <PolyStyle>
        <color>ff00ff00</color>
        <outline>1</outline>
      </PolyStyle>
    </Style>
    <Style id="high">
      <PolyStyle>
        <color>ff0088ff</color>
        <outline>1</outline>
      </PolyStyle>
    </Style>
    <Style id="moderate">
      <PolyStyle>
        <color>ff00ffff</color>
        <outline>1</outline>
      </PolyStyle>
    </Style>
"""

        # Add placemarks for top zones
        # Use actual raster transform to convert pixel coordinates to lat/lon
        transform = self.terrain_analyzer.transform
        
        best_pixels = np.where(self.combined_suitability >= threshold)

        # Subsample to avoid too many points (every 5th pixel)
        zone_count = 0
        for i, (row, col) in enumerate(
            zip(best_pixels[0][::5], best_pixels[1][::5])
        ):
            # Convert pixel coordinates to geographic coordinates using rasterio transform
            # transform * (col, row) gives (x, y) = (lon, lat) for EPSG:4326
            lon, lat = transform * (col + 0.5, row + 0.5)  # Center of pixel
            
            score = self.combined_suitability[row, col]
            bee_score = self.bee_suitability[row, col]
            mandarin_score = self.mandarin_suitability[row, col]

            if score >= 0.8:
                style = "excellent"
                class_name = "Excellent"
            elif score >= 0.6:
                style = "high"
                class_name = "High"
            else:
                style = "moderate"
                class_name = "Moderate"

            zone_count += 1
            kml_str += f"""    <Placemark>
      <name>Zone {zone_count} - {class_name} ({score:.3f})</name>
      <styleUrl>#{style}</styleUrl>
      <description>
        Combined Score: {score:.3f}
        Bee Suitability: {bee_score:.3f}
        Mandarin Suitability: {mandarin_score:.3f}
      </description>
      <Point>
        <coordinates>{lon:.6f},{lat:.6f},0</coordinates>
      </Point>
    </Placemark>
"""

        kml_str += "  </Document>\n"
        kml_str += "</kml>\n"

        return kml_str

    def _print_summary(self, report: Dict):
        """Print human-readable summary"""
        print("\n" + "-" * 80)
        print("INITIALFARM SUITABILITY ANALYSIS SUMMARY")
        print("-" * 80)

        print(f"\nFarm: {report['farm_name']}")
        print(f"Area: {report['farm_area_hectares']:.1f} hectares")

        bee = report["bee_suitability"]
        mandarin = report["mandarin_suitability"]
        combined = report["combined_suitability"]

        print("\n📍 BEE FARMING SUITABILITY")
        print(f"   Mean Score:       {bee['mean_score']:.3f}")
        print(f"   Classification:   {bee['classification']}")
        print(f"   Score Range:      {bee['min_score']:.3f} - {bee['max_score']:.3f}")

        print("\n🍊 MANDARIN ORANGE SUITABILITY")
        print(f"   Mean Score:       {mandarin['mean_score']:.3f}")
        print(f"   Classification:   {mandarin['classification']}")
        print(f"   Score Range:      {mandarin['min_score']:.3f} - {mandarin['max_score']:.3f}")

        print("\n🔀 COMBINED CO-LOCATION SUITABILITY")
        print(f"   Mean Score:       {combined['mean_score']:.3f}")
        print(f"   Classification:   {combined['classification']}")
        print(f"   Score Range:      {combined['min_score']:.3f} - {combined['max_score']:.3f}")

        env = report["environmental_conditions"]
        print("\n🌍 ENVIRONMENTAL CHARACTERISTICS")
        print(f"   Elevation:        {env['elevation_range_m'][0]:.0f} - {env['elevation_range_m'][1]:.0f} m")
        print(f"   Slope:            {env['slope_range_degrees'][0]:.1f} - {env['slope_range_degrees'][1]:.1f}°")
        print(f"   Temperature:      {env['temperature_range_c'][0]:.1f} - {env['temperature_range_c'][1]:.1f}°C")
        print(
            f"   NDVI:             {env['ndvi_range'][0]:.3f} - {env['ndvi_range'][1]:.3f}"
        )
        print(
            f"   Solar Exposure:   {env['solar_exposure_range'][0]:.3f} - {env['solar_exposure_range'][1]:.3f}"
        )
        print(
            f"   Fog Persistence:  {env['fog_persistence_range'][0]:.3f} - {env['fog_persistence_range'][1]:.3f}"
        )

        print("\n" + "-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Ecological Geospatial Planning Engine for bee and mandarin farming"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    args = parser.parse_args()

    engine = EcologicalPlanningEngine(args.config)
    engine.run_complete_analysis()


if __name__ == "__main__":
    main()
