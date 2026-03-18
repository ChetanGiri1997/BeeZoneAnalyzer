"""
Main Analysis Pipeline - Orchestrates the entire agricultural suitability analysis
"""

import numpy as np
import yaml
import json
import os
from tqdm import tqdm
from typing import Dict, List
import argparse

from kml_parser import KMLParser
from terrain_analyzer import TerrainAnalyzer
from satellite_processor import SatelliteProcessor
from suitability_model import SuitabilityModel
from kml_generator import KMLGenerator


class AgriculturalAnalyzer:
    def __init__(self, config_file: str = "config.yaml"):
        # Load configuration
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.kml_parser = KMLParser(self.config["input"]["kml_file"])
        self.terrain_analyzer = None
        self.satellite_processor = None
        self.model = SuitabilityModel(config_file)
        self.kml_generator = KMLGenerator()

        # Create output directory
        os.makedirs(self.config["output"]["maps_dir"], exist_ok=True)

    def run_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "=" * 70)
        print("AGRICULTURAL SUITABILITY ANALYSIS")
        print("=" * 70)

        # Step 1: Parse KML and get farm location
        print("\n[1/6] Parsing KML file...")
        farm_data = self.kml_parser.parse()
        search_bounds = self.kml_parser.get_search_bounds(
            farm_data["center"], self.config["search_radius_km"]
        )

        print(f"  Farm: {farm_data['name']}")
        print(
            f"  Center: {farm_data['center'][1]:.6f}°N, {farm_data['center'][0]:.6f}°E"
        )
        print(f"  Area: {farm_data['area_hectares']:.2f} hectares")
        print(f"  Search radius: {self.config['search_radius_km']} km")

        # Step 2: Load and process terrain data
        print("\n[2/6] Processing terrain data...")
        self.terrain_analyzer = TerrainAnalyzer(self.config["input"]["dem_file"])
        self.terrain_analyzer.load_dem(search_bounds)

        slope = self.terrain_analyzer.calculate_slope()
        aspect = self.terrain_analyzer.calculate_aspect()

        # Save terrain outputs
        self.terrain_analyzer.save_geotiff(
            self.terrain_analyzer.dem_data,
            os.path.join(self.config["output"]["maps_dir"], "elevation.tif"),
        )
        self.terrain_analyzer.save_geotiff(
            slope, os.path.join(self.config["output"]["maps_dir"], "slope.tif")
        )
        self.terrain_analyzer.save_geotiff(
            aspect, os.path.join(self.config["output"]["maps_dir"], "aspect.tif")
        )

        # Step 3: Process satellite imagery
        print("\n[3/6] Processing satellite imagery...")
        self.satellite_processor = SatelliteProcessor(
            self.config["input"]["landsat_dir"], self.config["input"]["sentinel_dir"]
        )

        satellite_data = self.satellite_processor.process_landsat(search_bounds)

        if satellite_data:
            # Save vegetation indices
            for name, data in satellite_data["indices"].items():
                self.satellite_processor.save_geotiff(
                    data, os.path.join(self.config["output"]["maps_dir"], f"{name}.tif")
                )

        # Step 4: Analyze initial farm location
        print("\n[4/6] Analyzing initial farm location...")
        initial_features = self._extract_features(
            farm_data["center"][0],
            farm_data["center"][1],
            slope,
            aspect,
            satellite_data,
        )

        initial_analysis = self.model.analyze_location(initial_features)

        print(f"\n  Initial Farm Suitability:")
        print(f"    Bee Farming: {initial_analysis['bee_farming']['probability']:.1f}%")
        print(
            f"    Mandarin Farming: {initial_analysis['mandarin_farming']['probability']:.1f}%"
        )

        # Step 5: Search for alternative locations
        print(f"\n[5/6] Searching for alternative locations (4km radius)...")
        bee_locations, mandarin_locations = self._search_locations(
            search_bounds, slope, aspect, satellite_data
        )

        print(f"  Found {len(bee_locations)} suitable bee farming locations")
        print(f"  Found {len(mandarin_locations)} suitable mandarin farming locations")

        # Step 6: Generate outputs
        print("\n[6/6] Generating outputs...")

        # Save results JSON
        results = {
            "initial_farm": {
                "name": farm_data["name"],
                "center": farm_data["center"],
                "area_hectares": farm_data["area_hectares"],
                "bee_farming": initial_analysis["bee_farming"],
                "mandarin_farming": initial_analysis["mandarin_farming"],
                "features": initial_features,
            },
            "bee_farming_locations": bee_locations[
                : self.config["analysis"]["top_n_locations"]
            ],
            "mandarin_farming_locations": mandarin_locations[
                : self.config["analysis"]["top_n_locations"]
            ],
            "search_parameters": {
                "radius_km": self.config["search_radius_km"],
                "grid_resolution_m": self.config["grid_resolution_m"],
            },
        }

        with open(self.config["output"]["results_json"], "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {self.config['output']['results_json']}")

        # Generate KML files
        self.kml_generator.create_initial_farm_kml(
            farm_data, initial_analysis, "initial_farm_analysis.kml"
        )

        self.kml_generator.create_location_kml(
            bee_locations[: self.config["analysis"]["top_n_locations"]],
            self.config["output"]["bee_farming_kml"],
            "Bee Farming",
            "ff00ff00",
        )

        self.kml_generator.create_location_kml(
            mandarin_locations[: self.config["analysis"]["top_n_locations"]],
            self.config["output"]["mandarin_farming_kml"],
            "Mandarin Orange Farming",
            "ff0080ff",
        )

        self.kml_generator.create_comparison_kml(
            bee_locations[:10],
            mandarin_locations[:10],
            self.config["output"]["recommended_kml"],
        )

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE!")
        print("=" * 70)
        print(f"\nOutput files:")
        print(f"  - {self.config['output']['results_json']}")
        print(f"  - {self.config['output']['bee_farming_kml']}")
        print(f"  - {self.config['output']['mandarin_farming_kml']}")
        print(f"  - {self.config['output']['recommended_kml']}")
        print(f"  - initial_farm_analysis.kml")
        print(f"  - {self.config['output']['maps_dir']}/ (GeoTIFF files)")

        return results

    def _extract_features(
        self,
        lon: float,
        lat: float,
        slope: np.ndarray,
        aspect: np.ndarray,
        satellite_data: Dict,
    ) -> Dict:
        """Extract all features for a location"""
        features = {}

        # Terrain features
        terrain_features = self.terrain_analyzer.get_terrain_features(lon, lat)
        if terrain_features:
            features.update(terrain_features)

        # Satellite features
        if satellite_data:
            for name, data in satellite_data["indices"].items():
                value = self.satellite_processor.get_pixel_value(data, lon, lat)
                features[name] = value

        # Default temperature (could be enhanced with actual climate data)
        features["temperature"] = 22  # Placeholder

        return features

    def _search_locations(
        self, bounds: Dict, slope: np.ndarray, aspect: np.ndarray, satellite_data: Dict
    ) -> tuple:
        """Search for suitable locations in the area"""
        # Create analysis grid
        lon_grid, lat_grid = self.terrain_analyzer.create_analysis_grid(
            bounds, self.config["grid_resolution_m"]
        )

        bee_locations = []
        mandarin_locations = []

        total_points = lon_grid.size
        min_score = self.config["analysis"]["min_suitability_score"]

        print(f"  Analyzing {total_points} grid points...")

        # Flatten grids for iteration
        lons = lon_grid.flatten()
        lats = lat_grid.flatten()

        for i in tqdm(range(len(lons)), desc="  Progress"):
            lon, lat = lons[i], lats[i]

            # Extract features
            features = self._extract_features(lon, lat, slope, aspect, satellite_data)

            # Skip if missing critical features
            if "elevation" not in features or np.isnan(
                features.get("elevation", np.nan)
            ):
                continue

            # Analyze location
            analysis = self.model.analyze_location(features)

            # Check bee farming suitability
            if analysis["bee_farming"]["total_score"] >= min_score:
                bee_locations.append(
                    {
                        "lon": lon,
                        "lat": lat,
                        "score": analysis["bee_farming"]["total_score"],
                        "probability": analysis["bee_farming"]["probability"],
                        "component_scores": analysis["bee_farming"]["component_scores"],
                        **features,
                    }
                )

            # Check mandarin farming suitability
            if analysis["mandarin_farming"]["total_score"] >= min_score:
                mandarin_locations.append(
                    {
                        "lon": lon,
                        "lat": lat,
                        "score": analysis["mandarin_farming"]["total_score"],
                        "probability": analysis["mandarin_farming"]["probability"],
                        "component_scores": analysis["mandarin_farming"][
                            "component_scores"
                        ],
                        **features,
                    }
                )

        # Sort by score (descending)
        bee_locations.sort(key=lambda x: x["score"], reverse=True)
        mandarin_locations.sort(key=lambda x: x["score"], reverse=True)

        return bee_locations, mandarin_locations


def main():
    parser = argparse.ArgumentParser(description="Agricultural Suitability Analysis")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    args = parser.parse_args()

    analyzer = AgriculturalAnalyzer(args.config)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
