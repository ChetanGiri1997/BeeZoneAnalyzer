"""
Suitability Model - ML model for agricultural suitability analysis
"""

import numpy as np
from typing import Dict, List, Tuple
import yaml


class SuitabilityModel:
    def __init__(self, config_file: str = "config.yaml"):
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.bee_criteria = self.config["bee_farming"]
        self.mandarin_criteria = self.config["mandarin_farming"]

    def score_elevation(self, elevation: float, criteria: Dict) -> float:
        """Score elevation suitability (0-1)"""
        if np.isnan(elevation):
            return 0.0

        min_elev = criteria["elevation"]["min"]
        max_elev = criteria["elevation"]["max"]
        opt_min = criteria["elevation"]["optimal_min"]
        opt_max = criteria["elevation"]["optimal_max"]

        if elevation < min_elev or elevation > max_elev:
            return 0.0
        elif opt_min <= elevation <= opt_max:
            return 1.0
        elif elevation < opt_min:
            # Linear interpolation between min and optimal_min
            return (elevation - min_elev) / (opt_min - min_elev)
        else:
            # Linear interpolation between optimal_max and max
            return (max_elev - elevation) / (max_elev - opt_max)

    def score_temperature(self, temperature: float, criteria: Dict) -> float:
        """Score temperature suitability (0-1)"""
        if np.isnan(temperature):
            return 0.5  # Neutral if no data

        min_temp = criteria["temperature"]["min"]
        max_temp = criteria["temperature"]["max"]
        opt_min = criteria["temperature"]["optimal_min"]
        opt_max = criteria["temperature"]["optimal_max"]

        if temperature < min_temp or temperature > max_temp:
            return 0.0
        elif opt_min <= temperature <= opt_max:
            return 1.0
        elif temperature < opt_min:
            return (temperature - min_temp) / (opt_min - min_temp)
        else:
            return (max_temp - temperature) / (max_temp - opt_max)

    def score_slope(self, slope: float, criteria: Dict) -> float:
        """Score slope suitability (0-1)"""
        if np.isnan(slope):
            return 0.0

        if "min" in criteria["slope"]:
            # Mandarin: needs some slope for drainage
            min_slope = criteria["slope"]["min"]
            max_slope = criteria["slope"]["max"]
            opt_min = criteria["slope"]["optimal_min"]
            opt_max = criteria["slope"]["optimal_max"]

            if slope < min_slope or slope > max_slope:
                return 0.0
            elif opt_min <= slope <= opt_max:
                return 1.0
            elif slope < opt_min:
                return (slope - min_slope) / (opt_min - min_slope)
            else:
                return (max_slope - slope) / (max_slope - opt_max)
        else:
            # Bee farming: prefer flatter terrain
            max_slope = criteria["slope"]["max"]
            opt_max = criteria["slope"]["optimal_max"]

            if slope > max_slope:
                return 0.0
            elif slope <= opt_max:
                return 1.0
            else:
                return (max_slope - slope) / (max_slope - opt_max)

    def score_aspect(self, aspect: float, criteria: Dict) -> float:
        """Score aspect suitability (0-1) - mainly for mandarin"""
        if "aspect" not in criteria:
            return 1.0  # Not applicable for bee farming

        if np.isnan(aspect):
            return 0.5

        preferred = criteria["aspect"]["preferred"]

        # Define aspect ranges for each direction
        aspect_ranges = {
            "N": (337.5, 22.5),
            "NE": (22.5, 67.5),
            "E": (67.5, 112.5),
            "SE": (112.5, 157.5),
            "S": (157.5, 202.5),
            "SW": (202.5, 247.5),
            "W": (247.5, 292.5),
            "NW": (292.5, 337.5),
        }

        # Check if aspect falls in preferred directions
        for direction in preferred:
            if direction in aspect_ranges:
                min_a, max_a = aspect_ranges[direction]
                if direction == "N":  # Special case for North (wraps around)
                    if aspect >= min_a or aspect < max_a:
                        return 1.0
                else:
                    if min_a <= aspect < max_a:
                        return 1.0

        # Partial score for adjacent directions
        return 0.5

    def score_ndvi(self, ndvi: float, criteria: Dict) -> float:
        """Score vegetation health (0-1)"""
        if "vegetation_ndvi" not in criteria:
            return 1.0

        if np.isnan(ndvi):
            return 0.5

        min_ndvi = criteria["vegetation_ndvi"]["min"]
        opt_min = criteria["vegetation_ndvi"]["optimal_min"]

        if ndvi < min_ndvi:
            return 0.0
        elif ndvi >= opt_min:
            return 1.0
        else:
            return (ndvi - min_ndvi) / (opt_min - min_ndvi)

    def score_soil_moisture(self, ndwi: float, criteria: Dict) -> float:
        """Score soil moisture from NDWI (0-1)"""
        if "soil_moisture" not in criteria:
            return 1.0

        if np.isnan(ndwi):
            return 0.5

        # Convert NDWI (-1 to 1) to approximate soil moisture (0 to 1)
        moisture = (ndwi + 1) / 2

        min_moist = criteria["soil_moisture"]["min"]
        max_moist = criteria["soil_moisture"]["max"]
        opt_min = criteria["soil_moisture"]["optimal_min"]
        opt_max = criteria["soil_moisture"]["optimal_max"]

        if moisture < min_moist or moisture > max_moist:
            return 0.0
        elif opt_min <= moisture <= opt_max:
            return 1.0
        elif moisture < opt_min:
            return (moisture - min_moist) / (opt_min - min_moist)
        else:
            return (max_moist - moisture) / (max_moist - opt_max)

    def calculate_bee_suitability(self, features: Dict) -> Dict:
        """Calculate bee farming suitability score"""
        scores = {
            "elevation": self.score_elevation(
                features.get("elevation", np.nan), self.bee_criteria
            ),
            "temperature": self.score_temperature(
                features.get("temperature", 20), self.bee_criteria
            ),
            "slope": self.score_slope(features.get("slope", np.nan), self.bee_criteria),
            "ndvi": self.score_ndvi(features.get("ndvi", np.nan), self.bee_criteria),
        }

        # Weighted average
        weights = {
            "elevation": self.bee_criteria["elevation"]["weight"],
            "temperature": self.bee_criteria["temperature"]["weight"],
            "slope": self.bee_criteria["slope"]["weight"],
            "ndvi": self.bee_criteria["vegetation_ndvi"]["weight"],
        }

        total_score = sum(scores[k] * weights[k] for k in scores.keys())

        return {
            "total_score": total_score,
            "probability": total_score * 100,  # Convert to percentage
            "component_scores": scores,
            "weights": weights,
        }

    def calculate_mandarin_suitability(self, features: Dict) -> Dict:
        """Calculate mandarin orange farming suitability score"""
        scores = {
            "elevation": self.score_elevation(
                features.get("elevation", np.nan), self.mandarin_criteria
            ),
            "temperature": self.score_temperature(
                features.get("temperature", 20), self.mandarin_criteria
            ),
            "slope": self.score_slope(
                features.get("slope", np.nan), self.mandarin_criteria
            ),
            "aspect": self.score_aspect(
                features.get("aspect", np.nan), self.mandarin_criteria
            ),
            "soil_moisture": self.score_soil_moisture(
                features.get("ndwi", np.nan), self.mandarin_criteria
            ),
        }

        # Weighted average
        weights = {
            "elevation": self.mandarin_criteria["elevation"]["weight"],
            "temperature": self.mandarin_criteria["temperature"]["weight"],
            "slope": self.mandarin_criteria["slope"]["weight"],
            "aspect": self.mandarin_criteria["aspect"]["weight"],
            "soil_moisture": self.mandarin_criteria["soil_moisture"]["weight"],
        }

        total_score = sum(scores[k] * weights[k] for k in scores.keys())

        return {
            "total_score": total_score,
            "probability": total_score * 100,  # Convert to percentage
            "component_scores": scores,
            "weights": weights,
        }

    def analyze_location(self, features: Dict) -> Dict:
        """Analyze a location for both farming types"""
        bee_result = self.calculate_bee_suitability(features)
        mandarin_result = self.calculate_mandarin_suitability(features)

        return {
            "bee_farming": bee_result,
            "mandarin_farming": mandarin_result,
            "features": features,
        }


if __name__ == "__main__":
    # Test the model
    model = SuitabilityModel()

    # Example features
    test_features = {
        "elevation": 1200,
        "slope": 10,
        "aspect": 150,  # SE
        "temperature": 22,
        "ndvi": 0.6,
        "ndwi": 0.2,
    }

    result = model.analyze_location(test_features)

    print("=" * 60)
    print("SUITABILITY ANALYSIS TEST")
    print("=" * 60)
    print(f"\nBee Farming Probability: {result['bee_farming']['probability']:.1f}%")
    print("Component scores:")
    for k, v in result["bee_farming"]["component_scores"].items():
        print(f"  {k}: {v:.2f}")

    print(
        f"\nMandarin Farming Probability: {result['mandarin_farming']['probability']:.1f}%"
    )
    print("Component scores:")
    for k, v in result["mandarin_farming"]["component_scores"].items():
        print(f"  {k}: {v:.2f}")
