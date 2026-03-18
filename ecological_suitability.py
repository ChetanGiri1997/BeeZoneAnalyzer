"""
Ecological Suitability Model - Deterministic spatial analysis for bee and mandarin farming

PART 2: Bee Thriving Model
BeeSuitabilityScore =
    0.20 * NDVI +
    0.15 * Elevation_optimality +
    0.15 * SolarExposure +
    0.15 * WindExposure_inverse +
    0.10 * Temperature_optimality +
    0.10 * Fog_inverse +
    0.10 * Vegetation_stability +
    0.05 * Slope_optimality

PART 3: Mandarin Orange Model
MandarinSuitabilityScore =
    0.25 * Elevation_optimality +
    0.20 * Temperature_optimality +
    0.15 * SolarExposure +
    0.15 * Slope_optimality +
    0.10 * NDVI +
    0.10 * Fog_inverse +
    0.05 * FrostRisk_inverse

PART 4: Combined Co-Location Model
CombinedScore =
    (BeeSuitabilityScore * 0.5) +
    (MandarinSuitabilityScore * 0.5)
"""

import numpy as np
from typing import Dict, Tuple
import yaml


class EcologicalSuitabilityModel:
    def __init__(self):
        """Initialize with hardcoded optimal ranges for both species"""
        # Bee farming optimal ranges
        self.bee_params = {
            "elevation_min": 500,
            "elevation_max": 2000,
            "elevation_optimal": (1000, 1800),
            "slope_max": 15,
            "slope_optimal": 10,
            "temperature_min": 15,
            "temperature_max": 30,
            "temperature_optimal": (20, 28),
            "ndvi_threshold": 0.3,
            "ndvi_optimal": 0.5,  # Moderate vegetation for diverse flowers
            "vegetation_proximity_km": 1.5,
            "wind_exposure_max": 0.6,  # Avoid exposed ridge crests
            "elevation_gain_1km_max": 150,  # meters, energy cost constraint
        }

        # Mandarin orange optimal ranges
        self.mandarin_params = {
            "elevation_min": 800,
            "elevation_max": 1500,
            "elevation_optimal": (1000, 1500),
            "slope_min": 5,
            "slope_max": 20,
            "slope_optimal": (8, 15),
            "temperature_min": 15,
            "temperature_max": 25,
            "temperature_optimal": (18, 25),
            "ndvi_optimal": 0.5,  # Moderate vegetation
            "aspect_preferred": ["S", "SE", "SW"],  # South-facing for winter sun
            "frost_risk_threshold": 0.3,  # Low frost risk
        }

    def score_elevation_optimality(
        self, elevation: float, opt_min: float, opt_max: float, hard_min: float, hard_max: float
    ) -> float:
        """
        Score elevation suitability using triangular distribution.
        Returns 0-1 score.
        """
        if np.isnan(elevation):
            return 0.0

        if elevation < hard_min or elevation > hard_max:
            return 0.0
        elif opt_min <= elevation <= opt_max:
            return 1.0
        elif elevation < opt_min:
            # Linear increase from hard_min to opt_min
            return (elevation - hard_min) / (opt_min - hard_min)
        else:
            # Linear decrease from opt_max to hard_max
            return (hard_max - elevation) / (hard_max - opt_max)

    def score_temperature_optimality(
        self, temperature: float, opt_min: float, opt_max: float, hard_min: float, hard_max: float
    ) -> float:
        """Score temperature suitability using triangular distribution."""
        if np.isnan(temperature):
            return 0.5  # Neutral if no data

        if temperature < hard_min or temperature > hard_max:
            return 0.0
        elif opt_min <= temperature <= opt_max:
            return 1.0
        elif temperature < opt_min:
            return (temperature - hard_min) / (opt_min - hard_min)
        else:
            return (hard_max - temperature) / (hard_max - opt_max)

    def score_slope_optimality(
        self,
        slope: float,
        opt_min: float,
        opt_max: float,
        hard_min: float,
        hard_max: float,
    ) -> float:
        """Score slope suitability for terrain."""
        if np.isnan(slope):
            return 0.0

        if slope < hard_min or slope > hard_max:
            return 0.0
        elif opt_min <= slope <= opt_max:
            return 1.0
        elif slope < opt_min:
            return (slope - hard_min) / (opt_min - hard_min)
        else:
            return (hard_max - slope) / (hard_max - opt_max)

    def score_aspect_for_mandarin(self, aspect: float) -> float:
        """
        Score aspect for mandarin (prefers south-facing for winter sun).
        Aspect in degrees: 0/360=N, 90=E, 180=S, 270=W
        """
        if np.isnan(aspect):
            return 0.5

        # South-facing: 135-225° (peak at 180°)
        # Score as Gaussian centered at 180°
        south_center = 180
        score = np.exp(-((aspect - south_center) ** 2) / (2 * 45**2))  # 45° std dev

        return score

    def score_frost_risk_inverse(self, fog_persistence: float, elevation: float) -> float:
        """
        Frost risk is high in:
        - Valley cold sinks (low elevation with fog)
        - North-facing slopes with high fog

        We invert fog_persistence to get frost_risk_inverse for scoring.
        Areas with LOW fog have LOW frost risk = HIGH score for mandarins.
        """
        # Areas with high fog persistence have high frost risk
        frost_risk = fog_persistence

        # Return inverse: high fog = low score, low fog = high score
        frost_risk_inverse = 1.0 - frost_risk

        return frost_risk_inverse

    def calculate_bee_suitability_raster(
        self,
        ndvi: np.ndarray,
        elevation: np.ndarray,
        solar_exposure: np.ndarray,
        wind_exposure: np.ndarray,
        temperature: np.ndarray,
        fog_persistence: np.ndarray,
        vegetation_stability: np.ndarray,
        slope: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate bee suitability score for entire raster.

        BeeSuitabilityScore =
            0.20 * NDVI +
            0.15 * Elevation_optimality +
            0.15 * SolarExposure +
            0.15 * WindExposure_inverse +
            0.10 * Temperature_optimality +
            0.10 * Fog_inverse +
            0.10 * Vegetation_stability +
            0.05 * Slope_optimality
        """

        # Component 1: NDVI (0.20 weight)
        # Normalize NDVI (typically -1 to 1) to 0-1, with threshold at 0.3
        ndvi_score = np.clip((ndvi - 0.3) / (0.7), 0, 1)

        # Component 2: Elevation optimality (0.15 weight)
        elev_opt = np.vectorize(
            lambda e: self.score_elevation_optimality(
                e,
                self.bee_params["elevation_optimal"][0],
                self.bee_params["elevation_optimal"][1],
                self.bee_params["elevation_min"],
                self.bee_params["elevation_max"],
            )
        )(elevation)

        # Component 3: Solar exposure (0.15 weight) - already normalized 0-1
        solar_score = solar_exposure

        # Component 4: Wind exposure inverse (0.15 weight)
        # Bees prefer LOW wind, so invert wind exposure (1 - wind = good)
        wind_inverse = 1.0 - wind_exposure

        # Component 5: Temperature optimality (0.10 weight)
        temp_opt = np.vectorize(
            lambda t: self.score_temperature_optimality(
                t,
                self.bee_params["temperature_optimal"][0],
                self.bee_params["temperature_optimal"][1],
                self.bee_params["temperature_min"],
                self.bee_params["temperature_max"],
            )
        )(temperature)

        # Component 6: Fog inverse (0.10 weight)
        # Bees thrive in LOW fog areas
        fog_inverse = 1.0 - fog_persistence

        # Component 7: Vegetation stability (0.10 weight)
        # Use vegetation_stability directly (seasonal NDVI variability)
        veg_stability_score = vegetation_stability

        # Component 8: Slope optimality (0.05 weight)
        slope_opt = np.vectorize(
            lambda s: self.score_slope_optimality(s, 0, self.bee_params["slope_optimal"], 0, self.bee_params["slope_max"])
        )(slope)

        # Combine with weights
        bee_score = (
            0.20 * ndvi_score
            + 0.15 * elev_opt
            + 0.15 * solar_score
            + 0.15 * wind_inverse
            + 0.10 * temp_opt
            + 0.10 * fog_inverse
            + 0.10 * veg_stability_score
            + 0.05 * slope_opt
        )

        # Normalize to 0-1
        bee_score = np.clip(bee_score, 0, 1)

        return bee_score

    def calculate_mandarin_suitability_raster(
        self,
        elevation: np.ndarray,
        temperature: np.ndarray,
        solar_exposure: np.ndarray,
        slope: np.ndarray,
        ndvi: np.ndarray,
        fog_persistence: np.ndarray,
        aspect: np.ndarray = None,
    ) -> np.ndarray:
        """
        Calculate mandarin orange suitability score for entire raster.

        MandarinSuitabilityScore =
            0.25 * Elevation_optimality +
            0.20 * Temperature_optimality +
            0.15 * SolarExposure +
            0.15 * Slope_optimality +
            0.10 * NDVI +
            0.10 * Fog_inverse +
            0.05 * FrostRisk_inverse
        """

        # Component 1: Elevation optimality (0.25 weight)
        elev_opt = np.vectorize(
            lambda e: self.score_elevation_optimality(
                e,
                self.mandarin_params["elevation_optimal"][0],
                self.mandarin_params["elevation_optimal"][1],
                self.mandarin_params["elevation_min"],
                self.mandarin_params["elevation_max"],
            )
        )(elevation)

        # Component 2: Temperature optimality (0.20 weight)
        temp_opt = np.vectorize(
            lambda t: self.score_temperature_optimality(
                t,
                self.mandarin_params["temperature_optimal"][0],
                self.mandarin_params["temperature_optimal"][1],
                self.mandarin_params["temperature_min"],
                self.mandarin_params["temperature_max"],
            )
        )(temperature)

        # Component 3: Solar exposure (0.15 weight)
        solar_score = solar_exposure

        # Component 4: Slope optimality (0.15 weight)
        slope_opt = np.vectorize(
            lambda s: self.score_slope_optimality(
                s,
                self.mandarin_params["slope_optimal"][0],
                self.mandarin_params["slope_optimal"][1],
                self.mandarin_params["slope_min"],
                self.mandarin_params["slope_max"],
            )
        )(slope)

        # Component 5: NDVI (0.10 weight) - moderate vegetation desired
        ndvi_score = np.clip((ndvi - 0.3) / (0.7), 0, 1)

        # Component 6: Fog inverse (0.10 weight)
        # Mandarins struggle in persistent fog zones
        fog_inverse = 1.0 - fog_persistence

        # Component 7: Frost risk inverse (0.05 weight)
        # Frost risk is high in valleys with fog
        frost_risk_inverse = self.score_frost_risk_inverse(fog_persistence, elevation)

        # Combine with weights
        mandarin_score = (
            0.25 * elev_opt
            + 0.20 * temp_opt
            + 0.15 * solar_score
            + 0.15 * slope_opt
            + 0.10 * ndvi_score
            + 0.10 * fog_inverse
            + 0.05 * frost_risk_inverse
        )

        # Normalize to 0-1
        mandarin_score = np.clip(mandarin_score, 0, 1)

        return mandarin_score

    def calculate_combined_co_location_raster(
        self, bee_score: np.ndarray, mandarin_score: np.ndarray
    ) -> np.ndarray:
        """
        Calculate combined optimal co-location score.

        CombinedScore =
            (BeeSuitabilityScore * 0.5) +
            (MandarinSuitabilityScore * 0.5)

        This identifies zones where:
        - Bees thrive naturally
        - Mandarin oranges grow optimally
        - Sunlight is strong year-round
        - Fog is minimal
        - Energy cost to bees is low
        """
        combined_score = 0.5 * bee_score + 0.5 * mandarin_score
        combined_score = np.clip(combined_score, 0, 1)

        return combined_score

    def classify_suitability(self, score: float) -> Tuple[str, float]:
        """
        Classify suitability based on score.

        0.8–1.0 → Excellent
        0.6–0.8 → High
        0.4–0.6 → Moderate
        <0.4 → Low
        """
        if score >= 0.8:
            return "Excellent", score
        elif score >= 0.6:
            return "High", score
        elif score >= 0.4:
            return "Moderate", score
        else:
            return "Low", score


if __name__ == "__main__":
    print("Ecological Suitability Model loaded successfully.")
    print("This module provides deterministic scoring for bee and mandarin farming.")
