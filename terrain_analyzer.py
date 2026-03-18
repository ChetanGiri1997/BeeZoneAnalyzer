"""
Terrain Analyzer - Process DEM data to extract elevation, slope, aspect,
and advanced derived layers for ecological suitability assessment
"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.ndimage import generic_filter, maximum_filter, minimum_filter
from scipy import ndimage
from typing import Dict, Tuple
import os


class TerrainAnalyzer:
    def __init__(self, dem_file: str):
        self.dem_file = dem_file
        self.dem_data = None
        self.transform = None
        self.crs = None

    def load_dem(self, bounds: Dict = None):
        """Load DEM data, optionally cropped to bounds"""
        with rasterio.open(self.dem_file) as src:
            self.crs = src.crs

            if bounds:
                # Create window from bounds
                from rasterio.windows import from_bounds

                window = from_bounds(
                    bounds["min_lon"],
                    bounds["min_lat"],
                    bounds["max_lon"],
                    bounds["max_lat"],
                    src.transform,
                )
                self.dem_data = src.read(1, window=window)
                self.transform = src.window_transform(window)
            else:
                self.dem_data = src.read(1)
                self.transform = src.transform

        print(
            f"Loaded DEM: {self.dem_data.shape}, min: {np.nanmin(self.dem_data):.1f}m, max: {np.nanmax(self.dem_data):.1f}m"
        )
        return self.dem_data

    def calculate_slope(self) -> np.ndarray:
        """Calculate slope in degrees"""
        if self.dem_data is None:
            raise ValueError("DEM data not loaded. Call load_dem() first.")

        # Get pixel size in meters
        pixel_size_x = abs(self.transform[0]) * 111000  # Approximate meters
        pixel_size_y = abs(self.transform[4]) * 111000

        # Calculate gradients
        dy, dx = np.gradient(self.dem_data, pixel_size_y, pixel_size_x)

        # Calculate slope in degrees
        slope = np.arctan(np.sqrt(dx**2 + dy**2)) * 180 / np.pi

        print(f"Slope range: {np.nanmin(slope):.1f}° to {np.nanmax(slope):.1f}°")
        return slope

    def calculate_aspect(self) -> np.ndarray:
        """Calculate aspect (direction of slope) in degrees from North"""
        if self.dem_data is None:
            raise ValueError("DEM data not loaded. Call load_dem() first.")

        # Get pixel size in meters
        pixel_size_x = abs(self.transform[0]) * 111000
        pixel_size_y = abs(self.transform[4]) * 111000

        # Calculate gradients
        dy, dx = np.gradient(self.dem_data, pixel_size_y, pixel_size_x)

        # Calculate aspect in degrees (0 = North, 90 = East, 180 = South, 270 = West)
        aspect = np.arctan2(-dx, dy) * 180 / np.pi
        aspect = (aspect + 360) % 360  # Normalize to 0-360

        return aspect

    def calculate_tri(self, window_size: int = 3) -> np.ndarray:
        """Calculate Terrain Ruggedness Index"""
        if self.dem_data is None:
            raise ValueError("DEM data not loaded. Call load_dem() first.")

        def tri_func(values):
            center = values[len(values) // 2]
            return np.sqrt(np.sum((values - center) ** 2))

        tri = generic_filter(
            self.dem_data, tri_func, size=window_size, mode="constant", cval=np.nan
        )
        return tri

    def calculate_terrain_roughness(self, window_size: int = 5) -> np.ndarray:
        """
        Calculate terrain roughness using standard deviation of elevation
        within a local window. Higher values = rougher terrain.
        """
        if self.dem_data is None:
            raise ValueError("DEM data not loaded. Call load_dem() first.")

        def roughness_func(values):
            return np.nanstd(values)

        roughness = generic_filter(
            self.dem_data,
            roughness_func,
            size=window_size,
            mode="constant",
            cval=np.nan,
        )
        # Normalize 0-1
        roughness_min = np.nanmin(roughness)
        roughness_max = np.nanmax(roughness)
        if roughness_max > roughness_min:
            roughness = (roughness - roughness_min) / (roughness_max - roughness_min)
        else:
            roughness = np.zeros_like(roughness)

        print(
            f"Terrain roughness calculated: min={np.nanmin(roughness):.3f}, max={np.nanmax(roughness):.3f}"
        )
        return roughness

    def calculate_relative_topographic_position(
        self, window_size: int = 11
    ) -> np.ndarray:
        """
        Calculate Relative Topographic Position (RTP) using deviation from
        local mean. Values: -1 (valleys) to +1 (ridges).
        Useful for identifying valley cold sinks vs exposed ridge crests.
        """
        if self.dem_data is None:
            raise ValueError("DEM data not loaded. Call load_dem() first.")

        from scipy.ndimage import uniform_filter

        # Calculate local mean elevation
        local_mean = uniform_filter(self.dem_data, size=window_size, mode="nearest")

        # Calculate RTP: (elevation - local_mean) / (std deviation within window)
        def std_func(values):
            return np.nanstd(values)

        local_std = generic_filter(
            self.dem_data,
            std_func,
            size=window_size,
            mode="constant",
            cval=np.nan,
        )

        # Avoid division by zero
        local_std = np.where(local_std < 0.1, 0.1, local_std)

        rtp = (self.dem_data - local_mean) / local_std

        # Normalize to -1 to 1 range
        rtp_min = np.nanpercentile(rtp, 1)
        rtp_max = np.nanpercentile(rtp, 99)
        rtp = 2 * ((rtp - rtp_min) / (rtp_max - rtp_min)) - 1
        rtp = np.clip(rtp, -1, 1)

        print(f"RTP calculated: min={np.nanmin(rtp):.3f}, max={np.nanmax(rtp):.3f}")
        return rtp

    def calculate_fog_persistence_proxy(self) -> np.ndarray:
        """
        Calculate fog persistence proxy using elevation and topographic depression.
        Fog persists in:
        - Low elevation valleys (cold sinks)
        - North-facing slopes (less solar radiation)
        - Areas with high terrain roughness around them

        Output: 0 = low fog, 1 = high fog persistence
        """
        if self.dem_data is None:
            raise ValueError("DEM data not loaded. Call load_dem() first.")

        # Get RTP (negative values = valleys where fog pools)
        rtp = self.calculate_relative_topographic_position()

        # Get aspect (north-facing = higher values for fog)
        aspect = self.calculate_aspect()

        # Aspect: North-facing (0° or 360°) gets high fog score
        # Score: 1.0 for N (0°), decreasing to 0 for S (180°)
        aspect_fog = np.abs(180 - aspect) / 180  # 0 at 180° (S), 1 at 0° (N)

        # Elevation component: lower elevation = higher fog
        # Normalize elevation to 0-1 (higher elevation = lower fog score)
        elev_min = np.nanmin(self.dem_data)
        elev_max = np.nanmax(self.dem_data)
        elev_norm = (elev_max - self.dem_data) / (elev_max - elev_min)

        # Combine: valleys (RTP < 0) + north-facing + low elevation
        fog_proxy = (
            0.4 * np.maximum(0, -rtp)  # Negative RTP (valleys) enhance fog
            + 0.3 * aspect_fog  # North-facing enhances fog
            + 0.3 * elev_norm  # Low elevation enhances fog
        )

        fog_proxy = np.clip(fog_proxy, 0, 1)

        print(
            f"Fog persistence proxy: min={np.nanmin(fog_proxy):.3f}, max={np.nanmax(fog_proxy):.3f}"
        )
        return fog_proxy

    def calculate_solar_exposure_index(self) -> np.ndarray:
        """
        Calculate solar exposure using slope and aspect.
        South-facing slopes (in northern hemisphere) receive maximum sun.
        Steep slopes perpendicular to sun receive less radiation.
        High elevation open terrain gets maximum solar radiation.

        Output: 0 = low solar exposure, 1 = high solar exposure
        """
        if self.dem_data is None:
            raise ValueError("DEM data not loaded. Call load_dem() first.")

        slope_rad = self.calculate_slope() * np.pi / 180
        aspect_rad = self.calculate_aspect() * np.pi / 180

        # Solar altitude at noon during spring equinox for ~28°N latitude
        # Use latitude from bounds (approximation)
        solar_altitude = 45 * np.pi / 180  # 45° at equinox for 28°N

        # Hillshade-like calculation: how much sun hits each pixel
        # Simplified: higher on south-facing, steeper slopes get intermediate exposure
        # North-facing gets low exposure

        # Solar radiation intensity depends on:
        # 1. Aspect (south = good, north = bad)
        # 2. Slope angle (perpendicular to sun = optimal)

        # Aspect component: South-facing (180°) gets score 1.0
        # Normalize aspect to 0-1 (180° = 1.0, 0° or 360° = 0)
        aspect_deg = self.calculate_aspect()
        aspect_score = np.abs(180 - aspect_deg) / 180  # 0 at 0°/360°, 1 at 180°

        # Slope component: moderate slopes are best
        # Very steep or very flat get lower scores
        slope_deg = self.calculate_slope()
        slope_rad_safe = np.radians(slope_deg)

        # Optimal slope for solar exposure: ~30-45°
        # Score based on how well aligned with solar radiation
        optimal_slope = 35
        slope_score = np.exp(-((slope_deg - optimal_slope) ** 2) / (2 * 30**2))

        # Elevation component: higher elevation = thinner atmosphere, more radiation
        elev_min = np.nanmin(self.dem_data)
        elev_max = np.nanmax(self.dem_data)
        elev_norm = (self.dem_data - elev_min) / (elev_max - elev_min)

        # Combine factors
        solar_index = 0.4 * aspect_score + 0.35 * slope_score + 0.25 * elev_norm

        solar_index = np.clip(solar_index, 0, 1)

        print(
            f"Solar exposure index: min={np.nanmin(solar_index):.3f}, max={np.nanmax(solar_index):.3f}"
        )
        return solar_index

    def calculate_elevation_variation_1km(self) -> np.ndarray:
        """
        Calculate elevation variation within 1 km radius.
        This estimates the energy cost for bees to forage and return.
        High variation = steep climbs required = higher energy cost.

        Used for "Energy Cost Constraint": Penalize steep vertical gain zones.
        """
        if self.dem_data is None:
            raise ValueError("DEM data not loaded. Call load_dem() first.")

        # Convert 1 km to pixels (assuming ~30m resolution for DEM)
        # More precisely: use actual pixel size from transform
        pixel_size_m = abs(self.transform[0]) * 111000

        window_size_pixels = int(np.ceil(1000 / pixel_size_m))
        if window_size_pixels % 2 == 0:
            window_size_pixels += 1  # Make odd for centered window

        # For each pixel, calculate range of elevation in 1km radius
        def elev_range(values):
            return np.nanmax(values) - np.nanmin(values)

        elev_range_arr = generic_filter(
            self.dem_data,
            elev_range,
            size=window_size_pixels,
            mode="constant",
            cval=np.nan,
        )

        # Normalize: typical range might be 0-500m, map to 0-1
        # But cap at reasonable climbing limits
        max_reasonable_climb = 300  # meters
        elev_variation_score = np.minimum(elev_range_arr / max_reasonable_climb, 1.0)

        print(
            f"Elevation variation (1km): min={np.nanmin(elev_variation_score):.3f}, max={np.nanmax(elev_variation_score):.3f}"
        )
        return elev_variation_score

    def calculate_wind_exposure(self) -> np.ndarray:
        """
        Approximate wind exposure using topographic position and slope.
        Wind is stronger on:
        - Exposed ridge crests (high RTP)
        - Steep slopes
        - High elevation open areas

        Output: 0 = low wind exposure, 1 = high wind exposure
        Bees prefer low wind, so we'll invert this later (1 - wind) for suitability
        """
        if self.dem_data is None:
            raise ValueError("DEM data not loaded. Call load_dem() first.")

        # RTP: negative (valleys) = sheltered, positive (ridges) = exposed
        rtp = self.calculate_relative_topographic_position()

        # Slope component: steeper = more exposed
        slope_deg = self.calculate_slope()
        slope_score = slope_deg / np.nanmax(slope_deg)  # Normalize

        # Elevation: higher = more exposed
        elev_min = np.nanmin(self.dem_data)
        elev_max = np.nanmax(self.dem_data)
        elev_score = (self.dem_data - elev_min) / (elev_max - elev_min)

        # Combine: ridges + steep + high elevation = more wind
        # Make positive RTP contribute to wind exposure
        rtp_exposed = np.maximum(0, rtp)  # Only positive RTP (ridges)

        wind_exposure = 0.35 * rtp_exposed + 0.35 * slope_score + 0.30 * elev_score

        wind_exposure = np.clip(wind_exposure, 0, 1)

        print(
            f"Wind exposure: min={np.nanmin(wind_exposure):.3f}, max={np.nanmax(wind_exposure):.3f}"
        )
        return wind_exposure

    def get_terrain_features(self, lon: float, lat: float) -> Dict:
        """Get terrain features for a specific location"""
        if self.dem_data is None:
            raise ValueError("DEM data not loaded. Call load_dem() first.")

        # Convert lon/lat to pixel coordinates
        from rasterio.transform import rowcol

        row, col = rowcol(self.transform, lon, lat)

        # Check if within bounds
        if (
            row < 0
            or row >= self.dem_data.shape[0]
            or col < 0
            or col >= self.dem_data.shape[1]
        ):
            return None

        # Calculate slope and aspect if not already done
        slope = self.calculate_slope()
        aspect = self.calculate_aspect()

        return {
            "elevation": float(self.dem_data[row, col]),
            "slope": float(slope[row, col]),
            "aspect": float(aspect[row, col]),
            "aspect_category": self._categorize_aspect(aspect[row, col]),
        }

    def _categorize_aspect(self, aspect_deg: float) -> str:
        """Categorize aspect into cardinal directions"""
        if np.isnan(aspect_deg):
            return "Flat"
        elif aspect_deg < 22.5 or aspect_deg >= 337.5:
            return "N"
        elif aspect_deg < 67.5:
            return "NE"
        elif aspect_deg < 112.5:
            return "E"
        elif aspect_deg < 157.5:
            return "SE"
        elif aspect_deg < 202.5:
            return "S"
        elif aspect_deg < 247.5:
            return "SW"
        elif aspect_deg < 292.5:
            return "W"
        else:
            return "NW"

    def create_analysis_grid(
        self, bounds: Dict, resolution_m: float = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create a grid of points for analysis"""
        # Calculate grid spacing in degrees
        lat_deg_per_m = 1.0 / 111000
        lon_deg_per_m = 1.0 / (
            111000 * np.cos(np.radians((bounds["min_lat"] + bounds["max_lat"]) / 2))
        )

        lat_spacing = resolution_m * lat_deg_per_m
        lon_spacing = resolution_m * lon_deg_per_m

        lats = np.arange(bounds["min_lat"], bounds["max_lat"], lat_spacing)
        lons = np.arange(bounds["min_lon"], bounds["max_lon"], lon_spacing)

        lon_grid, lat_grid = np.meshgrid(lons, lats)

        print(
            f"Created analysis grid: {lon_grid.shape[0]} x {lon_grid.shape[1]} = {lon_grid.size} points"
        )
        return lon_grid, lat_grid

    def save_geotiff(
        self, data: np.ndarray, output_file: str, nodata_value: float = -9999
    ):
        """Save array as GeoTIFF"""
        with rasterio.open(
            output_file,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=self.crs,
            transform=self.transform,
            nodata=nodata_value,
        ) as dst:
            dst.write(data, 1)
        print(f"Saved: {output_file}")


if __name__ == "__main__":
    # Test the terrain analyzer
    import sys
    from kml_parser import KMLParser

    # Parse KML to get bounds
    parser = KMLParser("InitialFarm.kml")
    farm_data = parser.parse()
    search_bounds = parser.get_search_bounds(farm_data["center"], 4.0)

    # Load and analyze terrain
    analyzer = TerrainAnalyzer("USGS/n28_e081_1arc_v3.tif")
    analyzer.load_dem(search_bounds)

    # Get terrain features for farm center
    features = analyzer.get_terrain_features(
        farm_data["center"][0], farm_data["center"][1]
    )

    print("\n" + "=" * 60)
    print("TERRAIN ANALYSIS - INITIAL FARM LOCATION")
    print("=" * 60)
    if features:
        print(f"Elevation: {features['elevation']:.1f} m")
        print(f"Slope: {features['slope']:.1f}°")
        print(f"Aspect: {features['aspect']:.1f}° ({features['aspect_category']})")
    else:
        print("Location outside DEM bounds")

    # Calculate slope and aspect for entire area
    slope = analyzer.calculate_slope()
    aspect = analyzer.calculate_aspect()

    # Save outputs
    os.makedirs("suitability_maps", exist_ok=True)
    analyzer.save_geotiff(analyzer.dem_data, "suitability_maps/elevation.tif")
    analyzer.save_geotiff(slope, "suitability_maps/slope.tif")
    analyzer.save_geotiff(aspect, "suitability_maps/aspect.tif")
