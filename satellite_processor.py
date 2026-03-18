"""
Satellite Processor - Extract vegetation indices and features from satellite imagery
Supports Sentinel-2 L2A and Landsat 8/9 L2 data for ecological analysis
"""

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pathlib import Path
import tarfile
import zipfile
from typing import Dict, List, Tuple
import os
from pyproj import Transformer


class SatelliteProcessor:
    def __init__(self, landsat_dir: str, sentinel_dir: str):
        self.landsat_dir = Path(landsat_dir)
        self.sentinel_dir = Path(sentinel_dir)

    def find_latest_landsat(self) -> Path:
        """Find the most recent Landsat scene"""
        landsat_files = list(self.landsat_dir.glob("LC0*_L2SP_*.tar"))
        if not landsat_files:
            return None
        # Sort by date in filename (YYYYMMDD)
        landsat_files.sort(reverse=True)
        return landsat_files[0]

    def find_latest_sentinel(self) -> Path:
        """Find the most recent Sentinel-2 scene"""
        sentinel_files = list(self.sentinel_dir.glob("S2*_MSIL2A_*.zip"))
        if not sentinel_files:
            return None
        # Sort by date in filename (YYYYMMDD)
        sentinel_files.sort(reverse=True)
        return sentinel_files[0]

    def extract_landsat_bands(
        self, tar_file: Path, bounds: Dict, output_dir: str = "temp_landsat"
    ) -> Dict:
        """Extract relevant Landsat bands from tar file"""
        os.makedirs(output_dir, exist_ok=True)

        print(f"Extracting Landsat: {tar_file.name}")

        bands = {}
        with tarfile.open(tar_file, "r") as tar:
            members = tar.getmembers()

            # Find band files
            band_mapping = {
                "B2": "blue",  # Blue
                "B3": "green",  # Green
                "B4": "red",  # Red
                "B5": "nir",  # Near Infrared
                "B6": "swir1",  # Shortwave Infrared 1
                "B7": "swir2",  # Shortwave Infrared 2
                "ST_B10": "thermal",  # Thermal (for temperature)
            }

            for member in members:
                for band_code, band_name in band_mapping.items():
                    if f"_{band_code}.TIF" in member.name:
                        # Extract file
                        tar.extract(member, output_dir)
                        extracted_path = os.path.join(output_dir, member.name)

                        # Read and crop to bounds
                        try:
                            with rasterio.open(extracted_path) as src:
                                scene_crs = src.crs
                                scene_bounds = src.bounds
                                
                                # Transform bounds from WGS84 to scene CRS if different
                                if str(scene_crs) != 'EPSG:4326':
                                    transformer = Transformer.from_crs('EPSG:4326', scene_crs, always_xy=True)
                                    # Transform all four corners
                                    min_x, min_y = transformer.transform(bounds["min_lon"], bounds["min_lat"])
                                    max_x, max_y = transformer.transform(bounds["max_lon"], bounds["max_lat"])
                                    
                                    # Calculate intersection
                                    int_left = max(min_x, scene_bounds.left)
                                    int_bottom = max(min_y, scene_bounds.bottom)
                                    int_right = min(max_x, scene_bounds.right)
                                    int_top = min(max_y, scene_bounds.top)
                                else:
                                    # Calculate intersection directly
                                    int_left = max(bounds["min_lon"], scene_bounds.left)
                                    int_bottom = max(bounds["min_lat"], scene_bounds.bottom)
                                    int_right = min(bounds["max_lon"], scene_bounds.right)
                                    int_top = min(bounds["max_lat"], scene_bounds.top)
                                
                                if int_left < int_right and int_bottom < int_top:
                                    window = from_bounds(
                                        int_left, int_bottom, int_right, int_top,
                                        src.transform,
                                    )
                                    data = src.read(1, window=window)
                                    if data.size > 0:
                                        bands[band_name] = data.astype(np.float32)
                                        if band_name == "nir":
                                            self.transform = src.window_transform(window)
                                            self.crs = src.crs
                                            # Store bounds for output coord conversion
                                            self.scene_bounds = (int_left, int_bottom, int_right, int_top)
                                else:
                                    print(f"    Warning: Scene does not overlap bounds for {band_name}")
                        except Exception as e:
                            print(f"    Warning: Could not read {band_name}: {e}")

        return bands

    def calculate_ndvi(self, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Calculate Normalized Difference Vegetation Index"""
        # Avoid division by zero
        denominator = nir + red
        ndvi = np.where(denominator != 0, (nir - red) / denominator, 0)
        # Clip to valid range
        ndvi = np.clip(ndvi, -1, 1)
        return ndvi

    def calculate_ndwi(self, green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Calculate Normalized Difference Water Index"""
        denominator = green + nir
        ndwi = np.where(denominator != 0, (green - nir) / denominator, 0)
        ndwi = np.clip(ndwi, -1, 1)
        return ndwi

    def calculate_evi(
        self, blue: np.ndarray, red: np.ndarray, nir: np.ndarray
    ) -> np.ndarray:
        """Calculate Enhanced Vegetation Index"""
        # EVI = 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))
        denominator = nir + 6 * red - 7.5 * blue + 1
        evi = np.where(denominator != 0, 2.5 * (nir - red) / denominator, 0)
        evi = np.clip(evi, -1, 1)
        return evi

    def calculate_savi(
        self, red: np.ndarray, nir: np.ndarray, L: float = 0.5
    ) -> np.ndarray:
        """Calculate Soil Adjusted Vegetation Index"""
        # SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
        denominator = nir + red + L
        savi = np.where(denominator != 0, ((nir - red) / denominator) * (1 + L), 0)
        savi = np.clip(savi, -1, 1)
        return savi

    def calculate_vegetation_seasonal_variability(
        self, ndvi_data: np.ndarray, window_size: int = 5
    ) -> np.ndarray:
        """
        Calculate vegetation seasonal variability proxy using local NDVI variance.
        Areas with stable moderate vegetation (diverse flowering) score high.
        Extremely stable vegetation (dense forest) or highly variable (sparse) score low.

        This approximates temporal flowering stability without multi-temporal data.
        We use spatial texture as proxy for diversity/stability.
        """
        from scipy.ndimage import generic_filter

        def local_variance(values):
            return np.nanvar(values)

        local_var = generic_filter(
            ndvi_data,
            local_variance,
            size=window_size,
            mode="constant",
            cval=np.nan,
        )

        # Normalize local variance
        var_min = np.nanpercentile(local_var, 5)
        var_max = np.nanpercentile(local_var, 95)

        if var_max > var_min:
            variability_norm = (local_var - var_min) / (var_max - var_min)
        else:
            variability_norm = np.ones_like(local_var)

        # Optimal variability is moderate (some diversity in vegetation types)
        # Score highest at intermediate variability, lower at extremes
        variability_score = np.exp(-((variability_norm - 0.5) ** 2) / (2 * 0.2**2))

        return np.clip(variability_score, 0, 1)

    def process_landsat(self, bounds: Dict) -> Dict:
        """Process Landsat imagery and calculate indices"""
        latest_file = self.find_latest_landsat()
        if not latest_file:
            print("No Landsat files found")
            return None

        print(f"Processing Landsat: {latest_file.name}")

        # Extract bands
        bands = self.extract_landsat_bands(latest_file, bounds)

        if not all(k in bands for k in ["red", "nir", "green", "blue"]):
            print("Missing required bands")
            return None

        # Calculate indices
        indices = {
            "ndvi": self.calculate_ndvi(bands["red"], bands["nir"]),
            "ndwi": self.calculate_ndwi(bands["green"], bands["nir"]),
            "evi": self.calculate_evi(bands["blue"], bands["red"], bands["nir"]),
            "savi": self.calculate_savi(bands["red"], bands["nir"]),
        }

        # Calculate statistics
        stats = {}
        for name, data in indices.items():
            if data.size > 0 and not np.all(np.isnan(data)):
                stats[name] = {
                    "mean": float(np.nanmean(data)),
                    "std": float(np.nanstd(data)),
                    "min": float(np.nanmin(data)),
                    "max": float(np.nanmax(data)),
                }
            else:
                stats[name] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                }

        print("\nVegetation Indices:")
        for name, stat in stats.items():
            print(f"  {name.upper()}: mean={stat['mean']:.3f}, std={stat['std']:.3f}")

        return {
            "bands": bands,
            "indices": indices,
            "stats": stats,
            "date": self._extract_date_from_filename(latest_file.name),
        }

    def _extract_date_from_filename(self, filename: str) -> str:
        """Extract date from Landsat/Sentinel filename"""
        # Landsat: LC08_L2SP_143040_20260104_...
        # Sentinel: S2B_MSIL2A_20260122T051019_...
        parts = filename.split("_")
        for part in parts:
            if len(part) == 8 and part.isdigit():
                return f"{part[:4]}-{part[4:6]}-{part[6:8]}"
        return "Unknown"

    def get_pixel_value(self, data: np.ndarray, lon: float, lat: float) -> float:
        """Get pixel value at specific coordinates"""
        from rasterio.transform import rowcol

        row, col = rowcol(self.transform, lon, lat)

        if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
            return float(data[row, col])
        return np.nan

    def save_geotiff(self, data: np.ndarray, output_file: str):
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
            nodata=-9999,
        ) as dst:
            dst.write(data, 1)
        print(f"Saved: {output_file}")


if __name__ == "__main__":
    from kml_parser import KMLParser

    # Parse KML to get bounds
    parser = KMLParser("InitialFarm.kml")
    farm_data = parser.parse()
    search_bounds = parser.get_search_bounds(farm_data["center"], 4.0)

    # Process satellite imagery
    processor = SatelliteProcessor("USGS", "sentinal2")
    result = processor.process_landsat(search_bounds)

    if result:
        print("\n" + "=" * 60)
        print("SATELLITE ANALYSIS - INITIAL FARM LOCATION")
        print("=" * 60)
        print(f"Image Date: {result['date']}")

        # Get values at farm center
        for name, data in result["indices"].items():
            value = processor.get_pixel_value(
                data, farm_data["center"][0], farm_data["center"][1]
            )
            print(f"{name.upper()} at farm center: {value:.3f}")

        # Save outputs
        os.makedirs("suitability_maps", exist_ok=True)
        for name, data in result["indices"].items():
            processor.save_geotiff(data, f"suitability_maps/{name}.tif")
