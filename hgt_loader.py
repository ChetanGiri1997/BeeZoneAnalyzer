"""
HGT DEM Loader - Loads USGS HGT binary DEM format to GeoTIFF
SRTM 1-arc-second format is stored as signed 16-bit integers
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
import struct
import os
from pathlib import Path


def load_hgt_dem(hgt_file: str, output_tif: str = None) -> tuple:
    """
    Load USGS HGT (SRTM) binary DEM file and optionally convert to GeoTIFF.
    
    Args:
        hgt_file: Path to .hgt or .dt2 file
        output_tif: If provided, save converted GeoTIFF
        
    Returns:
        (dem_array, transform, crs)
    """
    filename = os.path.basename(hgt_file)
    
    # Extract lat/lon from filename: n28_e081_1arc_v3.dt2 or similar
    parts = filename.replace('.hgt', '').replace('.dt2', '').split('_')
    
    lat_str = parts[0]  # e.g., 'n28'
    lon_str = parts[1]  # e.g., 'e081'
    
    # Parse latitude
    if lat_str.startswith('n'):
        lat = int(lat_str[1:])
    else:  # 's'
        lat = -int(lat_str[1:])
    
    # Parse longitude
    if lon_str.startswith('e'):
        lon = int(lon_str[1:])
    else:  # 'w'
        lon = -int(lon_str[1:])
    
    print(f"Parsing HGT file: {filename}")
    print(f"  Tile location: {lat_str}{lon_str}")
    print(f"  Geographic bounds: {lat}°N to {lat+1}°N, {lon}°E to {lon+1}°E")
    
    # Load binary data
    # SRTM 1-arc-second tiles are 3601 x 3601 pixels (1° = 3600 arc-seconds + 1 for edge)
    # Data is signed 16-bit integers, big-endian
    
    with open(hgt_file, 'rb') as f:
        data = f.read()
    
    # Convert bytes to 16-bit integers (big-endian)
    num_pixels = len(data) // 2
    dem_1d = np.array(struct.unpack('>' + 'h' * num_pixels, data), dtype=np.float32)
    
    # Reshape based on tile size
    if num_pixels == 3601 * 3601:
        dem_array = dem_1d.reshape(3601, 3601)
        tile_size = 3601
    elif num_pixels == 1201 * 1201:
        # 3-arc-second resolution
        dem_array = dem_1d.reshape(1201, 1201)
        tile_size = 1201
    else:
        raise ValueError(f"Unexpected DEM size: {num_pixels} pixels")
    
    print(f"  Loaded DEM array: {dem_array.shape}")
    print(f"  Elevation range: {np.nanmin(dem_array):.1f} - {np.nanmax(dem_array):.1f} m")
    
    # Replace -32768 (void) with NaN
    dem_array[dem_array == -32768] = np.nan
    
    # Create geotransform
    # SRTM data goes from top-left (NW) corner, sample at top-left corner
    # Each pixel is ~30m for 1-arc-second (1/120 degree)
    pixel_size = 1.0 / (tile_size - 1)  # degrees per pixel
    
    transform = from_bounds(lon, lat, lon + 1, lat + 1, dem_array.shape[1], dem_array.shape[0])
    crs = "EPSG:4326"  # WGS84
    
    # Save as GeoTIFF if requested
    if output_tif:
        os.makedirs(os.path.dirname(output_tif) or ".", exist_ok=True)
        
        with rasterio.open(
            output_tif,
            'w',
            driver='GTiff',
            height=dem_array.shape[0],
            width=dem_array.shape[1],
            count=1,
            dtype=dem_array.dtype,
            crs=crs,
            transform=transform,
            nodata=np.nan,
            compress='lzw'
        ) as dst:
            dst.write(dem_array, 1)
        
        print(f"Saved GeoTIFF: {output_tif}")
    
    return dem_array, transform, crs


def batch_convert_hgt_to_tif(input_dir: str, output_dir: str):
    """Convert all HGT files in a directory to GeoTIFF format"""
    input_path = Path(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    hgt_files = list(input_path.glob("*.hgt")) + list(input_path.glob("*.dt2"))
    
    print(f"Found {len(hgt_files)} HGT files")
    
    for hgt_file in hgt_files:
        output_tif = os.path.join(output_dir, hgt_file.stem + ".tif")
        try:
            load_hgt_dem(str(hgt_file), output_tif)
        except Exception as e:
            print(f"Error processing {hgt_file}: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        hgt_file = sys.argv[1]
        output_tif = sys.argv[2] if len(sys.argv) > 2 else None
        load_hgt_dem(hgt_file, output_tif)
    else:
        print("Usage: python hgt_loader.py <input.hgt> [output.tif]")
