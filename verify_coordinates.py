#!/usr/bin/env python3
"""Verify that best zones coordinates are within dataset bounds"""

import rasterio
import json
import re

# Check DEM bounds
print('=== DEM BOUNDS ===')
with rasterio.open('USGS/n28_e081_1arc_v3.tif') as src:
    dem_bounds = src.bounds
    print(f'DEM Bounds: {dem_bounds}')
    print(f'  West:  {dem_bounds.left:.6f}°E')
    print(f'  East:  {dem_bounds.right:.6f}°E')
    print(f'  South: {dem_bounds.bottom:.6f}°N')
    print(f'  North: {dem_bounds.top:.6f}°N')

# Check InitialFarm KML coordinates
print('\n=== FARM LOCATION ===')
with open('suitability_maps/initialfarm_probability_report.json') as f:
    report = json.load(f)
    farm_lat = report["farm_center"]["lat"]
    farm_lon = report["farm_center"]["lon"]
    print(f'Farm Center: {farm_lat:.6f}°N, {farm_lon:.6f}°E')

# Check best zones KML coordinates
print('\n=== BEST ZONES KML COORDINATES ===')
with open('suitability_maps/initialfarm_best_zones.kml') as f:
    content = f.read()

coords = re.findall(r'<coordinates>([^<]+)</coordinates>', content)
print(f'Found {len(coords)} zone coordinates')

# Parse and check bounds
all_lons = []
all_lats = []
outside_count = 0

for c in coords:
    parts = c.strip().split(',')
    lon, lat = float(parts[0]), float(parts[1])
    all_lons.append(lon)
    all_lats.append(lat)
    
    # Check if within DEM bounds
    if not (dem_bounds.left <= lon <= dem_bounds.right and 
            dem_bounds.bottom <= lat <= dem_bounds.top):
        outside_count += 1

print(f'\nFirst 5 zone coordinates:')
for i in range(min(5, len(all_lons))):
    in_bounds = (dem_bounds.left <= all_lons[i] <= dem_bounds.right and 
                 dem_bounds.bottom <= all_lats[i] <= dem_bounds.top)
    status = "✓ IN BOUNDS" if in_bounds else "✗ OUTSIDE"
    print(f'  Zone {i+1}: {all_lats[i]:.6f}°N, {all_lons[i]:.6f}°E - {status}')

print(f'\n=== SUMMARY ===')
print(f'Zone Longitude range: {min(all_lons):.6f} to {max(all_lons):.6f}')
print(f'Zone Latitude range:  {min(all_lats):.6f} to {max(all_lats):.6f}')
print(f'DEM Longitude range:  {dem_bounds.left:.6f} to {dem_bounds.right:.6f}')
print(f'DEM Latitude range:   {dem_bounds.bottom:.6f} to {dem_bounds.top:.6f}')
print(f'\nCoordinates outside bounds: {outside_count} / {len(coords)}')

if outside_count > 0:
    print('\n⚠ ERROR: Some coordinates are outside the dataset bounds!')
    print('This is a bug in coordinate calculation.')
