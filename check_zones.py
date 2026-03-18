#!/usr/bin/env python3
"""Verify zone placement relative to farm"""

from math import radians, cos, sqrt

# Farm center
farm_lat = 28.776578
farm_lon = 81.480797

# Zone coordinate ranges from the KML
zone_lon_min, zone_lon_max = 81.511797, 81.775797
zone_lat_min, zone_lat_max = 28.518578, 28.776578

# Calculate distance from farm to farthest zone point
def haversine_approx(lat1, lon1, lat2, lon2):
    # Quick approximation in km
    lat_diff = abs(lat2 - lat1) * 111  # 111 km per degree latitude
    lon_diff = abs(lon2 - lon1) * 111 * cos(radians((lat1 + lat2) / 2))
    return sqrt(lat_diff**2 + lon_diff**2)

dist_to_far_corner = haversine_approx(farm_lat, farm_lon, zone_lat_min, zone_lon_max)
dist_to_close_edge = haversine_approx(farm_lat, farm_lon, farm_lat, zone_lon_min)

print(f'Farm Center: {farm_lat:.6f}°N, {farm_lon:.6f}°E')
print(f'Zone Lon range: {zone_lon_min:.6f} to {zone_lon_max:.6f}')
print(f'Zone Lat range: {zone_lat_min:.6f} to {zone_lat_max:.6f}')
print()
print(f'Distance from farm to closest zone edge: {dist_to_close_edge:.2f} km')
print(f'Distance from farm to farthest zone corner: {dist_to_far_corner:.2f} km')
print(f'Expected search radius: 4.0 km')
print()

# The issue: zones are EAST and SOUTH of farm, not centered around it
print('ISSUE DETECTED:')
print(f'  Zones are only EAST of farm (lon > {farm_lon:.2f})')
print(f'  Zones are only SOUTH of farm (lat <= {farm_lat:.2f})')
print('  This is because the KML generator used wrong pixel offset direction!')
