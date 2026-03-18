#!/usr/bin/env python3
"""Verify zone placement relative to farm - CORRECTED"""

from math import radians, cos, sqrt

# Farm center
farm_lat = 28.776578
farm_lon = 81.480797

# Zone coordinate ranges from the FIXED KML
zone_lon_min, zone_lon_max = 81.448434, 81.521767
zone_lat_min, zone_lat_max = 28.740808, 28.812475

# Calculate distance from farm to farthest zone point
def haversine_approx(lat1, lon1, lat2, lon2):
    # Quick approximation in km
    lat_diff = abs(lat2 - lat1) * 111  # 111 km per degree latitude
    lon_diff = abs(lon2 - lon1) * 111 * cos(radians((lat1 + lat2) / 2))
    return sqrt(lat_diff**2 + lon_diff**2)

# Check all corners
dist_nw = haversine_approx(farm_lat, farm_lon, zone_lat_max, zone_lon_min)
dist_ne = haversine_approx(farm_lat, farm_lon, zone_lat_max, zone_lon_max)
dist_sw = haversine_approx(farm_lat, farm_lon, zone_lat_min, zone_lon_min)
dist_se = haversine_approx(farm_lat, farm_lon, zone_lat_min, zone_lon_max)

print(f'Farm Center: {farm_lat:.6f}°N, {farm_lon:.6f}°E')
print(f'\nZone Bounds:')
print(f'  Longitude: {zone_lon_min:.6f} to {zone_lon_max:.6f}')
print(f'  Latitude:  {zone_lat_min:.6f} to {zone_lat_max:.6f}')
print(f'\nDistance from farm to zone corners:')
print(f'  NW corner: {dist_nw:.2f} km')
print(f'  NE corner: {dist_ne:.2f} km')
print(f'  SW corner: {dist_sw:.2f} km')
print(f'  SE corner: {dist_se:.2f} km')
print(f'\nExpected search radius: 4.0 km')

# Check if farm is inside or near the zones
farm_in_lon = zone_lon_min <= farm_lon <= zone_lon_max
farm_in_lat = zone_lat_min <= farm_lat <= zone_lat_max

print(f'\nFarm position check:')
print(f'  Farm lon ({farm_lon:.4f}) within zone lon range: {farm_in_lon}')
print(f'  Farm lat ({farm_lat:.4f}) within zone lat range: {farm_in_lat}')

if farm_in_lon and farm_in_lat:
    print('\n✓ SUCCESS: Zones are properly centered around the farm!')
elif max(dist_nw, dist_ne, dist_sw, dist_se) <= 5:
    print('\n✓ SUCCESS: Zones are within ~5km of farm (acceptable)')
else:
    print('\n⚠ ISSUE: Zones may not be properly positioned')
