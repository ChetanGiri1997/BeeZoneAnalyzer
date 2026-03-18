"""
KML Parser - Extract coordinates and metadata from InitialFarm.kml
"""

import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict
import numpy as np


class KMLParser:
    def __init__(self, kml_file: str):
        self.kml_file = kml_file
        self.namespace = {
            "kml": "http://www.opengis.net/kml/2.2",
            "gx": "http://www.google.com/kml/ext/2.2",
        }

    def parse(self) -> Dict:
        """Parse KML file and extract farm location data"""
        tree = ET.parse(self.kml_file)
        root = tree.getroot()

        # Find the placemark
        placemark = root.find(".//kml:Placemark", self.namespace)

        if placemark is None:
            raise ValueError("No Placemark found in KML file")

        # Extract name
        name_elem = placemark.find("kml:name", self.namespace)
        name = name_elem.text if name_elem is not None else "Unknown"

        # Extract coordinates
        coords_elem = placemark.find(".//kml:coordinates", self.namespace)
        if coords_elem is None:
            raise ValueError("No coordinates found in KML file")

        coordinates = self._parse_coordinates(coords_elem.text)

        # Calculate center point and bounds
        center = self._calculate_center(coordinates)
        bounds = self._calculate_bounds(coordinates)
        area = self._calculate_area(coordinates)

        return {
            "name": name,
            "coordinates": coordinates,
            "center": center,
            "bounds": bounds,
            "area_hectares": area,
            "num_points": len(coordinates),
        }

    def _parse_coordinates(self, coord_text: str) -> List[Tuple[float, float, float]]:
        """Parse coordinate string into list of (lon, lat, alt) tuples"""
        coords = []
        for coord in coord_text.strip().split():
            parts = coord.split(",")
            if len(parts) >= 2:
                lon = float(parts[0])
                lat = float(parts[1])
                alt = float(parts[2]) if len(parts) > 2 else 0.0
                coords.append((lon, lat, alt))
        return coords

    def _calculate_center(
        self, coordinates: List[Tuple[float, float, float]]
    ) -> Tuple[float, float]:
        """Calculate center point of polygon"""
        lons = [c[0] for c in coordinates]
        lats = [c[1] for c in coordinates]
        return (np.mean(lons), np.mean(lats))

    def _calculate_bounds(self, coordinates: List[Tuple[float, float, float]]) -> Dict:
        """Calculate bounding box"""
        lons = [c[0] for c in coordinates]
        lats = [c[1] for c in coordinates]
        return {
            "min_lon": min(lons),
            "max_lon": max(lons),
            "min_lat": min(lats),
            "max_lat": max(lats),
        }

    def _calculate_area(self, coordinates: List[Tuple[float, float, float]]) -> float:
        """Calculate approximate area in hectares using Shoelace formula"""
        # Convert to meters approximately (rough estimation)
        # 1 degree latitude ≈ 111 km
        # 1 degree longitude ≈ 111 km * cos(latitude)

        if len(coordinates) < 3:
            return 0.0

        # Get average latitude for longitude conversion
        avg_lat = np.mean([c[1] for c in coordinates])
        lat_to_m = 111000  # meters per degree latitude
        lon_to_m = 111000 * np.cos(np.radians(avg_lat))  # meters per degree longitude

        # Convert to meters
        x = np.array([c[0] * lon_to_m for c in coordinates])
        y = np.array([c[1] * lat_to_m for c in coordinates])

        # Shoelace formula
        area_m2 = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        area_hectares = area_m2 / 10000  # Convert to hectares

        return area_hectares

    def get_search_bounds(self, center: Tuple[float, float], radius_km: float) -> Dict:
        """Calculate search bounds for given radius around center point"""
        lon, lat = center

        # Approximate degrees per km
        # 1 degree latitude ≈ 111 km
        # 1 degree longitude ≈ 111 km * cos(latitude)
        lat_deg_per_km = 1.0 / 111.0
        lon_deg_per_km = 1.0 / (111.0 * np.cos(np.radians(lat)))

        lat_offset = radius_km * lat_deg_per_km
        lon_offset = radius_km * lon_deg_per_km

        return {
            "min_lon": lon - lon_offset,
            "max_lon": lon + lon_offset,
            "min_lat": lat - lat_offset,
            "max_lat": lat + lat_offset,
            "center_lon": lon,
            "center_lat": lat,
            "radius_km": radius_km,
        }


if __name__ == "__main__":
    # Test the parser
    parser = KMLParser("InitialFarm.kml")
    farm_data = parser.parse()

    print("=" * 60)
    print("INITIAL FARM ANALYSIS")
    print("=" * 60)
    print(f"Name: {farm_data['name']}")
    print(f"Center: {farm_data['center'][1]:.6f}°N, {farm_data['center'][0]:.6f}°E")
    print(f"Area: {farm_data['area_hectares']:.2f} hectares")
    print(f"Number of boundary points: {farm_data['num_points']}")
    print(f"\nBounds:")
    print(
        f"  Latitude: {farm_data['bounds']['min_lat']:.6f} to {farm_data['bounds']['max_lat']:.6f}"
    )
    print(
        f"  Longitude: {farm_data['bounds']['min_lon']:.6f} to {farm_data['bounds']['max_lon']:.6f}"
    )

    # Calculate search area
    search_bounds = parser.get_search_bounds(farm_data["center"], 4.0)
    print(f"\nSearch area (4km radius):")
    print(
        f"  Latitude: {search_bounds['min_lat']:.6f} to {search_bounds['max_lat']:.6f}"
    )
    print(
        f"  Longitude: {search_bounds['min_lon']:.6f} to {search_bounds['max_lon']:.6f}"
    )
