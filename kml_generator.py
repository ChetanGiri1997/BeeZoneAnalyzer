"""
KML Generator - Create KML files for recommended locations
"""

import simplekml
from typing import List, Dict, Tuple
import numpy as np


class KMLGenerator:
    def __init__(self):
        self.kml = simplekml.Kml()

    def create_location_kml(
        self,
        locations: List[Dict],
        output_file: str,
        farming_type: str = "General",
        color: str = "ff0000ff",
    ):
        """Create KML file with recommended locations"""
        kml = simplekml.Kml()
        kml.document.name = f"{farming_type} Suitable Locations"

        # Create folder for locations
        folder = kml.newfolder(name=f"Top {len(locations)} Locations")

        for i, loc in enumerate(locations, 1):
            # Create placemark
            pnt = folder.newpoint(
                name=f"Location {i}", coords=[(loc["lon"], loc["lat"])]
            )

            # Set description
            desc = self._create_description(loc, farming_type, i)
            pnt.description = desc

            # Set style based on suitability score
            pnt.style.iconstyle.color = self._get_color_from_score(loc["score"])
            pnt.style.iconstyle.scale = 1.2
            pnt.style.iconstyle.icon.href = (
                "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"
            )

            # Add label
            pnt.style.labelstyle.scale = 0.8

        # Save KML
        kml.save(output_file)
        print(f"Saved KML: {output_file}")
        return output_file

    def create_comparison_kml(
        self,
        bee_locations: List[Dict],
        mandarin_locations: List[Dict],
        output_file: str,
    ):
        """Create KML with both bee and mandarin locations"""
        kml = simplekml.Kml()
        kml.document.name = "Agricultural Suitability Analysis"

        # Bee farming folder
        bee_folder = kml.newfolder(name="Bee Farming Locations")
        for i, loc in enumerate(bee_locations, 1):
            pnt = bee_folder.newpoint(
                name=f"Bee Farm {i}", coords=[(loc["lon"], loc["lat"])]
            )
            pnt.description = self._create_description(loc, "Bee Farming", i)
            pnt.style.iconstyle.color = "ff00ff00"  # Green
            pnt.style.iconstyle.scale = 1.2
            pnt.style.iconstyle.icon.href = (
                "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"
            )

        # Mandarin farming folder
        mandarin_folder = kml.newfolder(name="Mandarin Orange Farming Locations")
        for i, loc in enumerate(mandarin_locations, 1):
            pnt = mandarin_folder.newpoint(
                name=f"Mandarin Farm {i}", coords=[(loc["lon"], loc["lat"])]
            )
            pnt.description = self._create_description(loc, "Mandarin Farming", i)
            pnt.style.iconstyle.color = "ff0080ff"  # Orange
            pnt.style.iconstyle.scale = 1.2
            pnt.style.iconstyle.icon.href = (
                "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"
            )

        kml.save(output_file)
        print(f"Saved comparison KML: {output_file}")
        return output_file

    def create_initial_farm_kml(
        self, farm_data: Dict, analysis_results: Dict, output_file: str
    ):
        """Create KML showing initial farm with analysis results"""
        kml = simplekml.Kml()
        kml.document.name = "Initial Farm Analysis"

        # Create polygon for farm boundary
        if "coordinates" in farm_data:
            coords = [(c[0], c[1]) for c in farm_data["coordinates"]]
            pol = kml.newpolygon(name=farm_data["name"])
            pol.outerboundaryis = coords
            pol.style.linestyle.color = "ff0000ff"
            pol.style.linestyle.width = 3
            pol.style.polystyle.color = "400000ff"

            # Add description
            desc = f"""
            <h3>Initial Farm Analysis</h3>
            <p><b>Area:</b> {farm_data["area_hectares"]:.2f} hectares</p>
            <p><b>Center:</b> {farm_data["center"][1]:.6f}°N, {farm_data["center"][0]:.6f}°E</p>
            
            <h4>Bee Farming Suitability</h4>
            <p><b>Probability:</b> {analysis_results["bee_farming"]["probability"]:.1f}%</p>
            <p><b>Score:</b> {analysis_results["bee_farming"]["total_score"]:.2f}/1.0</p>
            
            <h4>Mandarin Orange Farming Suitability</h4>
            <p><b>Probability:</b> {analysis_results["mandarin_farming"]["probability"]:.1f}%</p>
            <p><b>Score:</b> {analysis_results["mandarin_farming"]["total_score"]:.2f}/1.0</p>
            
            <h4>Environmental Features</h4>
            <p><b>Elevation:</b> {analysis_results["features"].get("elevation", "N/A")} m</p>
            <p><b>Slope:</b> {analysis_results["features"].get("slope", "N/A")}°</p>
            <p><b>Aspect:</b> {analysis_results["features"].get("aspect_category", "N/A")}</p>
            <p><b>NDVI:</b> {analysis_results["features"].get("ndvi", "N/A")}</p>
            """
            pol.description = desc

        # Add center point
        center_pnt = kml.newpoint(
            name="Farm Center",
            coords=[(farm_data["center"][0], farm_data["center"][1])],
        )
        center_pnt.style.iconstyle.color = "ffff0000"
        center_pnt.style.iconstyle.scale = 1.5

        kml.save(output_file)
        print(f"Saved initial farm KML: {output_file}")
        return output_file

    def _create_description(self, loc: Dict, farming_type: str, rank: int) -> str:
        """Create HTML description for location"""
        desc = f"""
        <h3>{farming_type} - Location #{rank}</h3>
        <p><b>Suitability Score:</b> {loc["score"]:.2f}/1.0 ({loc["probability"]:.1f}%)</p>
        <p><b>Coordinates:</b> {loc["lat"]:.6f}°N, {loc["lon"]:.6f}°E</p>
        
        <h4>Environmental Features</h4>
        <p><b>Elevation:</b> {loc.get("elevation", "N/A")} m</p>
        <p><b>Slope:</b> {loc.get("slope", "N/A")}°</p>
        <p><b>Aspect:</b> {loc.get("aspect_category", "N/A")}</p>
        <p><b>NDVI:</b> {loc.get("ndvi", "N/A")}</p>
        
        <h4>Component Scores</h4>
        """

        if "component_scores" in loc:
            for component, score in loc["component_scores"].items():
                desc += f"<p><b>{component.title()}:</b> {score:.2f}</p>"

        return desc

    def _get_color_from_score(self, score: float) -> str:
        """Get KML color based on suitability score"""
        # Score 0-1, map to color gradient (red to green)
        # KML color format: aabbggrr (alpha, blue, green, red)

        if score >= 0.8:
            return "ff00ff00"  # Bright green
        elif score >= 0.6:
            return "ff00ff80"  # Light green
        elif score >= 0.4:
            return "ff00ffff"  # Yellow
        elif score >= 0.2:
            return "ff0080ff"  # Orange
        else:
            return "ff0000ff"  # Red


if __name__ == "__main__":
    # Test KML generator
    generator = KMLGenerator()

    # Sample locations
    test_locations = [
        {
            "lon": 81.481,
            "lat": 28.777,
            "score": 0.85,
            "probability": 85,
            "elevation": 1200,
            "slope": 8,
            "aspect_category": "SE",
            "ndvi": 0.65,
            "component_scores": {"elevation": 0.9, "slope": 0.8, "aspect": 1.0},
        },
        {
            "lon": 81.485,
            "lat": 28.780,
            "score": 0.75,
            "probability": 75,
            "elevation": 1150,
            "slope": 12,
            "aspect_category": "S",
            "ndvi": 0.60,
            "component_scores": {"elevation": 0.85, "slope": 0.7, "aspect": 0.9},
        },
    ]

    generator.create_location_kml(test_locations, "test_locations.kml", "Bee Farming")
    print("Test KML created successfully!")
