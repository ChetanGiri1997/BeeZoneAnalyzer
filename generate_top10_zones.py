#!/usr/bin/env python3
"""
Generate Top 10 Best Location Polygons for Bee + Mandarin Co-Location
Uses spatial clustering to group high-suitability areas into distinct zones
"""

import numpy as np
import rasterio
from scipy import ndimage
from scipy.ndimage import label, find_objects
from shapely.geometry import Polygon, MultiPoint
from shapely.ops import unary_union
import json


def generate_top10_polygons(
    combined_tif: str,
    bee_tif: str,
    mandarin_tif: str,
    output_kml: str,
    output_json: str,
    top_n: int = 10,
    min_cluster_pixels: int = 5
):
    """
    Generate top N best location polygons from suitability rasters.
    
    Args:
        combined_tif: Path to combined suitability GeoTIFF
        bee_tif: Path to bee suitability GeoTIFF
        mandarin_tif: Path to mandarin suitability GeoTIFF
        output_kml: Path to output KML file
        output_json: Path to output JSON summary
        top_n: Number of top zones to identify
        min_cluster_pixels: Minimum pixels to form a valid cluster
    """
    
    print("Loading suitability rasters...")
    
    # Load rasters
    with rasterio.open(combined_tif) as src:
        combined = src.read(1)
        transform = src.transform
        crs = src.crs
    
    with rasterio.open(bee_tif) as src:
        bee = src.read(1)
    
    with rasterio.open(mandarin_tif) as src:
        mandarin = src.read(1)
    
    print(f"Raster shape: {combined.shape}")
    print(f"Combined score range: {np.nanmin(combined):.3f} - {np.nanmax(combined):.3f}")
    
    # Find threshold for top 15% pixels (gives more clusters to choose from)
    threshold = np.nanpercentile(combined, 85)
    print(f"Top 15% threshold: {threshold:.3f}")
    
    # Create binary mask of high-scoring areas
    high_score_mask = combined >= threshold
    
    # Label connected components (clusters)
    structure = np.ones((3, 3))  # 8-connectivity
    labeled_array, num_clusters = label(high_score_mask, structure=structure)
    print(f"Found {num_clusters} initial clusters")
    
    # Analyze each cluster
    clusters = []
    for cluster_id in range(1, num_clusters + 1):
        cluster_mask = labeled_array == cluster_id
        pixel_count = np.sum(cluster_mask)
        
        if pixel_count < min_cluster_pixels:
            continue
        
        # Get cluster statistics
        combined_scores = combined[cluster_mask]
        bee_scores = bee[cluster_mask]
        mandarin_scores = mandarin[cluster_mask]
        
        # Get pixel coordinates
        rows, cols = np.where(cluster_mask)
        
        # Convert to geographic coordinates
        coords = []
        for r, c in zip(rows, cols):
            lon, lat = transform * (c + 0.5, r + 0.5)
            coords.append((lon, lat))
        
        # Calculate cluster centroid
        center_lon = np.mean([c[0] for c in coords])
        center_lat = np.mean([c[1] for c in coords])
        
        # Create convex hull polygon
        if len(coords) >= 3:
            points = MultiPoint(coords)
            hull = points.convex_hull
            if hull.geom_type == 'Polygon':
                polygon_coords = list(hull.exterior.coords)
            else:
                # For very small clusters, create a buffer
                hull = points.centroid.buffer(0.001)
                polygon_coords = list(hull.exterior.coords)
        else:
            # Create small square for tiny clusters
            polygon_coords = [
                (center_lon - 0.0005, center_lat - 0.0005),
                (center_lon + 0.0005, center_lat - 0.0005),
                (center_lon + 0.0005, center_lat + 0.0005),
                (center_lon - 0.0005, center_lat + 0.0005),
                (center_lon - 0.0005, center_lat - 0.0005),
            ]
        
        clusters.append({
            'id': cluster_id,
            'pixel_count': int(pixel_count),
            'center_lon': float(center_lon),
            'center_lat': float(center_lat),
            'combined_mean': float(np.mean(combined_scores)),
            'combined_max': float(np.max(combined_scores)),
            'bee_mean': float(np.mean(bee_scores)),
            'bee_max': float(np.max(bee_scores)),
            'mandarin_mean': float(np.mean(mandarin_scores)),
            'mandarin_max': float(np.max(mandarin_scores)),
            'polygon_coords': polygon_coords,
            'area_hectares': float(pixel_count * 30 * 30 / 10000),  # Approx area
        })
    
    print(f"Valid clusters (>= {min_cluster_pixels} pixels): {len(clusters)}")
    
    # Sort by combined mean score and take top N
    clusters.sort(key=lambda x: x['combined_mean'], reverse=True)
    top_clusters = clusters[:top_n]
    
    print(f"\nTop {len(top_clusters)} Best Zones:")
    for i, c in enumerate(top_clusters):
        print(f"  {i+1}. Combined: {c['combined_mean']:.3f}, "
              f"Bee: {c['bee_mean']:.3f}, Mandarin: {c['mandarin_mean']:.3f}, "
              f"Area: {c['area_hectares']:.1f} ha")
    
    # Generate KML
    kml = generate_kml(top_clusters)
    with open(output_kml, 'w') as f:
        f.write(kml)
    print(f"\nSaved: {output_kml}")
    
    # Generate JSON summary
    summary = {
        'total_clusters_found': len(clusters),
        'top_zones': top_n,
        'threshold_percentile': 85,
        'threshold_score': float(threshold),
        'zones': []
    }
    
    for i, c in enumerate(top_clusters):
        summary['zones'].append({
            'rank': i + 1,
            'center': {'lat': c['center_lat'], 'lon': c['center_lon']},
            'area_hectares': c['area_hectares'],
            'pixel_count': c['pixel_count'],
            'scores': {
                'combined_mean': c['combined_mean'],
                'combined_max': c['combined_max'],
                'bee_mean': c['bee_mean'],
                'bee_max': c['bee_max'],
                'mandarin_mean': c['mandarin_mean'],
                'mandarin_max': c['mandarin_max'],
            },
            'classification': classify_score(c['combined_mean'])
        })
    
    with open(output_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {output_json}")
    
    return top_clusters


def classify_score(score):
    """Classify suitability score"""
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.6:
        return "High"
    elif score >= 0.4:
        return "Moderate"
    else:
        return "Low"


def generate_kml(clusters):
    """Generate KML with polygon zones"""
    
    kml = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Top 10 Best Co-Location Zones</name>
    <description>Optimal zones for combined Bee Farming + Mandarin Orange cultivation</description>
    
    <!-- Styles for different suitability levels -->
    <Style id="excellent">
      <LineStyle><color>ff00ff00</color><width>3</width></LineStyle>
      <PolyStyle><color>8000ff00</color></PolyStyle>
    </Style>
    <Style id="high">
      <LineStyle><color>ff00aaff</color><width>3</width></LineStyle>
      <PolyStyle><color>8000aaff</color></PolyStyle>
    </Style>
    <Style id="moderate">
      <LineStyle><color>ff00ffff</color><width>2</width></LineStyle>
      <PolyStyle><color>6000ffff</color></PolyStyle>
    </Style>
'''
    
    for i, c in enumerate(clusters):
        rank = i + 1
        
        # Determine style based on score
        if c['combined_mean'] >= 0.8:
            style = 'excellent'
            rating = 'Excellent'
        elif c['combined_mean'] >= 0.6:
            style = 'high'
            rating = 'High'
        else:
            style = 'moderate'
            rating = 'Moderate'
        
        # Create coordinate string for polygon
        coord_str = ' '.join([f"{lon:.6f},{lat:.6f},0" for lon, lat in c['polygon_coords']])
        
        kml += f'''
    <Placemark>
      <name>Zone {rank}: {rating} Suitability ({c['combined_mean']:.2f})</name>
      <styleUrl>#{style}</styleUrl>
      <description><![CDATA[
<h3>🏆 Rank #{rank} Best Location</h3>
<table border="1" cellpadding="5">
<tr><td><b>Combined Score</b></td><td>{c['combined_mean']:.3f} (max: {c['combined_max']:.3f})</td></tr>
<tr><td><b>🐝 Bee Suitability</b></td><td>{c['bee_mean']:.3f} (max: {c['bee_max']:.3f})</td></tr>
<tr><td><b>🍊 Mandarin Suitability</b></td><td>{c['mandarin_mean']:.3f} (max: {c['mandarin_max']:.3f})</td></tr>
<tr><td><b>Area</b></td><td>~{c['area_hectares']:.1f} hectares</td></tr>
<tr><td><b>Center</b></td><td>{c['center_lat']:.5f}°N, {c['center_lon']:.5f}°E</td></tr>
</table>
<p><b>Classification:</b> {rating}</p>
      ]]></description>
      <Polygon>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>{coord_str}</coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>
'''
    
    kml += '''
  </Document>
</kml>
'''
    
    return kml


if __name__ == "__main__":
    import os
    
    base_dir = "suitability_maps"
    
    generate_top10_polygons(
        combined_tif=os.path.join(base_dir, "combined_suitability.tif"),
        bee_tif=os.path.join(base_dir, "bee_suitability.tif"),
        mandarin_tif=os.path.join(base_dir, "mandarin_suitability.tif"),
        output_kml=os.path.join(base_dir, "top10_best_zones.kml"),
        output_json=os.path.join(base_dir, "top10_best_zones.json"),
        top_n=10,
        min_cluster_pixels=3
    )
