#!/usr/bin/env python3
"""
Complete Ecological Planning Pipeline Runner
Handles DEM conversion from HGT format and executes full analysis
"""

import os
import sys
from pathlib import Path


def main():
    print("\n" + "=" * 80)
    print("ECOLOGICAL GEOSPATIAL PLANNING ENGINE - SETUP & EXECUTION")
    print("=" * 80)

    # Step 1: Check and convert DEM if needed
    print("\n[STEP 1] Checking DEM file...")
    
    dem_hgt = Path("USGS/n28_e081_1arc_v3.dt2")
    dem_tif = Path("USGS/n28_e081_1arc_v3.tif")
    
    if not dem_hgt.exists() and not dem_tif.exists():
        print("❌ ERROR: DEM file not found!")
        print(f"   Expected: {dem_hgt} or {dem_tif}")
        sys.exit(1)
    
    if dem_hgt.exists() and not dem_tif.exists():
        print(f"  Converting HGT to GeoTIFF...")
        try:
            from hgt_loader import load_hgt_dem
            load_hgt_dem(str(dem_hgt), str(dem_tif))
            print(f"  ✓ DEM converted: {dem_tif}")
        except Exception as e:
            print(f"❌ ERROR converting DEM: {e}")
            sys.exit(1)
    else:
        print(f"  ✓ Using existing DEM: {dem_tif}")
    
    # Step 2: Check required files
    print("\n[STEP 2] Checking required input files...")
    
    required_files = {
        "config.yaml": "Configuration file",
        "InitialFarm.kml": "Area of Interest boundary",
    }
    
    for file, desc in required_files.items():
        if not Path(file).exists():
            print(f"❌ ERROR: Missing {desc}: {file}")
            sys.exit(1)
        else:
            print(f"  ✓ {desc}: {file}")
    
    # Step 3: Check KML parser
    print("\n[STEP 3] Validating KML...")
    try:
        from kml_parser import KMLParser
        parser = KMLParser("InitialFarm.kml")
        farm_data = parser.parse()
        print(f"  ✓ Farm: {farm_data['name']}")
        print(f"  ✓ Area: {farm_data['area_hectares']:.2f} hectares")
    except Exception as e:
        print(f"❌ ERROR parsing KML: {e}")
        sys.exit(1)
    
    # Step 4: Run main analysis
    print("\n[STEP 4] Executing ecological suitability analysis...")
    print("  This may take 5-15 minutes depending on data availability...\n")
    
    try:
        from main_new import EcologicalPlanningEngine
        
        engine = EcologicalPlanningEngine("config.yaml")
        engine.run_complete_analysis()
        
        print("\n✓ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("\nOutput files generated:")
        
        output_dir = Path("suitability_maps")
        if output_dir.exists():
            for file in sorted(output_dir.glob("*.tif")):
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  • {file.name} ({size_mb:.1f} MB)")
            
            for file in sorted(output_dir.glob("*.json")):
                print(f"  • {file.name}")
            
            for file in sorted(output_dir.glob("*.kml")):
                print(f"  • {file.name}")
        
        print(f"\nAll outputs saved to: {output_dir}/")
        
    except Exception as e:
        print(f"\n❌ ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
