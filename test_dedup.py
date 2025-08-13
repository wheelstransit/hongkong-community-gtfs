#!/usr/bin/env python3

import os
import pandas as pd
from src.common.database import get_db_engine
from src.export.export_gtfs import export_unified_feed

def test_kmb_trip_deduplication():
    """Test if KMB route 1 trips are properly deduplicated"""
    engine = get_db_engine()
    
    # Run just a quick export
    export_unified_feed(
        engine=engine,
        output_dir="test_output",
        journey_time_data=None,
        mtr_headway_data=None,
        osm_data=None,
        silent=False
    )
    
    # Check the trips.txt file
    trips_file = "test_output/unified_feed/trips.txt"
    if os.path.exists(trips_file):
        trips_df = pd.read_csv(trips_file)
        
        # Filter for KMB route 1
        kmb_1_trips = trips_df[trips_df['route_id'] == 'KMB-1']
        
        print(f"\nKMB Route 1 Analysis:")
        print(f"Total KMB-1 trips: {len(kmb_1_trips)}")
        
        # Check for duplicates
        duplicated_trip_ids = kmb_1_trips[kmb_1_trips.duplicated(subset=['trip_id'], keep=False)]
        print(f"Duplicated trip_ids: {len(duplicated_trip_ids)}")
        
        if len(duplicated_trip_ids) > 0:
            print("\nDuplicate trip_ids found:")
            print(duplicated_trip_ids[['trip_id', 'service_id', 'direction_id', 'shape_id']].head(20))
        else:
            print("✅ No duplicate trip_ids found!")
            
        # Show unique trips count
        unique_trips = kmb_1_trips.drop_duplicates(subset=['trip_id'])
        print(f"Unique trip_ids: {len(unique_trips)}")
        
        # Show sample trips
        print("\nSample KMB-1 trips:")
        print(kmb_1_trips[['trip_id', 'service_id', 'direction_id', 'shape_id']].head(10))

if __name__ == "__main__":
    test_kmb_trip_deduplication()
