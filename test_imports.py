#!/usr/bin/env python3
"""
Quick test to verify our enhanced matcher import works
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, '/app/src')

try:
    print("Testing imports...")
    from src.processing.gtfs_route_matcher import match_operator_routes_with_coop_fallback
    print("✅ Successfully imported match_operator_routes_with_coop_fallback")
    
    from src.common.database import get_db_engine
    print("✅ Successfully imported get_db_engine")
    
    print("✅ All imports working correctly!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
    
except Exception as e:
    print(f"❌ Other error: {e}")
    import traceback
    traceback.print_exc()
