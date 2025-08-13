#!/usr/bin/env python3
"""
Quick test to verify the syntax error is fixed
"""

try:
    print("Testing import of gtfs_route_matcher...")
    from src.processing.gtfs_route_matcher import match_operator_routes_with_coop_fallback
    print("✅ Import successful - syntax error fixed!")
    
except SyntaxError as e:
    print(f"❌ Syntax error still exists: {e}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    
except Exception as e:
    print(f"❌ Other error: {e}")
