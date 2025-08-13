#!/usr/bin/env python3
"""
Quick test of our CTB export integration to verify 22M circular route fix
"""

import sys
sys.path.insert(0, '/app/src')

from src.common.database import get_db_engine
from src.processing.gtfs_route_matcher import match_operator_routes_with_coop_fallback
import pandas as pd

def test_22m_fix():
    print("🧪 Testing 22M circular route fix...")
    
    engine = get_db_engine()
    
    # Test enhanced CTB matching for 22M
    print("1. Running enhanced CTB matching for 22M...")
    matches = match_operator_routes_with_coop_fallback(
        engine=engine,
        operator_name='CTB',
        route_filter=['22M'],
        debug=False
    )
    
    print(f"2. Found {len(matches)} route matches:")
    for route_key, services in matches.items():
        print(f"   {route_key}: {len(services)} services")
    
    # Check results
    has_22m_outbound = '22M-O-1' in matches
    has_22m_inbound = '22M-I-1' in matches
    
    print("3. Analysis:")
    if has_22m_outbound and not has_22m_inbound:
        print("   ✅ SUCCESS: 22M correctly identified as circular")
        print("   ✅ Only outbound route exists (22M-O-1)")
        print("   ✅ No inbound route created (22M-I-1 missing)")
        return True
    elif has_22m_outbound and has_22m_inbound:
        print("   ❌ ISSUE: Both inbound and outbound routes exist")
        print("   ❌ 22M should be circular (outbound only)")
        return False
    elif not has_22m_outbound:
        print("   ❌ ISSUE: No 22M routes found at all")
        return False
    else:
        print("   ❌ ISSUE: Unexpected state")
        return False

if __name__ == "__main__":
    success = test_22m_fix()
    print("\n🎯 Overall Result:")
    if success:
        print("✅ Circular route fix is working correctly!")
        print("🚀 Ready to run full export!")
    else:
        print("❌ Circular route fix needs more work")
        print("🔧 Need to debug further")
