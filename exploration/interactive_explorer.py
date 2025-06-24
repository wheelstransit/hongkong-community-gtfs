#!/usr/bin/env python3
"""
🚌 Hong Kong Transit Data Interactive Explorer 🚌

An interactive Python script to explore the rich Hong Kong transit dataset
with visualizations, maps, and analysis tools.

Usage:
    python interactive_explorer.py

Requirements:
    pip install pandas geopandas folium plotly sqlalchemy psycopg2-binary seaborn matplotlib jupyter

Author: Transit Data Explorer
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
import json
import os
from typing import Dict, List, Tuple, Optional
import sqlite3
from sqlalchemy import create_engine
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class HKTransitExplorer:
    """Interactive Hong Kong Transit Data Explorer"""

    def __init__(self, db_connection_string: str = None):
        """
        Initialize the explorer with database connection

        Args:
            db_connection_string: SQLAlchemy connection string
                                 If None, will try to use default PostgreSQL connection
        """
        self.engine = None
        self.data_cache = {}

        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # Hong Kong bounds for mapping
        self.hk_bounds = {
            'lat_min': 22.15, 'lat_max': 22.58,
            'lon_min': 113.83, 'lon_max': 114.41
        }

        self.hk_center = [22.3193, 114.1694]  # Central Hong Kong

        if db_connection_string:
            try:
                self.engine = create_engine(db_connection_string)
                print("✅ Database connection established!")
            except Exception as e:
                print(f"❌ Database connection failed: {e}")
                print("💡 Continuing without database - using sample data mode")

    def get_data(self, query: str, cache_key: str = None) -> pd.DataFrame:
        """Execute SQL query and return DataFrame with caching"""
        if cache_key and cache_key in self.data_cache:
            return self.data_cache[cache_key]

        if not self.engine:
            print("❌ No database connection available")
            return pd.DataFrame()

        try:
            df = pd.read_sql(query, self.engine)
            if cache_key:
                self.data_cache[cache_key] = df
            return df
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return pd.DataFrame()

    def quick_stats(self) -> None:
        """Display quick statistics about the transit system"""
        print("🚌 " + "="*60)
        print("🇭🇰 HONG KONG TRANSIT SYSTEM OVERVIEW")
        print("="*64)

        stats_query = """
        SELECT
            'KMB Routes' as metric, COUNT(*) as count FROM kmb_routes
        UNION ALL
        SELECT 'KMB Stops', COUNT(*) FROM kmb_stops
        UNION ALL
        SELECT 'Citybus Routes', COUNT(*) FROM citybus_routes
        UNION ALL
        SELECT 'Citybus Stops', COUNT(*) FROM citybus_stops
        UNION ALL
        SELECT 'Journey Time Records', COUNT(*) FROM journey_time_data
        ORDER BY count DESC
        """

        stats_df = self.get_data(stats_query, 'quick_stats')

        if not stats_df.empty:
            for _, row in stats_df.iterrows():
                print(f"📊 {row['metric']:<25}: {row['count']:>8,}")
        else:
            print("📊 Sample data mode - database stats unavailable")

        print("="*64)

    def explore_routes(self, operator: str = 'kmb', limit: int = 10) -> pd.DataFrame:
        """Explore routes for a specific operator"""
        print(f"🔍 Exploring {operator.upper()} Routes...")

        if operator.lower() == 'kmb':
            query = f"""
            SELECT
                route,
                orig_en as origin,
                dest_en as destination,
                service_type,
                bound
            FROM kmb_routes
            WHERE orig_en IS NOT NULL AND dest_en IS NOT NULL
            ORDER BY route
            LIMIT {limit}
            """
        else:
            print(f"❌ Operator {operator} not implemented yet")
            return pd.DataFrame()

        routes_df = self.get_data(query, f'{operator}_routes')

        if not routes_df.empty:
            print(f"📋 Sample {operator.upper()} routes:")
            print(routes_df.to_string(index=False))

        return routes_df

    def create_stop_map(self, operator: str = 'kmb', sample_size: int = 1000) -> folium.Map:
        """Create an interactive map of bus stops"""
        print(f"🗺️  Creating interactive map for {operator.upper()} stops...")

        if operator.lower() == 'kmb':
            query = f"""
            SELECT
                stop,
                name_en,
                ST_Y(geometry) as latitude,
                ST_X(geometry) as longitude
            FROM kmb_stops
            WHERE name_en IS NOT NULL
            ORDER BY RANDOM()
            LIMIT {sample_size}
            """
        else:
            print(f"❌ Operator {operator} not implemented yet")
            return None

        stops_df = self.get_data(query, f'{operator}_stops_map')

        if stops_df.empty:
            print("❌ No stop data available")
            return None

        # Create base map
        m = folium.Map(
            location=self.hk_center,
            zoom_start=11,
            tiles='OpenStreetMap'
        )

        # Add stops as markers
        for _, stop in stops_df.iterrows():
            folium.CircleMarker(
                location=[stop['latitude'], stop['longitude']],
                radius=3,
                popup=f"🚏 {stop['name_en']}<br>Stop ID: {stop['stop']}",
                color='red',
                fillColor='red',
                fillOpacity=0.6
            ).add_to(m)

        # Add title
        title_html = f'''
        <h3 align="center" style="font-size:16px"><b>{operator.upper()} Bus Stops in Hong Kong</b></h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))

        return m

    def analyze_journey_times(self) -> None:
        """Analyze journey time patterns"""
        print("⏱️  Analyzing Journey Time Patterns...")

        # Get journey time distribution
        query = """
        SELECT
            travel_time_seconds,
            travel_time_seconds / 60.0 as travel_time_minutes
        FROM journey_time_data
        WHERE travel_time_seconds > 0 AND travel_time_seconds < 3600
        ORDER BY travel_time_seconds
        """

        journey_df = self.get_data(query, 'journey_times')

        if journey_df.empty:
            print("❌ No journey time data available")
            return

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🚌 Hong Kong Bus Journey Time Analysis', fontsize=16, fontweight='bold')

        # Histogram of journey times
        axes[0,0].hist(journey_df['travel_time_minutes'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_xlabel('Travel Time (minutes)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Distribution of Journey Times')
        axes[0,0].grid(True, alpha=0.3)

        # Box plot
        axes[0,1].boxplot(journey_df['travel_time_minutes'])
        axes[0,1].set_ylabel('Travel Time (minutes)')
        axes[0,1].set_title('Journey Time Box Plot')
        axes[0,1].grid(True, alpha=0.3)

        # Cumulative distribution
        sorted_times = np.sort(journey_df['travel_time_minutes'])
        y = np.arange(1, len(sorted_times) + 1) / len(sorted_times) * 100
        axes[1,0].plot(sorted_times, y, linewidth=2, color='orange')
        axes[1,0].set_xlabel('Travel Time (minutes)')
        axes[1,0].set_ylabel('Cumulative Percentage')
        axes[1,0].set_title('Cumulative Distribution')
        axes[1,0].grid(True, alpha=0.3)

        # Statistics text
        stats_text = f"""
        Journey Time Statistics:

        📊 Total Records: {len(journey_df):,}
        ⏱️  Mean: {journey_df['travel_time_minutes'].mean():.1f} min
        📈 Median: {journey_df['travel_time_minutes'].median():.1f} min
        📉 Min: {journey_df['travel_time_minutes'].min():.1f} min
        📊 Max: {journey_df['travel_time_minutes'].max():.1f} min
        🎯 95th Percentile: {journey_df['travel_time_minutes'].quantile(0.95):.1f} min
        """

        axes[1,1].text(0.1, 0.9, stats_text, transform=axes[1,1].transAxes,
                      fontsize=10, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1,1].set_xlim(0, 1)
        axes[1,1].set_ylim(0, 1)
        axes[1,1].axis('off')

        plt.tight_layout()
        plt.show()

        # Print summary
        print(f"📊 Analyzed {len(journey_df):,} journey time records")
        print(f"⏱️  Average journey time: {journey_df['travel_time_minutes'].mean():.1f} minutes")
        print(f"🚀 Fastest journey: {journey_df['travel_time_minutes'].min():.1f} minutes")
        print(f"🐌 Slowest journey: {journey_df['travel_time_minutes'].max():.1f} minutes")

    def hourly_patterns(self) -> None:
        """Analyze hourly travel patterns"""
        print("🕐 Analyzing Hourly Travel Patterns...")

        query = """
        SELECT
            hour,
            AVG(travel_time_seconds) as avg_travel_time,
            COUNT(*) as journey_count
        FROM hourly_journey_time_data
        WHERE travel_time_seconds > 0
        GROUP BY hour
        ORDER BY hour
        """

        hourly_df = self.get_data(query, 'hourly_patterns')

        if hourly_df.empty:
            print("❌ No hourly data available")
            return

        # Create hourly pattern visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('🕐 Hong Kong Bus Travel Patterns by Hour', fontsize=16, fontweight='bold')

        # Average travel time by hour
        bars1 = ax1.bar(hourly_df['hour'], hourly_df['avg_travel_time'],
                       color='lightcoral', alpha=0.7, edgecolor='darkred')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Average Travel Time (seconds)')
        ax1.set_title('Average Travel Time by Hour')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(0, 24, 2))

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.0f}s', ha='center', va='bottom', fontsize=8)

        # Journey count by hour
        bars2 = ax2.bar(hourly_df['hour'], hourly_df['journey_count'],
                       color='lightblue', alpha=0.7, edgecolor='darkblue')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Number of Journeys')
        ax2.set_title('Journey Volume by Hour')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(0, 24, 2))

        plt.tight_layout()
        plt.show()

        # Find peak hours
        peak_travel_time = hourly_df.loc[hourly_df['avg_travel_time'].idxmax()]
        peak_volume = hourly_df.loc[hourly_df['journey_count'].idxmax()]

        print(f"🚦 Peak travel time: Hour {peak_travel_time['hour']} ({peak_travel_time['avg_travel_time']:.0f}s avg)")
        print(f"🚌 Peak volume: Hour {peak_volume['hour']} ({peak_volume['journey_count']:,} journeys)")

    def find_interesting_routes(self) -> None:
        """Find interesting and unusual routes"""
        print("🎭 Finding Interesting Routes...")

        queries = {
            "🏆 Longest Route Names": """
                SELECT route, orig_en, dest_en,
                       LENGTH(orig_en || ' to ' || dest_en) as name_length
                FROM kmb_routes
                WHERE orig_en IS NOT NULL AND dest_en IS NOT NULL
                ORDER BY name_length DESC
                LIMIT 5
            """,

            "🔄 Circular Routes": """
                SELECT route, orig_en as location, service_type
                FROM kmb_routes
                WHERE orig_en = dest_en AND orig_en IS NOT NULL
                ORDER BY route
                LIMIT 5
            """,

            "✈️ Airport Routes": """
                SELECT route, orig_en, dest_en
                FROM kmb_routes
                WHERE (orig_en ILIKE '%airport%' OR dest_en ILIKE '%airport%')
                   AND orig_en IS NOT NULL AND dest_en IS NOT NULL
                ORDER BY route
                LIMIT 5
            """
        }

        for title, query in queries.items():
            print(f"\n{title}")
            print("-" * 50)
            df = self.get_data(query)
            if not df.empty:
                print(df.to_string(index=False))
            else:
                print("No data available")

    def route_network_analysis(self) -> None:
        """Analyze the route network connectivity"""
        print("🕸️  Analyzing Route Network...")

        # Get route-stop connections
        query = """
        SELECT route, stop, seq
        FROM kmb_stop_sequences
        ORDER BY route, seq
        LIMIT 10000
        """

        network_df = self.get_data(query, 'network_analysis')

        if network_df.empty:
            print("❌ No network data available")
            return

        # Calculate network metrics
        route_stats = network_df.groupby('route').agg({
            'stop': 'count',
            'seq': 'max'
        }).rename(columns={'stop': 'stop_count', 'seq': 'max_sequence'})

        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('🕸️ Route Network Analysis', fontsize=16, fontweight='bold')

        # Distribution of stops per route
        ax1.hist(route_stats['stop_count'], bins=20, alpha=0.7, color='green', edgecolor='darkgreen')
        ax1.set_xlabel('Number of Stops per Route')
        ax1.set_ylabel('Number of Routes')
        ax1.set_title('Distribution of Route Lengths')
        ax1.grid(True, alpha=0.3)

        # Top 10 longest routes
        top_routes = route_stats.nlargest(10, 'stop_count')
        bars = ax2.barh(range(len(top_routes)), top_routes['stop_count'], color='orange', alpha=0.7)
        ax2.set_yticks(range(len(top_routes)))
        ax2.set_yticklabels(top_routes.index)
        ax2.set_xlabel('Number of Stops')
        ax2.set_title('Top 10 Longest Routes')
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{width}', ha='left', va='center', fontsize=9)

        plt.tight_layout()
        plt.show()

        print(f"📊 Network Statistics:")
        print(f"🚌 Total routes analyzed: {len(route_stats)}")
        print(f"🚏 Average stops per route: {route_stats['stop_count'].mean():.1f}")
        print(f"📏 Longest route: {route_stats['stop_count'].max()} stops")
        print(f"📏 Shortest route: {route_stats['stop_count'].min()} stops")

    def interactive_menu(self) -> None:
        """Display interactive menu for exploration"""
        while True:
            print("\n" + "="*60)
            print("🚌 HONG KONG TRANSIT DATA EXPLORER")
            print("="*60)
            print("1. 📊 Quick Statistics Overview")
            print("2. 🔍 Explore Routes (KMB)")
            print("3. 🗺️  Create Interactive Stop Map")
            print("4. ⏱️  Analyze Journey Times")
            print("5. 🕐 Hourly Travel Patterns")
            print("6. 🎭 Find Interesting Routes")
            print("7. 🕸️  Route Network Analysis")
            print("8. 💾 Export Data to CSV")
            print("9. 🎨 Create Custom Visualization")
            print("0. 🚪 Exit")
            print("="*60)

            try:
                choice = input("Enter your choice (0-9): ").strip()

                if choice == '1':
                    self.quick_stats()
                elif choice == '2':
                    limit = int(input("How many routes to show? (default 10): ") or "10")
                    self.explore_routes(limit=limit)
                elif choice == '3':
                    sample_size = int(input("Sample size for map (default 1000): ") or "1000")
                    map_obj = self.create_stop_map(sample_size=sample_size)
                    if map_obj:
                        filename = f"hk_stops_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                        map_obj.save(filename)
                        print(f"📁 Map saved as: {filename}")
                elif choice == '4':
                    self.analyze_journey_times()
                elif choice == '5':
                    self.hourly_patterns()
                elif choice == '6':
                    self.find_interesting_routes()
                elif choice == '7':
                    self.route_network_analysis()
                elif choice == '8':
                    self.export_data()
                elif choice == '9':
                    self.custom_visualization()
                elif choice == '0':
                    print("👋 Thanks for exploring Hong Kong transit data!")
                    break
                else:
                    print("❌ Invalid choice. Please try again.")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try again.")

    def export_data(self) -> None:
        """Export data to CSV files"""
        print("💾 Exporting Data...")

        exports = {
            "kmb_routes.csv": "SELECT * FROM kmb_routes LIMIT 1000",
            "kmb_stops.csv": "SELECT stop, name_en, ST_Y(geometry) as lat, ST_X(geometry) as lon FROM kmb_stops LIMIT 1000",
            "journey_times.csv": "SELECT * FROM journey_time_data LIMIT 1000"
        }

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        for filename, query in exports.items():
            df = self.get_data(query)
            if not df.empty:
                export_filename = f"{timestamp}_{filename}"
                df.to_csv(export_filename, index=False)
                print(f"📁 Exported: {export_filename} ({len(df)} rows)")
            else:
                print(f"❌ No data for {filename}")

    def custom_visualization(self) -> None:
        """Create custom visualizations based on user input"""
        print("🎨 Custom Visualization Creator")
        print("Available tables: kmb_routes, kmb_stops, journey_time_data, hourly_journey_time_data")

        query = input("Enter your SQL query: ").strip()

        if not query:
            print("❌ No query provided")
            return

        try:
            df = self.get_data(query)
            if df.empty:
                print("❌ Query returned no data")
                return

            print(f"✅ Query executed successfully! ({len(df)} rows)")
            print("\nFirst 5 rows:")
            print(df.head().to_string())

            # Offer visualization options
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if numeric_cols:
                print(f"\nNumeric columns available: {', '.join(numeric_cols)}")
                viz_type = input("Visualization type (histogram/scatter/bar): ").lower()

                if viz_type == 'histogram' and len(numeric_cols) >= 1:
                    col = input(f"Column for histogram ({numeric_cols[0]}): ") or numeric_cols[0]
                    if col in df.columns:
                        plt.figure(figsize=(10, 6))
                        plt.hist(df[col], bins=30, alpha=0.7)
                        plt.title(f'Distribution of {col}')
                        plt.xlabel(col)
                        plt.ylabel('Frequency')
                        plt.grid(True, alpha=0.3)
                        plt.show()

        except Exception as e:
            print(f"❌ Error executing query: {e}")


def main():
    """Main function to run the explorer"""
    print("🚀 Starting Hong Kong Transit Data Explorer...")

    # Try to connect to database
    # You can modify this connection string based on your setup
    db_connection = None

    # Common connection string patterns:
    # PostgreSQL: "postgresql://username:password@localhost:5432/database_name"
    # SQLite: "sqlite:///path/to/database.db"

    try:
        # Try to get connection from environment or use default
        import os
        db_connection = os.getenv('DATABASE_URL', 'postgresql://localhost:5432/hongkong_transit')
    except:
        pass

    # Initialize explorer
    explorer = HKTransitExplorer(db_connection)

    # Check if we have a database connection
    if not explorer.engine:
        print("⚠️  No database connection - running in demo mode")
        print("💡 To connect to your database, set the DATABASE_URL environment variable")
        print("   Example: export DATABASE_URL='postgresql://user:pass@localhost:5432/dbname'")
        print()

    # Start interactive exploration
    explorer.interactive_menu()


if __name__ == "__main__":
    main()
