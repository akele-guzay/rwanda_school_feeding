# ==============================================================================
# RWANDA SCHOOL FEEDING GIS ANALYSIS & REPORT GENERATOR
# Output: Summary statistics, Plotly Interactive Visualizations, Folium Maps
# ==============================================================================

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import FloatImage, Fullscreen, MiniMap, Search
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import base64

# Plotly Imports for Interactive Reporting
import plotly.express as px
import plotly.graph_objects as go

# Base Output Folder
OUTPUT_DIR = "./docs/output"
PROFILES_DIR = os.path.join(OUTPUT_DIR, "profiles")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROFILES_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 1. LOAD GEOPACKAGE LAYERS & ENRICH WITH ADMIN BOUNDARIES
# ------------------------------------------------------------------------------
GPKG_PATH = "./data/results_90K_40mins.gpkg"
VILLAGES_GPKG_PATH = "./data/villages.gpkg"  # Path to administrative boundary GPKG
DISTRICTS_GPKG_PATH = "./data/districts.gpkg" # Path to district-level base map GPKG

try:
    gdf_kitchens = gpd.read_file(GPKG_PATH, layer="kitchens_40")
    gdf_schools = gpd.read_file(GPKG_PATH, layer="schools_40")
    gdf_corridors = gpd.read_file(GPKG_PATH, layer="corridors_40")
    gdf_routes = gpd.read_file(GPKG_PATH, layer="routes_40")
    print("Primary GeoPackage layers loaded successfully.")

    # Load District Level GeoPackage
    if os.path.exists(DISTRICTS_GPKG_PATH):
        gdf_districts = gpd.read_file(DISTRICTS_GPKG_PATH)
        if gdf_districts.crs != gdf_kitchens.crs:
            gdf_districts = gdf_districts.to_crs(gdf_kitchens.crs)
        print("District base map loaded successfully.")
    else:
        gdf_districts = None
        print(f"Warning: {DISTRICTS_GPKG_PATH} not found. National map will proceed without district boundaries.")

    # Load Administrative Village Boundaries
    if os.path.exists(VILLAGES_GPKG_PATH):
        gdf_admin = gpd.read_file(VILLAGES_GPKG_PATH)

        # 1. Align Coordinate Reference Systems (CRS)
        if gdf_admin.crs != gdf_kitchens.crs:
            gdf_admin = gdf_admin.to_crs(gdf_kitchens.crs)

        # 2. Select relevant administrative columns + geometry
        admin_cols = ['province', 'district', 'sector', 'cell', 'village', 'geometry']
        gdf_admin_sub = gdf_admin[[c for c in admin_cols if c in gdf_admin.columns]]

        # 3. Perform Spatial Join (intersects) for BOTH Kitchens and Schools
        gdf_kitchens = gpd.sjoin(
            gdf_kitchens, 
            gdf_admin_sub, 
            how='left', 
            predicate='intersects'
        ).drop(columns=['index_right'], errors='ignore')

        gdf_schools = gpd.sjoin(
            gdf_schools, 
            gdf_admin_sub, 
            how='left', 
            predicate='intersects'
        ).drop(columns=['index_right'], errors='ignore')

        # Fill remaining unmatched kitchen points using nearest boundary
        unmatched_k = gdf_kitchens['district'].isna() if 'district' in gdf_kitchens.columns else False
        if unmatched_k.any():
            nearest_k = gpd.sjoin_nearest(
                gdf_kitchens[unmatched_k].drop(columns=['province', 'district', 'sector', 'cell', 'village'], errors='ignore'),
                gdf_admin_sub,
                how='left'
            ).drop(columns=['index_right', 'distance'], errors='ignore')
            gdf_kitchens.update(nearest_k)

        # Fill remaining unmatched school points using nearest boundary
        unmatched_s = gdf_schools['district'].isna() if 'district' in gdf_schools.columns else False
        if unmatched_s.any():
            nearest_s = gpd.sjoin_nearest(
                gdf_schools[unmatched_s].drop(columns=['province', 'district', 'sector', 'cell', 'village'], errors='ignore'),
                gdf_admin_sub,
                how='left'
            ).drop(columns=['index_right', 'distance'], errors='ignore')
            gdf_schools.update(nearest_s)

    # 4. Create readable Admin Display Name / Codename for Kitchens
    k_id_col = 'kitchen_node' if 'kitchen_node' in gdf_kitchens.columns else 'kitchen_id'
    
    gdf_kitchens['codename'] = gdf_kitchens.apply(
        lambda r: f"Kitchen {r.get(k_id_col, r.name)} - {r.get('district', 'N/A')} ({r.get('sector', 'N/A')} Sector)", 
        axis=1
    )

    print("Kitchens and Schools successfully enriched with official Rwanda administrative names.")

except Exception as e:
    print(f"Error during layer loading or administrative spatial join: {e}")

# ------------------------------------------------------------------------------
# 2. DATA PROCESSING & SUMMARY TABLES
# ------------------------------------------------------------------------------

def generate_summary_tables(gdf_k, gdf_s, gdf_c):
    """Generates clean summary dataframes and corridor metrics for report inclusion."""
    for gdf in [gdf_k, gdf_s, gdf_c]:
        if 'fid' not in gdf.columns:
            gdf['fid'] = gdf.index

    # Kitchen Typology & Size Summary
    kitchen_summary = gdf_k.groupby(['ctype', 'size_band']).agg(
        kitchen_count=('fid', 'count'),
        total_schools_served=('n_schools', 'sum'),
        total_meals_day=('meals_day', 'sum'),
        avg_drive_time=('mean_time_min', 'mean'),
        max_drive_time=('max_time_min', 'max')
    ).reset_index()
    
    # Overall Network KPI Summary
    kpi_summary = pd.DataFrame([{
        "Total Central Kitchens": len(gdf_k),
        "Independent Kitchens": len(gdf_k[gdf_k['ctype'] != 'school']),
        "School-Based Kitchens": len(gdf_k[gdf_k['ctype'] == 'school']),
        "Total Schools Served": len(gdf_s),
        "Total Student Meals/Day": gdf_s['demand'].sum(),
        "Mean Drive Time (mins)": gdf_s['time_min'].mean(),
        "95th Percentile Drive Time (mins)": gdf_s['time_min'].quantile(0.95),
        "Max Drive Time (mins)": gdf_s['time_min'].max()
    }])
    
    # Corridor Summary & Verifiable Logistics Statistics
    valid_corridors = gdf_c[gdf_c['corridor'] != -1]
    
    # Compute Kitchen Corridor Count Breakdown
    k_node_col = 'kitchen_node' if 'kitchen_node' in valid_corridors.columns else 'kitchen_id'
    corridors_per_kitchen = valid_corridors.groupby(k_node_col)['corridor'].nunique()
    k_3_5_cnt = ((corridors_per_kitchen >= 3) & (corridors_per_kitchen <= 5)).sum()
    k_3_5_pct = (k_3_5_cnt / len(gdf_k)) * 100

    # Dispatch Route Consolidation Efficiency
    total_schools_cnt = len(gdf_s)
    total_corridor_routes_cnt = len(valid_corridors[[k_node_col, 'corridor']].drop_duplicates())
    route_reduction_pct = ((total_schools_cnt - total_corridor_routes_cnt) / total_schools_cnt) * 100

    print(f"\n--- CORRIDOR NETWORK LOGISTICS METRICS ---")
    print(f"Kitchens with 3-5 Corridors: {k_3_5_pct:.1f}% ({k_3_5_cnt}/{len(gdf_k)})")
    print(f"Route Consolidation Reduction: {route_reduction_pct:.1f}% ({total_schools_cnt} schools -> {total_corridor_routes_cnt} corridors)")

    corridor_summary = valid_corridors.groupby('corridor').agg(
        schools_in_corridor=('school_id', 'count') if 'school_id' in valid_corridors.columns else ('demand', 'count'),
        total_demand=('demand', 'sum'),
        mean_bearing=('bearing_deg', 'mean') if 'bearing_deg' in valid_corridors.columns else ('corridor', lambda x: 0.0)
    ).reset_index()
    
    return kitchen_summary, kpi_summary, corridor_summary
# ------------------------------------------------------------------------------
# 3. INTERACTIVE VISUALIZATIONS (PLOTLY)
# ------------------------------------------------------------------------------

def plot_drive_time_distribution(gdf_schools, output_dir=OUTPUT_DIR):
    """Generates an interactive Plotly histogram of school drive times."""
    html_out = os.path.join(output_dir, "drive_time_dist.html")
    png_out = os.path.join(output_dir, "drive_time_dist.png")
    
    mean_val = gdf_schools['time_min'].mean()

    fig = px.histogram(
        gdf_schools,
        x="time_min",
        nbins=20,
        title="<b>School Drive Time Distribution to Assigned Central Kitchen</b>",
        labels={"time_min": "Drive Time (Minutes)"},
        color_discrete_sequence=["#2b5c8f"],
    )

    fig.update_traces(
        marker_line_color="white",
        marker_line_width=1,
        hovertemplate="Drive Time: %{x:.1f} min<br>Count: %{y} schools<extra></extra>",
    )

    # Threshold & Mean Lines
    fig.add_vline(
        x=40, line_width=2, line_dash="dash", line_color="#d95f02",
        annotation_text="40-Min Threshold", annotation_position="top right"
    )
    fig.add_vline(
        x=mean_val, line_width=2, line_dash="dot", line_color="#7570b3",
        annotation_text=f"Mean ({mean_val:.1f} min)", annotation_position="top left"
    )

    fig.update_layout(
        yaxis_title="Number of Schools",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="serif", size=12, color="#2c3e50"),
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
        margin=dict(l=40, r=40, t=50, b=40),
    )

    fig.write_html(html_out, include_plotlyjs="cdn")
    try:
        fig.write_image(png_out, scale=3)
    except Exception as e:
        print(f"PNG Export warning (requires kaleido): {e}")
        
    print(f"Saved Interactive Plotly Chart: {html_out}")


def plot_kitchen_meals_capacity(gdf_kitchens, output_dir=OUTPUT_DIR):
    """Generates an interactive Plotly scatter plot of kitchen capacities with median line."""
    html_out = os.path.join(output_dir, "kitchen_capacity.html")
    png_out = os.path.join(output_dir, "kitchen_capacity.png")

    median_meals = gdf_kitchens['meals_day'].median()

    fig = px.scatter(
        gdf_kitchens,
        x="n_schools",
        y="meals_day",
        color="size_band",
        symbol="ctype",
        title="<b>Kitchen Capacity: Daily Meals vs. Number of Schools Served</b>",
        labels={
            "n_schools": "Number of Schools Served",
            "meals_day": "Meals Per Day",
            "size_band": "Size Band",
            "ctype": "Kitchen Typology"
        },
        hover_data=["codename", "district"],
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="DarkSlateGrey")))

    # Replace 70K Cap with Median Line
    fig.add_hline(
        y=median_meals, line_width=2, line_dash="dash", line_color="#10b981",
        annotation_text=f"Median Kitchen Size ({int(median_meals):,} Meals/Day)", 
        annotation_position="bottom right"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="serif", size=12, color="#2c3e50"),
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            x=1.02, y=1, xanchor="left"
        ),
        margin=dict(l=40, r=140, t=50, b=40),
    )

    fig.write_html(html_out, include_plotlyjs="cdn")
    try:
        fig.write_image(png_out, scale=3)
    except Exception as e:
        print(f"PNG Export warning (requires kaleido): {e}")

    print(f"Saved Interactive Plotly Chart with Median Line: {html_out}")

def plot_terrain_penalties(output_dir=OUTPUT_DIR):
    """Generates an interactive Plotly line chart of terrain grade penalties."""
    html_out = os.path.join(output_dir, "terrain_penalties.html")
    png_out = os.path.join(output_dir, "terrain_penalties.png")

    grades = np.linspace(0, 20, 100)
    uphill = np.where(grades <= 5, 1.0, np.maximum(0.35, 1.0 - 0.08 * (grades - 5)))
    downhill = np.where(grades <= 5, 1.0, np.maximum(0.50, 1.0 - 0.05 * (grades - 5)))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=grades, y=uphill, mode="lines", name="Uphill Penalty",
            line=dict(color="#d95f02", width=3),
            hovertemplate="Grade: %{x:.1f}%<br>Speed Multiplier: %{y:.2f}<extra></extra>"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=grades, y=downhill, mode="lines", name="Downhill Penalty",
            line=dict(color="#7570b3", width=3, dash="dash"),
            hovertemplate="Grade: %{x:.1f}%<br>Speed Multiplier: %{y:.2f}<extra></extra>"
        )
    )

    fig.add_vline(
        x=5, line_width=1.5, line_dash="dot", line_color="#999999",
        annotation_text="5% Threshold"
    )
    fig.add_hline(
        y=0.35, line_width=1, line_dash="dot", line_color="#d95f02",
        annotation_text="Uphill Floor (0.35)"
    )
    fig.add_hline(
        y=0.50, line_width=1, line_dash="dot", line_color="#7570b3",
        annotation_text="Downhill Floor (0.50)"
    )

    fig.update_layout(
        title="<b>Terrain Speed Multiplier vs. Road Grade (%)</b>",
        xaxis_title="Road Grade (%)",
        yaxis_title="Speed Multiplier",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="serif", size=12, color="#2c3e50"),
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0", range=[0, 20]),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0", range=[0.2, 1.05]),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            x=1.02, y=1, xanchor="left"
        ),
        margin=dict(l=40, r=120, t=50, b=40),
    )

    fig.write_html(html_out, include_plotlyjs="cdn")
    try:
        fig.write_image(png_out, scale=3)
    except Exception as e:
        print(f"PNG Export warning (requires kaleido): {e}")

    print(f"Saved Interactive Plotly Chart: {html_out}")

# ------------------------------------------------------------------------------
# 4. INTERACTIVE LEAFLET DASHBOARDS (FOLIUM)
# ------------------------------------------------------------------------------

def get_capacity_color(total):
    if total <= 50000:
        return "#fee5d9"
    elif total <= 100000:
        return "#fcae91"
    elif total <= 150000:
        return "#fb6a4a"
    elif total <= 200000:
        return "#de2d26"
    else:
        return "#a50f15"

def build_interactive_leaflet_maps(gdf_k, gdf_s, gdf_r, gdf_c=None, output_dir=OUTPUT_DIR):
    """Generates two separate interactive maps based on GPKG data."""
    os.makedirs(output_dir, exist_ok=True)
    
    center_lat = gdf_s.geometry.y.mean()
    center_lon = gdf_s.geometry.x.mean()
    
    cmap = plt.get_cmap("tab20")
    
    k_id_col = next((c for c in ['kitchen_node', 'kitchen_id', 'fid'] if c in gdf_k.columns), gdf_k.columns[0])
    s_k_col = next((c for c in ['kitchen_node', 'kitchen_id', 'assigned_kitchen'] if c in gdf_s.columns), None)
    r_k_col = next((c for c in ['kitchen_node', 'kitchen_id'] if c in gdf_r.columns), None)

    unique_kitchens = gdf_k[k_id_col].unique()
    color_dict = {
        kid: mcolors.to_hex(cmap(i % 20)) for i, kid in enumerate(unique_kitchens)
    }

    # Dynamic Legend HTML
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; width: 220px; max-height: 350px; 
         background-color: rgba(255, 255, 255, 0.95); z-index:9999; font-size:12px;
         border: 1px solid grey; border-radius: 8px; padding: 10px;
         overflow-y: auto; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
         
         <b style="font-size:13px;">Daily Meals Capacity</b><br>
         <hr style="margin: 4px 0;">
         <div style="margin-bottom:3px;"><i style="background:#fee5d9; width:12px; height:12px; float:left; margin-right:6px; border:1px solid #ccc;"></i>0 - 50,000</div>
         <div style="margin-bottom:3px;"><i style="background:#fcae91; width:12px; height:12px; float:left; margin-right:6px; border:1px solid #ccc;"></i>50,000 - 100,000</div>
         <div style="margin-bottom:3px;"><i style="background:#fb6a4a; width:12px; height:12px; float:left; margin-right:6px; border:1px solid #ccc;"></i>100,000 - 150,000</div>
         <div style="margin-bottom:3px;"><i style="background:#de2d26; width:12px; height:12px; float:left; margin-right:6px; border:1px solid #ccc;"></i>150,000 - 200,000</div>
         <div style="margin-bottom:10px;"><i style="background:#a50f15; width:12px; height:12px; float:left; margin-right:6px; border:1px solid #ccc;"></i>> 200,000</div>

         <b style="font-size:13px;">Network Clusters</b><br>
         <hr style="margin: 4px 0;">
    """
    for kid, color in color_dict.items():
        legend_html += f'<div style="margin-bottom:3px;"><i style="background:{color}; width:12px; height:12px; float:left; margin-right:6px; border-radius:50%; border:1px solid #333;"></i>Kitchen {kid}</div>'
    legend_html += "</div>"

    # Info Title Card HTML
    title_html = """
    <div style="position: absolute; top: 10px; left: 60px; z-index: 9999;">
        <button id="infoBtn" onclick="toggleTitle()" 
                style="width: 35px; height: 35px; border-radius: 50%; background-color: white; 
                       border: 2px solid #333; font-weight: bold; font-size: 18px; 
                       cursor: pointer; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
            i
        </button>
        
        <div id="titleCard" 
             style="display: none; margin-top: 5px; background-color: rgba(255, 255, 255, 0.95); 
                    padding: 15px; border-radius: 5px; border: 1px solid black; 
                    box-shadow: 2px 2px 5px rgba(0,0,0,0.3); max-width: 420px;">
            <h3 style="margin: 0 0 8px 0; font-size: 15px; font-weight: bold; border-bottom: 1px solid #ccc; padding-bottom: 5px;">
                Optimal Kitchen Network (40 min Drive Time Threshold)
            </h3>
            <p style="margin: 0; font-size: 12px; line-height: 1.4; color: #333;">
                Spatial distribution and routing network supporting regional school feeding operations. Interactive layer toggles allow inspection of central kitchens, transport corridors, and supported schools.
            </p>
            <hr style="margin: 10px 0 6px 0; border: 0; border-top: 1px solid #ccc;">
            <p style="margin: 0; font-size: 11px; color: #555; font-style: italic;">
                Prepared for Solid Africa School Feeding Program
            </p>
        </div>
    </div>
    <script>
        function toggleTitle() {
            var card = document.getElementById("titleCard");
            card.style.display = (card.style.display === "none") ? "block" : "none";
        }
    </script>
    """

    # -------------------------------------------------------------------------
    # MAP 1: Primary Macro View
    # -------------------------------------------------------------------------
    m1 = folium.Map(location=[center_lat, center_lon], zoom_start=9, control_scale=True)

    folium.TileLayer("cartodbpositron", name="Clean Street Map", overlay=False).add_to(m1)
    folium.TileLayer("opentopomap", name="Topographic Map", overlay=False).add_to(m1)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Esri Satellite", overlay=False
    ).add_to(m1)

    m1.get_root().html.add_child(folium.Element(legend_html))
    m1.get_root().html.add_child(folium.Element(title_html))

    # Routes Layer
    folium.GeoJson(
        gdf_r,
        name="Delivery Routes",
        style_function=lambda feature: {"color": "#ffffff", "weight": 1.5, "opacity": 0.8},
    ).add_to(m1)

    # Schools Layer
    school_fields = [c for c in ['school_id', 'time_min', 'demand'] if c in gdf_s.columns]
    school_aliases = ['School ID:', 'Drive Time (min):', 'Meals/Day:'][:len(school_fields)]
    
    school_popup = folium.GeoJsonPopup(fields=school_fields, aliases=school_aliases, labels=True)

    schools_layer = folium.GeoJson(
        gdf_s,
        name="Supported Schools",
        marker=folium.CircleMarker(radius=4),
        popup=school_popup,
        style_function=lambda feature: {
            "fillColor": color_dict.get(feature["properties"].get(s_k_col), "#808080"),
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.9,
            "fill": True,
        },
    )
    schools_layer.add_to(m1)

    # Kitchens Marker Group
    kitchens_group = folium.FeatureGroup(name="Central Kitchens")
    for idx, row in gdf_k.iterrows():
        total_served = row.get("meals_day", 0)
        capacity_color = get_capacity_color(total_served)
        kid = row[k_id_col]
        network_color = color_dict.get(kid, "#333333")

        popup_content = f"""
            <div style="min-width: 170px; font-size: 12px;">
                <b style="font-size:13px; border-bottom:1px solid #ccc; display:block; padding-bottom:3px; margin-bottom:5px;">
                    Kitchen Details
                </b>
                <b>Kitchen ID:</b> {kid}<br>
                <b>District:</b> {row.get('district', 'N/A')}<br>
                <b>Sector:</b> {row.get('sector', 'N/A')}<br>
                <b>Type:</b> {row.get('ctype', 'N/A')}<br>
                <b>Daily Meals:</b> {int(total_served):,}<br>
                <b>Schools Served:</b> {row.get('n_schools', 'N/A')}<br>
                <b>Avg. Drive Time:</b> {row.get('mean_time_min', 0):.1f} mins<br>
                <b>Max. Drive Time:</b> {row.get('max_time_min', 0):.1f} mins<br>
            </div>
        """

        icon_html = f"""
            <div style="background-color: {capacity_color}; 
                        border: 3px solid {network_color}; 
                        border-radius: 50%; 
                        width: 26px; height: 26px; 
                        display: flex; align-items: center; justify-content: center;
                        box-shadow: 2px 2px 5px rgba(0,0,0,0.4);">
                <i class="fa fa-utensils" style="color: black; font-size: 11px;"></i>
            </div>
        """

        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            icon=folium.DivIcon(html=icon_html, icon_anchor=(13, 13)),
            popup=folium.Popup(popup_content, max_width=280),
        ).add_to(kitchens_group)

    kitchens_group.add_to(m1)

    if os.path.exists("./assets/compass.png"):
        FloatImage("./assets/compass.png", bottom=10, left=83, width="5%").add_to(m1)
    
    Fullscreen(position="topright", force_separate_button=True).add_to(m1)
    MiniMap(toggle_display=True, position="bottomright", minimized=False).add_to(m1)
    
    if school_fields:
        Search(
            layer=schools_layer,
            geom_type="Point",
            placeholder="Search school...",
            collapsed=False,
            search_label=school_fields[0],
            position="topleft",
        ).add_to(m1)

    folium.LayerControl().add_to(m1)
    map1_path = os.path.join(output_dir, "40min_interactive_map_schools_kitchens.html")
    m1.save(map1_path)
    print(f"Saved: {map1_path}")

    # -------------------------------------------------------------------------
    # MAP 2: Corridor / Cluster View
    # -------------------------------------------------------------------------
    m2 = folium.Map(location=[center_lat, center_lon], zoom_start=9, control_scale=True)

    folium.TileLayer("cartodbpositron", name="Clean Street Map", overlay=False).add_to(m2)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Esri Satellite", overlay=False
    ).add_to(m2)

    for kid in unique_kitchens:
        corridor_group = folium.FeatureGroup(name=f"Corridor / Kitchen {kid}")

        if s_k_col and s_k_col in gdf_s.columns:
            sub_schools = gdf_s[gdf_s[s_k_col] == kid]
        else:
            sub_schools = gdf_s

        sub_routes = gdf_r[gdf_r[r_k_col] == kid] if r_k_col and r_k_col in gdf_r.columns else None

        if sub_routes is not None and not sub_routes.empty:
            folium.GeoJson(
                sub_routes,
                style_function=lambda f, color=color_dict.get(kid, "#333333"): {
                    "color": color, "weight": 2, "opacity": 0.7
                },
            ).add_to(corridor_group)

        if not sub_schools.empty:
            folium.GeoJson(
                sub_schools,
                marker=folium.CircleMarker(radius=4),
                popup=folium.GeoJsonPopup(fields=school_fields, aliases=school_aliases) if school_fields else None,
                style_function=lambda f, color=color_dict.get(kid, "#808080"): {
                    "fillColor": color, "color": "black", "weight": 1, "fillOpacity": 0.9
                },
            ).add_to(corridor_group)

        corridor_group.add_to(m2)

    kitchens_master = folium.FeatureGroup(name="All Central Kitchens", show=True)
    for idx, row in gdf_k.iterrows():
        total_served = row.get("meals_day", 0)
        capacity_color = get_capacity_color(total_served)
        kid = row[k_id_col]
        network_color = color_dict.get(kid, "#333333")

        icon_html = f"""
            <div style="background-color: {capacity_color}; border: 3px solid {network_color}; 
                        border-radius: 50%; width: 22px; height: 22px; 
                        display: flex; align-items: center; justify-content: center;">
                <i class="fa fa-utensils" style="color: black; font-size: 9px;"></i>
            </div>
        """
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            icon=folium.DivIcon(html=icon_html, icon_anchor=(11, 11)),
            popup=f"<b>Kitchen ID:</b> {kid}<br><b>Daily Meals:</b> {int(total_served):,}",
        ).add_to(kitchens_master)

    kitchens_master.add_to(m2)

    m2.get_root().html.add_child(folium.Element(title_html))
    Fullscreen(position="topright", force_separate_button=True).add_to(m2)
    folium.LayerControl(collapsed=False).add_to(m2)

    map2_path = os.path.join(output_dir, "40min_interactive_map_corridors.html")
    m2.save(map2_path)
    print(f"Saved: {map2_path}")

# ------------------------------------------------------------------------------
# 5. KITCHEN PROFILES & JSON METADATA GENERATOR (WITH OVERVIEW & LOCAL MAPS)
# ------------------------------------------------------------------------------

def generate_kitchen_profile_assets(gdf_k, gdf_s, gdf_r, gdf_c, gdf_districts=None, output_dir=PROFILES_DIR):
    """Generates local corridor maps, national overview maps, JSON metadata, and assigned school lists."""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "kitchen_profiles.json")
    
    k_id_col = next((c for c in ['kitchen_node', 'kitchen_id', 'fid'] if c in gdf_k.columns), gdf_k.columns[0])
    s_k_col = next((c for c in ['kitchen_node', 'kitchen_id', 'assigned_kitchen'] if c in gdf_s.columns), None)
    r_k_col = next((c for c in ['kitchen_node', 'kitchen_id'] if c in gdf_r.columns), None)
    c_k_col = next((c for c in ['kitchen_node', 'kitchen_id', 'assigned_kitchen'] if c in gdf_c.columns), None)

    corr_cmap = plt.get_cmap("tab10")
    profiles_data = []

    print(f"\n[INFO] Starting profile asset generation for {len(gdf_k)} kitchens...")

    fallback_bounds = None
    if gdf_districts is None or gdf_districts.empty:
        combined_geoms = pd.concat([gdf_k.geometry, gdf_s.geometry])
        fallback_bounds = combined_geoms.total_bounds

    for idx, (_, k_row) in enumerate(gdf_k.iterrows()):
        kid = k_row[k_id_col]
        
        district = str(k_row.get("district", "N/A"))
        sector = str(k_row.get("sector", "N/A"))
        cell = str(k_row.get("cell", "N/A"))
        village = str(k_row.get("village", "N/A"))
        codename = k_row.get("codename", f"Kitchen {kid} - {district}")
        
        sub_schools = gdf_s[gdf_s[s_k_col] == kid].copy() if s_k_col and s_k_col in gdf_s.columns else gdf_s.iloc[0:0].copy()
        sub_routes = gdf_r[gdf_r[r_k_col] == kid].copy() if r_k_col and r_k_col in gdf_r.columns else None

        kitchen_corridors = []
        corridor_color_map = {}

        if gdf_c is not None and not gdf_c.empty:
            sub_c = gdf_c[gdf_c[c_k_col] == kid].copy() if c_k_col and c_k_col in gdf_c.columns else gdf_c.copy()
            valid_c = sub_c[sub_c['corridor'] != -1] if 'corridor' in sub_c.columns else sub_c
            
            if not valid_c.empty:
                unique_corr_ids = sorted(valid_c['corridor'].unique())
                for c_idx, c_id in enumerate(unique_corr_ids):
                    corridor_color_map[c_id] = mcolors.to_hex(corr_cmap(c_idx % 10))

                # Map corridor IDs back onto schools
                if 'corridor' in valid_c.columns and not sub_schools.empty:
                    school_to_corr = valid_c.set_index('school_id')['corridor'].to_dict() if 'school_id' in valid_c.columns else {}
                    if 'school_id' in sub_schools.columns:
                        sub_schools['corridor'] = sub_schools['school_id'].map(school_to_corr).fillna(-1)

                grouped_c = valid_c.groupby('corridor').agg(
                    schools=('school_id', 'count') if 'school_id' in valid_c.columns else ('demand', 'count'),
                    demand=('demand', 'sum') if 'demand' in valid_c.columns else ('corridor', 'count'),
                    bearing=('bearing_deg', 'mean') if 'bearing_deg' in valid_c.columns else ('corridor', lambda x: 0.0)
                ).reset_index()

                for _, c_row in grouped_c.iterrows():
                    c_id_val = int(c_row['corridor'])
                    
                    # Extract list of schools for this specific corridor
                    c_sch_df = sub_schools[sub_schools['corridor'] == c_id_val] if 'corridor' in sub_schools.columns else sub_schools.iloc[0:0]
                    sch_list = []
                    for _, s_row in c_sch_df.iterrows():
                        sch_list.append({
                            "school_id": str(s_row.get("school_id", "N/A")),
                            "meals": int(s_row.get("demand", 0)),
                            "drive_time": round(float(s_row.get("time_min", 0.0)), 1),
                            "village": str(s_row.get("village", s_row.get("sector", "N/A")))
                        })

                    kitchen_corridors.append({
                        "Corridor ID": f"COR-{c_id_val}",
                        "Schools": int(c_row['schools']),
                        "Daily Demand": int(c_row['demand']),
                        "Avg Bearing (°)": round(float(c_row['bearing']), 1),
                        "school_details": sch_list
                    })

                if 'corridor' in valid_c.columns and sub_routes is not None and not sub_routes.empty:
                    if 'corridor' not in sub_routes.columns and 'school_id' in sub_routes.columns:
                        school_to_corr = valid_c.set_index('school_id')['corridor'].to_dict()
                        sub_routes['corridor'] = sub_routes['school_id'].map(school_to_corr).fillna(-1)

        # 1. National Overview Map
        fig_nat, ax_nat = plt.subplots(figsize=(4, 4))
        fig_nat.patch.set_alpha(0.0)
        ax_nat.patch.set_alpha(0.0)

        if gdf_districts is not None and not gdf_districts.empty:
            gdf_districts.plot(ax=ax_nat, facecolor="#e2e8f0", edgecolor="#475569", linewidth=0.7, zorder=1)
            minx, miny, maxx, maxy = gdf_districts.total_bounds
            ax_nat.set_xlim(minx, maxx)
            ax_nat.set_ylim(miny, maxy)
        elif fallback_bounds is not None:
            minx, miny, maxx, maxy = fallback_bounds
            pad_x = (maxx - minx) * 0.1
            pad_y = (maxy - miny) * 0.1
            ax_nat.set_xlim(minx - pad_x, maxx + pad_x)
            ax_nat.set_ylim(miny - pad_y, maxy + pad_y)

        gdf_k.plot(ax=ax_nat, color="#64748b", markersize=14, alpha=0.6, zorder=2)

        if not sub_schools.empty:
            s_minx, s_miny, s_maxx, s_maxy = sub_schools.total_bounds
            cx, cy = (s_minx + s_maxx) / 2, (s_miny + s_maxy) / 2
            ax_nat.plot(cx, cy, marker='o', markersize=30, color='#e74c3c', alpha=0.2, zorder=3)
            ax_nat.plot(cx, cy, marker='o', markersize=14, color='#e74c3c', alpha=0.4, zorder=4)

        k_point = gpd.GeoSeries([k_row.geometry], crs=gdf_k.crs)
        k_point.plot(ax=ax_nat, color='#dc2626', markersize=80, marker='*', edgecolor='white', linewidth=0.8, zorder=5)

        ax_nat.set_title("Location in Rwanda", fontsize=8, fontweight='bold', pad=4)
        ax_nat.axis('off')
        
        nat_filename = f"overview_map_k_{kid}.png"
        nat_path = os.path.join(output_dir, nat_filename)
        plt.tight_layout()
        plt.savefig(nat_path, dpi=160, transparent=True)
        plt.close()

        # 2. Local Cluster Map
        fig_loc, ax_loc = plt.subplots(figsize=(5, 5))
        fig_loc.patch.set_alpha(0.0)
        ax_loc.patch.set_alpha(0.0)

        if sub_routes is not None and not sub_routes.empty:
            if 'corridor' in sub_routes.columns:
                for c_id, color in corridor_color_map.items():
                    c_routes = sub_routes[sub_routes['corridor'] == c_id]
                    if not c_routes.empty:
                        c_routes.plot(ax=ax_loc, color=color, linewidth=2.0, alpha=0.85, label=f"COR-{int(c_id)}")
                unclass_routes = sub_routes[~sub_routes['corridor'].isin(corridor_color_map.keys())]
                if not unclass_routes.empty:
                    unclass_routes.plot(ax=ax_loc, color='#bdc3c7', linewidth=1.2, alpha=0.5)
            else:
                sub_routes.plot(ax=ax_loc, color='#2b5c8f', linewidth=1.5, alpha=0.7)

        if not sub_schools.empty:
            if 'corridor' in sub_schools.columns and corridor_color_map:
                for c_id, color in corridor_color_map.items():
                    c_sch = sub_schools[sub_schools['corridor'] == c_id]
                    if not c_sch.empty:
                        c_sch.plot(ax=ax_loc, color=color, markersize=35, edgecolor='black', zorder=3)
                unclass_sch = sub_schools[~sub_schools['corridor'].isin(corridor_color_map.keys())]
                if not unclass_sch.empty:
                    unclass_sch.plot(ax=ax_loc, color='#7f8c8d', markersize=25, edgecolor='black', zorder=3)
            else:
                sub_schools.plot(ax=ax_loc, color='#e74c3c', markersize=30, edgecolor='black', zorder=3)

        k_point.plot(ax=ax_loc, color='#f1c40f', markersize=140, marker='*', edgecolor='black', zorder=5)

        if not sub_schools.empty:
            minx, miny, maxx, maxy = sub_schools.total_bounds
            pad_x = max((maxx - minx) * 0.18, 0.02)
            pad_y = max((maxy - miny) * 0.18, 0.02)
            ax_loc.set_xlim(minx - pad_x, maxx + pad_x)
            ax_loc.set_ylim(miny - pad_y, maxy + pad_y)

        ax_loc.set_title("Corridor & Network Cluster Map", fontsize=8, fontweight='bold', pad=4)
        if corridor_color_map:
            ax_loc.legend(loc='lower right', fontsize=6.5, frameon=True, facecolor='white', framealpha=0.75)

        sns.despine()

        loc_filename = f"profile_map_k_{kid}.png"
        loc_path = os.path.join(output_dir, loc_filename)
        plt.tight_layout()
        plt.savefig(loc_path, dpi=160, transparent=True)
        plt.close()

        profile_entry = {
            "kitchen_id": int(kid) if isinstance(kid, (int, np.integer)) else str(kid),
            "codename": str(codename),
            "district": district,
            "sector": sector,
            "cell": cell,
            "village": village,
            "latitude": round(float(k_row.geometry.y), 5),
            "longitude": round(float(k_row.geometry.x), 5),
            "typology": "School-Based" if k_row.get("ctype") == "school" else "Standalone",
            "size_band": str(k_row.get("size_band", "N/A")),
            "daily_meals": int(k_row.get("meals_day", 0)),
            "schools_served": int(k_row.get("n_schools", len(sub_schools))),
            "avg_drive_time_min": round(float(k_row.get("mean_time_min", 0)), 1),
            "max_drive_time_min": round(float(k_row.get("max_time_min", 0)), 1),
            "map_image_path": loc_path,
            "overview_map_path": nat_path,
            "corridors": kitchen_corridors
        }
        profiles_data.append(profile_entry)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(profiles_data, f, indent=2)
        
    print(f"[SUCCESS] Exported kitchen profiles with assigned school details to: {json_path}")
    

# ------------------------------------------------------------------------------
# 6. OPERATIONAL DISPATCH SIMULATION GENERATOR (5 TRUCKS + REAL ROUTES + ESRI SATELLITE)
# ------------------------------------------------------------------------------

def generate_dispatch_simulation(gdf_k, gdf_s, gdf_r, gdf_c, output_dir=OUTPUT_DIR):
    """Generates an interactive Leaflet + JS dispatch simulation with 5 trucks and Esri satellite basemaps."""
    os.makedirs(output_dir, exist_ok=True)
    html_out = os.path.join(output_dir, "dispatch_simulation.html")

    k_id_col = next((c for c in ['kitchen_node', 'kitchen_id', 'fid'] if c in gdf_k.columns), gdf_k.columns[0])
    s_k_col = next((c for c in ['kitchen_node', 'kitchen_id', 'assigned_kitchen'] if c in gdf_s.columns), None)
    r_k_col = next((c for c in ['kitchen_node', 'kitchen_id'] if c in gdf_r.columns), None)

    # Pick a major central kitchen hub serving 10 to 18 schools to justify a 5-truck fleet
    kitchen_counts = gdf_s.groupby(s_k_col).size()
    ideal_kitchens = kitchen_counts[(kitchen_counts >= 10) & (kitchen_counts <= 18)].index
    target_kid = ideal_kitchens[0] if len(ideal_kitchens) > 0 else kitchen_counts.idxmax()

    k_target = gdf_k[gdf_k[k_id_col] == target_kid].iloc[0]
    sub_schools = gdf_s[gdf_s[s_k_col] == target_kid].copy()
    sub_routes = gdf_r[gdf_r[r_k_col] == target_kid].copy() if r_k_col and r_k_col in gdf_r.columns else gpd.GeoDataFrame()

    k_lat, k_lon = float(k_target.geometry.y), float(k_target.geometry.x)
    codename = str(k_target.get('codename', f'Kitchen {target_kid}'))

    # Vibrant modern palette for high contrast against satellite imagery
    corridor_colors = ['#60a5fa', '#34d399', '#fbbf24', '#f87171', '#c084fc']
    
    # Force partition across 5 delivery routes/trucks
    sub_schools['truck_group'] = (np.arange(len(sub_schools)) % 5) + 1

    trucks_data = []

    for t_idx in range(1, 6):
        corr_schools = sub_schools[sub_schools['truck_group'] == t_idx].sort_values('time_min')
        if corr_schools.empty:
            continue
            
        # Extract ONLY real road polyline geometries from gdf_routes
        route_coords = []
        if not sub_routes.empty:
            # Match routes associated with this kitchen
            for geom in sub_routes.geometry:
                if geom.geom_type == 'LineString':
                    route_coords.append([[float(lat), float(lon)] for lon, lat in geom.coords])
                elif geom.geom_type == 'MultiLineString':
                    for line in geom.geoms:
                        route_coords.append([[float(lat), float(lon)] for lon, lat in line.coords])

        stops = []
        cum_time = 0.0

        for s_idx, (_, s_row) in enumerate(corr_schools.iterrows()):
            s_lat, s_lon = float(s_row.geometry.y), float(s_row.geometry.x)
            s_id = str(s_row.get('school_id', f'SCH-{s_idx+1}'))
            s_meals = int(s_row.get('demand', 0))
            s_drive_time = round(float(s_row.get('time_min', 12.0)), 1)
            s_village = str(s_row.get('village', s_row.get('sector', 'Local Sector')))

            arr_time = max(cum_time + 6.0, s_drive_time)
            dep_time = arr_time + 10.0  # 10 min dwell
            cum_time = dep_time

            stops.append({
                "school_id": s_id,
                "location": s_village,
                "meals": s_meals,
                "arrival_min": arr_time,
                "departure_min": dep_time,
                "lat": s_lat,
                "lon": s_lon
            })

        color = corridor_colors[(t_idx - 1) % len(corridor_colors)]
        
        trucks_data.append({
            "truck_id": f"Truck {t_idx}",
            "corridor_id": f"Route-{t_idx}",
            "color": color,
            "total_meals": sum(s['meals'] for s in stops),
            "total_schools": len(stops),
            "route_lines": route_coords, # Real road line segments only
            "stops": stops
        })

    trucks_json = json.dumps(trucks_data)
    total_target_schools = sum(t["total_schools"] for t in trucks_data)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Dispatch Simulation</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        html, body {{ margin: 0; padding: 0; height: 100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: transparent; color: #0f172a; }}
        #container {{ display: flex; flex-direction: row; height: 100vh; width: 100vw; background: transparent; gap: 12px; box-sizing: border-box; }}
        #map {{ flex: 2.2; height: 100%; border-radius: 8px; border: 1px solid #cbd5e1; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
        #sidebar {{ flex: 1; min-width: 290px; max-width: 350px; background: transparent; padding: 2px; box-sizing: border-box; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; }}
        .card {{ background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }}
        .card-title {{ font-size: 10.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; color: #64748b; margin-bottom: 4px; }}
        .metric-val {{ font-size: 20px; font-weight: 800; color: #0f172a; font-variant-numeric: tabular-nums; }}
        .btn {{ background-color: #0f172a; color: white; border: none; padding: 7px 14px; border-radius: 6px; font-weight: 600; cursor: pointer; font-size: 12px; transition: background 0.15s; }}
        .btn:hover {{ background-color: #334155; }}
        .btn-sec {{ background-color: #f1f5f9; color: #334155; border: 1px solid #cbd5e1; }}
        .btn-sec:hover {{ background-color: #e2e8f0; }}
        .truck-row {{ font-size: 11px; border-bottom: 1px solid #f1f5f9; padding-bottom: 8px; margin-bottom: 8px; }}
        .truck-row:last-child {{ border-bottom: none; margin-bottom: 0; padding-bottom: 0; }}
        .status-badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 9.5px; font-weight: 700; color: white; text-transform: uppercase; }}
        .vehicle-pin {{
            background: #0f172a;
            color: #ffffff;
            border: 2px solid #ffffff;
            border-radius: 14px;
            padding: 3px 8px;
            font-size: 10.5px;
            font-weight: 800;
            box-shadow: 0 3px 10px rgba(0,0,0,0.4);
            display: flex;
            align-items: center;
            gap: 4px;
            white-space: nowrap;
        }}
    </style>
</head>
<body>
<div id="container">
    <div id="map"></div>
    <div id="sidebar">
        <div>
            <h3 style="margin: 0 0 2px 0; font-size: 15px; font-weight: 700; color: #0f172a;">🚛 Dispatch Simulation</h3>
            <p style="margin: 0; font-size: 11px; color: #64748b;">{codename}</p>
        </div>

        <div class="card" style="background-color: #f8fafc; border-color: #cbd5e1;">
            <div class="card-title">Pre-Lunch Delivery Window (06:00 - 12:00)</div>
            <div id="clock" class="metric-val" style="color: #0f172a;">06:00</div>
            <div style="font-size: 10.5px; color: #64748b; margin-top: 2px;">Target Window: <span style="color:#059669; font-weight:bold;">Before 12:00 PM</span></div>
        </div>

        <div style="display: flex; gap: 6px;">
            <button id="playBtn" class="btn" onclick="toggleSimulation()">▶ Start Dispatch</button>
            <button class="btn btn-sec" onclick="resetSimulation()">↺ Reset</button>
            <select id="speedSelect" onchange="changeSpeed(this.value)" style="border-radius: 6px; border: 1px solid #cbd5e1; font-size: 11px; padding: 5px; margin-left: auto; background: white;">
                <option value="400">1x Speed</option>
                <option value="200" selected>2x Speed</option>
                <option value="80">5x Speed</option>
            </select>
        </div>

        <div class="card">
            <div class="card-title">Delivery Progress</div>
            <div style="display: flex; justify-content: space-between; margin-top: 4px;">
                <div>
                    <span style="font-size: 10.5px; color: #64748b;">Meals Delivered</span>
                    <div id="mealsDelivered" class="metric-val">0</div>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 10.5px; color: #64748b;">Schools Served</span>
                    <div id="schoolsServed" class="metric-val">0 / {total_target_schools}</div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-title">Fleet Telemetry Monitor (5 Trucks)</div>
            <div id="truckList" style="margin-top: 6px;"></div>
        </div>
    </div>
</div>

<script>
    var map = L.map('map', {{ zoomControl: true }}).setView([{k_lat}, {k_lon}], 12);
    
    // ESRI World Imagery (Satellite) - Completely free, no API key required
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
        attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
        maxZoom: 18
    }}).addTo(map);

    // Optional subtle transportation overlay for road reference
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
        opacity: 0.6,
        maxZoom: 18
    }}).addTo(map);

    // Central Kitchen Icon
    var kIcon = L.divIcon({{
        html: '<div style="background:#0f172a; color:white; border:2px solid white; border-radius:6px; width:28px; height:28px; display:flex; align-items:center; justify-content:center; box-shadow:0 2px 8px rgba(0,0,0,0.5); font-size:14px;">🏬</div>',
        iconSize: [28, 28], iconAnchor: [14, 14]
    }});
    L.marker([{k_lat}, {k_lon}], {{icon: kIcon, zIndexOffset: 500}}).addTo(map)
        .bindPopup("<b>Central Kitchen Hub</b><br>{codename}");

    var trucksData = {trucks_json};
    var truckMarkers = [];
    var isRunning = false;
    var timeStepMin = 0;
    var simInterval = null;
    var stepDelayMs = 200;

    // Render ONLY actual road polylines from GeoPackage routes layer
    trucksData.forEach(function(truck, tIdx) {{
        if (truck.route_lines && truck.route_lines.length > 0) {{
            truck.route_lines.forEach(function(lineGeom) {{
                L.polyline(lineGeom, {{
                    color: truck.color,
                    weight: 2.5,
                    opacity: 0.8,
                    lineCap: 'round',
                    lineJoin: 'round'
                }}).addTo(map);
            }});
        }}

        // Minimalist school target rings color-coded by truck route
        truck.stops.forEach(function(stop) {{
            var sIcon = L.divIcon({{
                html: '<div style="background:#ffffff; border:2.5px solid ' + truck.color + '; border-radius:50%; width:10px; height:10px; box-shadow:0 1px 4px rgba(0,0,0,0.5);"></div>',
                iconSize: [10, 10], iconAnchor: [5, 5]
            }});
            L.marker([stop.lat, stop.lon], {{icon: sIcon, zIndexOffset: 100}}).addTo(map)
                .bindPopup("<b>School: " + stop.school_id + "</b><br>Location: " + stop.location + "<br>Demand: " + stop.meals + " meals");
        }});

        // High-Visibility Vehicle Pin
        var vIcon = L.divIcon({{
            className: 'custom-vehicle-pin',
            html: '<div class="vehicle-pin" style="border-color:' + truck.color + ';"><span>🚛</span> ' + truck.truck_id + '</div>',
            iconAnchor: [30, 12]
        }});

        var marker = L.marker([{k_lat}, {k_lon}], {{icon: vIcon, zIndexOffset: 2000}}).addTo(map);
        truckMarkers.push(marker);
    }});

    // Fleet Monitor Sidebar
    function renderFleetMonitor() {{
        var html = '';
        trucksData.forEach(function(truck, idx) {{
            html += '<div class="truck-row">' +
                '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:3px;">' +
                    '<b style="font-size:11.5px; color:' + truck.color + ';">' + truck.truck_id + ' (' + truck.corridor_id + ')</b>' +
                    '<span class="status-badge" id="badge-' + idx + '" style="background:#64748b;">AT HUB</span>' +
                '</div>' +
                '<div style="font-size:10.5px; color:#475569;">Target: <span id="dest-' + idx + '">Central Kitchen</span></div>' +
                '<div style="font-size:10.5px; color:#475569;">Cargo Onboard: <span id="cargo-' + idx + '">' + truck.total_meals.toLocaleString() + '</span> meals</div>' +
            '</div>';
        }});
        document.getElementById('truckList').innerHTML = html;
    }}
    renderFleetMonitor();

    function formatTime24(minutesFrom0600) {{
        var totalMinutes = 6 * 60 + minutesFrom0600;
        var hours = Math.floor(totalMinutes / 60) % 24;
        var mins = totalMinutes % 60;
        return (hours < 10 ? '0' : '') + hours + ':' + (mins < 10 ? '0' : '') + mins;
    }}

    // Simulation Loop with 12:00 PM Cutoff
    function updateTick() {{
        timeStepMin += 1;
        document.getElementById('clock').innerText = formatTime24(timeStepMin);

        // Pre-lunch Cutoff at 12:00 PM (360 minutes past 06:00 AM)
        if (timeStepMin >= 360) {{
            clearInterval(simInterval);
            isRunning = false;
            document.getElementById('playBtn').innerText = "✓ Dispatch Completed";
            document.getElementById('playBtn').style.background = "#059669";
            document.getElementById('clock').innerText = "12:00";
            return;
        }}

        var grandDeliveredMeals = 0;
        var grandServedSchools = 0;

        trucksData.forEach(function(truck, idx) {{
            var badge = document.getElementById('badge-' + idx);
            var dest = document.getElementById('dest-' + idx);
            var cargo = document.getElementById('cargo-' + idx);
            var marker = truckMarkers[idx];

            var deliveredMeals = 0;
            var servedCount = 0;
            var currentStop = null;

            truck.stops.forEach(function(stop) {{
                if (timeStepMin >= stop.arrival_min && timeStepMin <= stop.departure_min) {{
                    currentStop = stop;
                }}
                if (timeStepMin > stop.departure_min) {{
                    deliveredMeals += stop.meals;
                    servedCount += 1;
                }}
            }});

            grandDeliveredMeals += deliveredMeals;
            grandServedSchools += servedCount;

            if (currentStop) {{
                badge.innerText = "UNLOADING";
                badge.style.background = "#d97706";
                dest.innerText = currentStop.school_id + " (" + currentStop.location + ")";
                marker.setLatLng([currentStop.lat, currentStop.lon]);
            }} else if (timeStepMin < truck.stops[0].arrival_min) {{
                badge.innerText = "TRANSIT OUT";
                badge.style.background = "#2563eb";
                dest.innerText = truck.stops[0].school_id;
                
                var ratio = Math.min(1.0, timeStepMin / truck.stops[0].arrival_min);
                var lat = {k_lat} + (truck.stops[0].lat - {k_lat}) * ratio;
                var lon = {k_lon} + (truck.stops[0].lon - {k_lon}) * ratio;
                marker.setLatLng([lat, lon]);
            }} else if (timeStepMin > truck.stops[truck.stops.length-1].departure_min) {{
                badge.innerText = "RETURNED";
                badge.style.background = "#059669";
                dest.innerText = "Central Kitchen";
                marker.setLatLng([{k_lat}, {k_lon}]);
            }} else {{
                badge.innerText = "EN ROUTE";
                badge.style.background = "#2563eb";
                dest.innerText = "Next Corridor Stop";
            }}

            cargo.innerText = Math.max(0, truck.total_meals - deliveredMeals).toLocaleString();
        }});

        document.getElementById('mealsDelivered').innerText = grandDeliveredMeals.toLocaleString();
        document.getElementById('schoolsServed').innerText = grandServedSchools + " / " + {total_target_schools};
    }}

    function toggleSimulation() {{
        var btn = document.getElementById('playBtn');
        if (!isRunning) {{
            if (timeStepMin >= 360) resetSimulation();
            isRunning = true;
            btn.innerText = "⏸ Pause";
            btn.style.background = "#dc2626";
            simInterval = setInterval(updateTick, stepDelayMs);
        }} else {{
            isRunning = false;
            btn.innerText = "▶ Resume";
            btn.style.background = "#0f172a";
            clearInterval(simInterval);
        }}
    }}

    function resetSimulation() {{
        isRunning = false;
        clearInterval(simInterval);
        timeStepMin = 0;
        var btn = document.getElementById('playBtn');
        btn.innerText = "▶ Start Dispatch";
        btn.style.background = "#0f172a";
        document.getElementById('clock').innerText = "06:00";
        document.getElementById('mealsDelivered').innerText = "0";
        document.getElementById('schoolsServed').innerText = "0 / " + {total_target_schools};
        truckMarkers.forEach(function(m) {{ m.setLatLng([{k_lat}, {k_lon}]); }});
        renderFleetMonitor();
    }}

    function changeSpeed(val) {{
        stepDelayMs = parseInt(val);
        if (isRunning) {{
            clearInterval(simInterval);
            simInterval = setInterval(updateTick, stepDelayMs);
        }}
    }}
</script>
</body>
</html>
"""

    with open(html_out, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[SUCCESS] Exported 5-Truck Satellite Dispatch Simulation to: {html_out}")
# ------------------------------------------------------------------------------
# 7. EXECUTION PIPELINE
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    if gdf_kitchens is not None and gdf_schools is not None:
        kitchen_sum, kpi_sum, corridor_sum = generate_summary_tables(gdf_kitchens, gdf_schools, gdf_corridors)
        
        print("\n--- NETWORK KEY PERFORMANCE INDICATORS ---")
        print(kpi_sum.to_string(index=False))
        
        print("\n--- KITCHEN SUMMARY BY TYPOLOGY AND SIZE ---")
        print(kitchen_sum.to_string(index=False))
        
        # Generate interactive Plotly charts inside ./output/
        plot_drive_time_distribution(gdf_schools, output_dir=OUTPUT_DIR)
        plot_kitchen_meals_capacity(gdf_kitchens, output_dir=OUTPUT_DIR)
        plot_terrain_penalties(output_dir=OUTPUT_DIR)
        
        # Build interactive Folium maps inside ./output/
        build_interactive_leaflet_maps(gdf_kitchens, gdf_schools, gdf_routes, gdf_corridors, output_dir=OUTPUT_DIR)

        # Generate profile cards & export JSON metadata inside ./output/profiles/
        generate_kitchen_profile_assets(
            gdf_kitchens, 
            gdf_schools, 
            gdf_routes, 
            gdf_corridors, 
            gdf_districts=gdf_districts, 
            output_dir=PROFILES_DIR
        )

        # Generating the drive time simulation and planning
        generate_dispatch_simulation(gdf_kitchens, gdf_schools, gdf_routes, gdf_corridors, output_dir=OUTPUT_DIR)

    else:
        print("Script failed: GeoPackage layers could not be loaded.")
