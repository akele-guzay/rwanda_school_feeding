# ==============================================================================
# RWANDA SCHOOL FEEDING GIS ANALYSIS & REPORT GENERATOR
# Output: Summary statistics, Seaborn/Matplotlib Charts, Interactive Folium Maps
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

# Set overall plot styling for report figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300

# Base Output Folder
OUTPUT_DIR = "./output"
PROFILES_DIR = os.path.join(OUTPUT_DIR, "profiles")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROFILES_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 1. LOAD GEOPACKAGE LAYERS & ENRICH WITH ADMIN BOUNDARIES
# ------------------------------------------------------------------------------
GPKG_PATH = "./data/results_90K_40mins.gpkg"
VILLAGES_GPKG_PATH = "./data/villages.gpkg"  # Path to administrative boundary GPKG

try:
    gdf_kitchens = gpd.read_file(GPKG_PATH, layer="kitchens_40")
    gdf_schools = gpd.read_file(GPKG_PATH, layer="schools_40")
    gdf_corridors = gpd.read_file(GPKG_PATH, layer="corridors_40")
    gdf_routes = gpd.read_file(GPKG_PATH, layer="routes_40")
    print("Primary GeoPackage layers loaded successfully.")

    # Load Administrative Village Boundaries
    gdf_admin = gpd.read_file(VILLAGES_GPKG_PATH)

    # 1. Align Coordinate Reference Systems (CRS)
    if gdf_admin.crs != gdf_kitchens.crs:
        gdf_admin = gdf_admin.to_crs(gdf_kitchens.crs)

    # 2. Select relevant administrative columns + geometry
    admin_cols = ['province', 'district', 'sector', 'cell', 'village', 'geometry']
    gdf_admin_sub = gdf_admin[[c for c in admin_cols if c in gdf_admin.columns]]

    # 3. Perform Spatial Join (intersects) with fallback to nearest neighbor
    gdf_kitchens = gpd.sjoin(
        gdf_kitchens, 
        gdf_admin_sub, 
        how='left', 
        predicate='intersects'
    ).drop(columns=['index_right'], errors='ignore')

    # Fill any remaining unmatched points using nearest boundary
    unmatched_mask = gdf_kitchens['district'].isna() if 'district' in gdf_kitchens.columns else False
    if unmatched_mask.any():
        nearest_matches = gpd.sjoin_nearest(
            gdf_kitchens[unmatched_mask].drop(columns=['province', 'district', 'sector', 'cell', 'village'], errors='ignore'),
            gdf_admin_sub,
            how='left'
        ).drop(columns=['index_right', 'distance'], errors='ignore')
        gdf_kitchens.update(nearest_matches)

    # 4. Create readable Admin Display Name / Codename
    k_id_col = 'kitchen_node' if 'kitchen_node' in gdf_kitchens.columns else 'kitchen_id'
    
    gdf_kitchens['codename'] = gdf_kitchens.apply(
        lambda r: f"Kitchen {r.get(k_id_col, r.name)} - {r.get('district', 'N/A')} ({r.get('sector', 'N/A')} Sector)", 
        axis=1
    )

    print("Kitchens successfully enriched with official Rwanda administrative names.")

except Exception as e:
    print(f"Error during layer loading or administrative spatial join: {e}")

# ------------------------------------------------------------------------------
# 2. DATA PROCESSING & SUMMARY TABLES
# ------------------------------------------------------------------------------

def generate_summary_tables(gdf_k, gdf_s, gdf_c):
    """Generates clean summary dataframes for report inclusion."""
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
    
    # Corridor Summary (Excluding kitchen sites marked -1)
    valid_corridors = gdf_c[gdf_c['corridor'] != -1]
    corridor_summary = valid_corridors.groupby('corridor').agg(
        schools_in_corridor=('school_id', 'count'),
        total_demand=('demand', 'sum'),
        mean_bearing=('bearing_deg', 'mean')
    ).reset_index()
    
    return kitchen_summary, kpi_summary, corridor_summary

# ------------------------------------------------------------------------------
# 3. STATIC CHARTS & VISUALIZATIONS (SEABORN)
# ------------------------------------------------------------------------------

def plot_drive_time_distribution(gdf_schools, output_dir=OUTPUT_DIR):
    """Plots distribution of school drive times to nearest central kitchen."""
    output_path = os.path.join(output_dir, "drive_time_dist.png")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    sns.histplot(
        gdf_schools['time_min'], 
        kde=True, 
        bins=20, 
        color='#2b5c8f', 
        edgecolor='white',
        ax=ax
    )
    
    ax.axvline(40, color='#d95f02', linestyle='--', linewidth=1.5, label='40-Min Threshold')
    ax.axvline(gdf_schools['time_min'].mean(), color='#7570b3', linestyle=':', linewidth=1.5, 
               label=f"Mean ({gdf_schools['time_min'].mean():.1f} min)")
    
    ax.set_title("School Drive Time Distribution to Assigned Central Kitchen", fontsize=12, pad=10)
    ax.set_xlabel("Drive Time (Minutes)", fontsize=10)
    ax.set_ylabel("Number of Schools", fontsize=10)
    ax.legend(frameon=True, facecolor='#fbf8f0')
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='#fbf8f0')
    plt.close()
    print(f"Saved: {output_path}")

def plot_kitchen_meals_capacity(gdf_kitchens, output_dir=OUTPUT_DIR):
    """Plots kitchen distribution by meals/day and capacity band."""
    output_path = os.path.join(output_dir, "kitchen_capacity.png")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    sns.scatterplot(
        data=gdf_kitchens,
        x='n_schools',
        y='meals_day',
        hue='size_band',
        style='ctype',
        palette='Set2',
        s=90,
        ax=ax
    )
    
    ax.axhline(70000, color='crimson', linestyle='--', label='Cap Target (70,000 Meals)')
    ax.set_title("Kitchen Capacity: Daily Meals vs. Number of Schools Served", fontsize=12, pad=10)
    ax.set_xlabel("Number of Schools Served", fontsize=10)
    ax.set_ylabel("Meals Per Day", fontsize=10)
    ax.legend(frameon=True, facecolor='#fbf8f0', bbox_to_anchor=(1.05, 1), loc='upper left')
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='#fbf8f0')
    plt.close()
    print(f"Saved: {output_path}")

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

def generate_kitchen_profile_assets(gdf_k, gdf_s, gdf_r, gdf_c, gdf_rwanda=None, output_dir=PROFILES_DIR):
    """Generates both local corridor maps and national Rwanda overview inset maps for each kitchen."""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "kitchen_profiles.json")
    
    k_id_col = next((c for c in ['kitchen_node', 'kitchen_id', 'fid'] if c in gdf_k.columns), gdf_k.columns[0])
    s_k_col = next((c for c in ['kitchen_node', 'kitchen_id', 'assigned_kitchen'] if c in gdf_s.columns), None)
    r_k_col = next((c for c in ['kitchen_node', 'kitchen_id'] if c in gdf_r.columns), None)
    c_k_col = next((c for c in ['kitchen_node', 'kitchen_id', 'assigned_kitchen'] if c in gdf_c.columns), None)

    corr_cmap = plt.get_cmap("tab10")
    profiles_data = []

    print(f"\n[INFO] Starting profile asset generation for {len(gdf_k)} kitchens...")

    for idx, (_, k_row) in enumerate(gdf_k.iterrows()):
        kid = k_row[k_id_col]
        
        district = str(k_row.get("district", "N/A"))
        sector = str(k_row.get("sector", "N/A"))
        cell = str(k_row.get("cell", "N/A"))
        village = str(k_row.get("village", "N/A"))
        codename = k_row.get("codename", f"Kitchen {kid} - {district}")
        
        sub_schools = gdf_s[gdf_s[s_k_col] == kid].copy() if s_k_col and s_k_col in gdf_s.columns else gdf_s.iloc[0:0].copy()
        sub_routes = gdf_r[gdf_r[r_k_col] == kid].copy() if r_k_col and r_k_col in gdf_r.columns else None

        # Filter corridors
        kitchen_corridors = []
        corridor_color_map = {}

        if gdf_c is not None and not gdf_c.empty:
            sub_c = gdf_c[gdf_c[c_k_col] == kid].copy() if c_k_col and c_k_col in gdf_c.columns else gdf_c.copy()
            valid_c = sub_c[sub_c['corridor'] != -1] if 'corridor' in sub_c.columns else sub_c
            
            if not valid_c.empty:
                unique_corr_ids = sorted(valid_c['corridor'].unique())
                for c_idx, c_id in enumerate(unique_corr_ids):
                    corridor_color_map[c_id] = mcolors.to_hex(corr_cmap(c_idx % 10))

                grouped_c = valid_c.groupby('corridor').agg(
                    schools=('school_id', 'count') if 'school_id' in valid_c.columns else ('demand', 'count'),
                    demand=('demand', 'sum') if 'demand' in valid_c.columns else ('corridor', 'count'),
                    bearing=('bearing_deg', 'mean') if 'bearing_deg' in valid_c.columns else ('corridor', lambda x: 0.0)
                ).reset_index()

                for _, c_row in grouped_c.iterrows():
                    kitchen_corridors.append({
                        "Corridor ID": f"COR-{int(c_row['corridor'])}",
                        "Schools": int(c_row['schools']),
                        "Daily Demand": int(c_row['demand']),
                        "Avg Bearing (°)": round(float(c_row['bearing']), 1)
                    })

                if 'corridor' in valid_c.columns and not sub_schools.empty:
                    school_to_corr = valid_c.set_index('school_id')['corridor'].to_dict() if 'school_id' in valid_c.columns else {}
                    if 'school_id' in sub_schools.columns:
                        sub_schools['corridor'] = sub_schools['school_id'].map(school_to_corr).fillna(-1)

                if 'corridor' in valid_c.columns and sub_routes is not None and not sub_routes.empty:
                    if 'corridor' not in sub_routes.columns and 'school_id' in sub_routes.columns:
                        school_to_corr = valid_c.set_index('school_id')['corridor'].to_dict()
                        sub_routes['corridor'] = sub_routes['school_id'].map(school_to_corr).fillna(-1)

        # ---------------------------------------------------------------------
        # 1. GENERATE NATIONAL RWANDA OVERVIEW INSET MAP
        # ---------------------------------------------------------------------
        fig_nat, ax_nat = plt.subplots(figsize=(4, 4))
        fig_nat.patch.set_alpha(0.0)
        ax_nat.patch.set_alpha(0.0)

        # Plot Rwanda national or district boundaries if available, else plot all kitchens as background
        if gdf_rwanda is not None and not gdf_rwanda.empty:
            gdf_rwanda.plot(ax=ax_nat, color="#ecf0f1", edgecolor="#bdc3c7", linewidth=0.8)
        else:
            gdf_k.plot(ax=ax_nat, color="#e0e0e0", markersize=15, alpha=0.5)

        # Highlight current kitchen location on the national map
        k_point = gpd.GeoSeries([k_row.geometry], crs=gdf_k.crs)
        
        # Add highlight bounding circle/box around the cluster region
        if not sub_schools.empty:
            minx, miny, maxx, maxy = sub_schools.total_bounds
            cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
            ax_nat.plot(cx, cy, marker='o', markersize=35, color='#e74c3c', alpha=0.25, zorder=3)
            ax_nat.plot(cx, cy, marker='o', markersize=18, color='#e74c3c', alpha=0.4, zorder=4)

        k_point.plot(ax=ax_nat, color='#d35400', markersize=80, marker='*', edgecolor='black', zorder=5)

        ax_nat.set_title("Location in Rwanda", fontsize=8, fontweight='bold', pad=4)
        ax_nat.axis('off')
        
        nat_filename = f"overview_map_k_{kid}.png"
        nat_path = os.path.join(output_dir, nat_filename)
        plt.tight_layout()
        plt.savefig(nat_path, dpi=160, transparent=True)
        plt.close()

        # ---------------------------------------------------------------------
        # 2. GENERATE LOCAL CLUSTER MAP
        # ---------------------------------------------------------------------
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

        # Build metadata JSON
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

    with open(json_path, "w") as f:
        json.dump(profiles_data, f, indent=2)
        
    print(f"[SUCCESS] Exported kitchen profiles with national overview and cluster maps to: {json_path}")

# ------------------------------------------------------------------------------
# 6. EXECUTION PIPELINE
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    if gdf_kitchens is not None and gdf_schools is not None:
        kitchen_sum, kpi_sum, corridor_sum = generate_summary_tables(gdf_kitchens, gdf_schools, gdf_corridors)
        
        print("\n--- NETWORK KEY PERFORMANCE INDICATORS ---")
        print(kpi_sum.to_string(index=False))
        
        print("\n--- KITCHEN SUMMARY BY TYPOLOGY AND SIZE ---")
        print(kitchen_sum.to_string(index=False))
        
        # Generate static Seaborn plots inside ./output/
        plot_drive_time_distribution(gdf_schools, output_dir=OUTPUT_DIR)
        plot_kitchen_meals_capacity(gdf_kitchens, output_dir=OUTPUT_DIR)
        
        # Build interactive Folium maps inside ./output/
        build_interactive_leaflet_maps(gdf_kitchens, gdf_schools, gdf_routes, gdf_corridors, output_dir=OUTPUT_DIR)

        # Generate profile cards & export JSON metadata inside ./output/profiles/
        generate_kitchen_profile_assets(gdf_kitchens, gdf_schools, gdf_routes, gdf_corridors, output_dir=PROFILES_DIR)

    else:
        print("Script failed: GeoPackage layers could not be loaded.")
