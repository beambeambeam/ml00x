"""Explore the four login locations in data/test.csv (no labels required)."""

from pathlib import Path
import math
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.basemap import Basemap
import pandas as pd
import plotly.graph_objects as go


def main():
    data = pd.read_csv("data/test.csv", usecols=lambda c: c == "player_id" or c.startswith(("login_lat_", "login_lon_")))
    locations = []
    summaries = []
    for period in range(1, 5):
        lat = data[f"login_lat_{period}"]
        lon = data[f"login_lon_{period}"]
        missing = lat.isna() | lon.isna()
        valid = ~missing & lat.between(-90, 90) & lon.between(-180, 180)
        summaries.append({"period": period, "rows": len(data), "missing_either": int(missing.sum()),
                          "out_of_range": int((~missing & ~valid).sum()), "valid_pairs": int(valid.sum())})
        locations.append(pd.DataFrame({"latitude": lat[valid], "longitude": lon[valid]}))

    points = pd.concat(locations, ignore_index=True)
    assets = Path("assets")
    assets.mkdir(exist_ok=True)
    pd.DataFrame(summaries).to_csv(assets / "login-location-summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(15, 7.5))
    world = Basemap(projection="cyl", llcrnrlon=-180, urcrnrlon=180,
                    llcrnrlat=-90, urcrnrlat=90, resolution="c", ax=ax)
    world.drawmapboundary(fill_color="#eaf3f8", linewidth=0)
    world.fillcontinents(color="#e5e7e4", lake_color="#eaf3f8", zorder=0)
    world.drawcoastlines(color="#637674", linewidth=0.45, zorder=3)
    world.drawparallels(range(-60, 91, 30), labels=[1, 0, 0, 0], color="#c3d1d5", linewidth=0.3, fontsize=8)
    world.drawmeridians(range(-180, 181, 60), labels=[0, 0, 0, 1], color="#c3d1d5", linewidth=0.3, fontsize=8)
    density = ax.hexbin(points.longitude, points.latitude, gridsize=(120, 60), mincnt=1,
                        norm=LogNorm(), cmap="magma", linewidths=0, zorder=2)
    colorbar = fig.colorbar(density, ax=ax, orientation="horizontal", fraction=0.045, pad=0.08, aspect=45)
    colorbar.set_label("Valid coordinate pairs per hexagon (log scale)")
    ax.set_title(f"Login locations across four periods\n{len(points):,} valid coordinate pairs from {len(data):,} players", loc="left", weight="bold", fontsize=15, pad=14)
    fig.savefig(assets / "login-locations-world-map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(points.longitude, points.latitude, s=1, alpha=0.07, color="#245a9b", rasterized=True)
    ax.set(xlabel="Longitude (degrees)", ylabel="Latitude (degrees)",
           title="Login coordinates across four periods")
    ax.grid(alpha=0.2)
    fig.savefig(assets / "login-coordinates-2d.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Sample players, not individual points, so each line retains its identity.
    sample = data.sample(n=min(300, len(data)), random_state=42)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    for _, row in sample.iterrows():
        lat = [row[f"login_lat_{i}"] for i in range(1, 5)]
        lon = [row[f"login_lon_{i}"] for i in range(1, 5)]
        valid = [pd.notna(a) and pd.notna(b) and -90 <= a <= 90 and -180 <= b <= 180
                 for a, b in zip(lat, lon)]
        # NaNs break lines across missing or invalid periods.
        lat = [a if ok else float("nan") for a, ok in zip(lat, valid)]
        lon = [b if ok else float("nan") for b, ok in zip(lon, valid)]
        ax.plot(lon, lat, range(1, 5), marker=".", markersize=2, linewidth=0.5, alpha=0.3, color="#245a9b")
    ax.set(xlabel="Longitude (degrees)", ylabel="Latitude (degrees)", zlabel="Period",
           title="Login coordinates over time: 300 sampled players")
    ax.set_zticks([1, 2, 3, 4])
    ax.view_init(elev=22, azim=-62)
    fig.savefig(assets / "login-coordinate-trajectories-3d.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Batch points and line segments into two traces for WebGL performance.
    xs, ys, zs = [], [], []
    line_x, line_y, line_z = [], [], []
    flagged_routes = []
    tolerance_degrees = 3.0
    for row in data.itertuples(index=False, name=None):
        coordinates = dict(zip(data.columns, row))
        route = []
        for period in range(1, 5):
            lat, lon = coordinates[f"login_lat_{period}"], coordinates[f"login_lon_{period}"]
            valid = pd.notna(lat) and pd.notna(lon) and -90 <= lat <= 90 and -180 <= lon <= 180
            xs.append(lon if valid else None)
            ys.append(lat if valid else None)
            zs.append(period if valid else None)
            route.append((lat, lon, period) if valid else None)
        segments = [(a, b) for a, b in zip(route, route[1:]) if a is not None and b is not None]
        changed = any(math.hypot(b[0] - a[0], min(abs(b[1] - a[1]), 360 - abs(b[1] - a[1])))
                      > tolerance_degrees for a, b in segments)
        if changed:
            route_x, route_y, route_z = [], [], []
            for a, b in segments:
                # Avoid a misleading line across the map at the antimeridian.
                if abs(b[1] - a[1]) <= 180:
                    line_x.extend([a[1], b[1], None])
                    line_y.extend([a[0], b[0], None])
                    line_z.extend([a[2], b[2], None])
                    route_x.extend([a[1], b[1], None])
                    route_y.extend([a[0], b[0], None])
                    route_z.extend([a[2], b[2], None])
            if route_x:
                flagged_routes.append({"id": coordinates["player_id"], "x": route_x,
                                       "y": route_y, "z": route_z})
    interactive = go.Figure()
    interactive.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs, mode="markers", marker={"size": 1, "color": "#3275b5"},
        opacity=0.3, name="Valid locations",
        hovertemplate="Longitude: %{x:.2f}°<br>Latitude: %{y:.2f}°<br>Period: %{z}<extra></extra>"))
    interactive.add_trace(go.Scatter3d(
        x=line_x, y=line_y, z=line_z, mode="lines", line={"width": 2, "color": "#e1593f"},
        opacity=0.6, name=f"Full routes with a change > {tolerance_degrees:g}°", hoverinfo="skip"))
    interactive.add_trace(go.Scatter3d(
        x=[], y=[], z=[], mode="lines+markers", line={"width": 8, "color": "#ffbf00"},
        marker={"size": 4, "color": "#ffbf00"}, name="Selected route", hoverinfo="skip"))
    interactive.update_layout(
        title=f"Login coordinates over time: all {len(data):,} players",
        scene={"xaxis_title": "Longitude (degrees)", "yaxis_title": "Latitude (degrees)",
               "zaxis": {"title": "Period", "tickvals": [1, 2, 3, 4]}},
        margin={"l": 0, "r": 0, "b": 0, "t": 60})
    script = """
    const routes = ROUTES_JSON;
    const graph = document.getElementById('{plot_id}');
    if (routes.length) {
        const controls = document.createElement('div');
        controls.style.cssText = 'padding:12px 24px;font:14px sans-serif;display:flex;gap:12px;align-items:center';
        const label = document.createElement('span');
        const slider = document.createElement('input');
        slider.type = 'range'; slider.min = '0'; slider.max = String(routes.length - 1);
        slider.value = '0'; slider.style.flex = '1'; slider.setAttribute('aria-label', 'Select player route');
        controls.append(label, slider); graph.parentNode.insertBefore(controls, graph);
        function selectRoute() {
            const index = Number(slider.value), route = routes[index];
            label.textContent = 'Route ' + (index + 1) + ' / ' + routes.length + ' · Player ' + route.id;
            Plotly.restyle(graph, {x: [route.x], y: [route.y], z: [route.z]}, [2]);
        }
        slider.addEventListener('input', selectRoute); selectRoute();
    }
    """.replace("ROUTES_JSON", json.dumps(flagged_routes, allow_nan=False))
    interactive.write_html(assets / "login-coordinate-trajectories-3d.html", include_plotlyjs=True, post_script=script)
    print(f"Routes available in slider: {len(flagged_routes):,}")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
