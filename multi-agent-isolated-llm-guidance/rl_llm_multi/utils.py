"""Shared route, sector, weather, and rendering helpers for the isolated multi-agent setup."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon

from configs import (
    AGENT_SPEED,
    MAX_AGENTS,
    ROUTE_MIRROR_SUFFIX,
    ROUTES_CSV,
    SECTOR_GEOJSON,
    WAYPOINTS_CSV,
)


def rescaling(
    r_min: float,
    r_max: float,
    measurement: float,
    t_min: float = 0.0,
    t_max: float = 100.0,
) -> float:
    """Map a value from the raw lat/lon range into the simulator grid."""
    return (measurement - r_min) * (t_max - t_min) / (r_max - r_min) + t_min


@dataclass(frozen=True)
class RouteSpec:
    route_id: str
    canonical_route_id: str
    is_reversed: bool
    origin: str
    destination: str
    waypoint_names: Tuple[str, ...]
    points: Tuple[Tuple[float, float], ...]
    linestring: LineString
    shared_prefix_key: str
    entry_distance: float
    entry_step: int


@dataclass(frozen=True)
class AssignedRoute:
    agent_id: str
    route: RouteSpec
    color: str
    start_step: int


@dataclass
class WeatherCell:
    center: Tuple[float, float]
    major_radius_nm: float
    minor_ratio: float
    angle_rad: float
    motion_heading_rad: float
    speed_units_per_step: float
    major_growth_nm_per_step: float

    @property
    def minor_radius_nm(self) -> float:
        return float(self.major_radius_nm * self.minor_ratio)


@lru_cache(maxsize=1)
def load_sector_polygon() -> Polygon:
    """Load the sector polygon and rescale it to the simulator grid."""
    df = gpd.read_file(SECTOR_GEOJSON)
    poly = df.geometry[5]
    x_vals, y_vals = poly.exterior.coords.xy
    points = [
        (rescaling(103, 108, float(xv)), rescaling(1, 5, float(yv)))
        for xv, yv in zip(x_vals, y_vals)
    ]
    sector = Polygon(points)
    if not sector.is_valid:
        sector = sector.buffer(0)
    return sector


@lru_cache(maxsize=1)
def load_sector_bounds_nm() -> Tuple[float, float, float, float]:
    """Return approximate sector bounds in NM and grid units."""
    df = gpd.read_file(SECTOR_GEOJSON)
    poly = df.geometry[5]
    min_lon, min_lat, max_lon, max_lat = poly.bounds
    lat_mid = 0.5 * (min_lat + max_lat)
    lon_nm_per_deg = 60.0 * math.cos(math.radians(lat_mid))
    width_nm = float(max_lon - min_lon) * lon_nm_per_deg
    height_nm = float(max_lat - min_lat) * 60.0
    sector = load_sector_polygon()
    min_x, min_y, max_x, max_y = sector.bounds
    return width_nm, height_nm, float(max_x - min_x), float(max_y - min_y)


def sector_nm_scales() -> Tuple[float, float]:
    """Return approximate NM per simulator unit for x and y axes."""
    width_nm, height_nm, width_units, height_units = load_sector_bounds_nm()
    return width_nm / max(width_units, 1e-6), height_nm / max(height_units, 1e-6)


def nm_to_grid_x(distance_nm: float) -> float:
    x_scale, _ = sector_nm_scales()
    return float(distance_nm / max(x_scale, 1e-6))


def nm_to_grid_y(distance_nm: float) -> float:
    _, y_scale = sector_nm_scales()
    return float(distance_nm / max(y_scale, 1e-6))


def nm_to_grid_mean(distance_nm: float) -> float:
    x_scale, y_scale = sector_nm_scales()
    avg_scale = 0.5 * (x_scale + y_scale)
    return float(distance_nm / max(avg_scale, 1e-6))


def weather_axis_units(cell: WeatherCell) -> Tuple[float, float]:
    return nm_to_grid_x(cell.major_radius_nm), nm_to_grid_y(cell.minor_radius_nm)


def weather_polygon(cell: WeatherCell, *, resolution: int = 96) -> Polygon:
    base = Point(0.0, 0.0).buffer(1.0, resolution=resolution)
    major_units, minor_units = weather_axis_units(cell)
    poly = affinity.scale(base, xfact=major_units, yfact=minor_units, origin=(0.0, 0.0))
    poly = affinity.rotate(poly, math.degrees(cell.angle_rad), origin=(0.0, 0.0))
    poly = affinity.translate(poly, xoff=cell.center[0], yoff=cell.center[1])
    return poly


def weather_signed_clearance(cell: WeatherCell, point: Tuple[float, float]) -> float:
    poly = weather_polygon(cell)
    cur = Point(point)
    distance = float(poly.exterior.distance(cur))
    return -distance if poly.covers(cur) else distance


def weather_trail(cell: WeatherCell, steps: int) -> List[Tuple[float, float]]:
    trail = [cell.center]
    x_val, y_val = cell.center
    for _ in range(int(max(0, steps))):
        x_val += cell.speed_units_per_step * math.cos(cell.motion_heading_rad)
        y_val += cell.speed_units_per_step * math.sin(cell.motion_heading_rad)
        trail.append((x_val, y_val))
    return trail


@lru_cache(maxsize=1)
def load_waypoint_map() -> Dict[str, Tuple[float, float]]:
    """Load waypoint names and rescaled coordinates."""
    df = pd.read_csv(WAYPOINTS_CSV)[["NAME", "LAT", "LON"]]
    mapping: Dict[str, Tuple[float, float]] = {}
    for _, row in df.iterrows():
        mapping[str(row["NAME"]).strip()] = (
            rescaling(103, 108, float(row["LON"])),
            rescaling(1, 5, float(row["LAT"])),
        )
    return mapping


def stable_agent_palette(max_agents: int = MAX_AGENTS) -> List[str]:
    """Return a stable high-contrast palette for up to twelve agents."""
    base = [
        "#111827",
        "#b91c1c",
        "#0f766e",
        "#7c3aed",
        "#b45309",
        "#2563eb",
        "#be123c",
        "#15803d",
        "#4338ca",
        "#9333ea",
        "#c2410c",
        "#0f766e",
    ]
    return base[:max_agents]


def possible_agent_ids(max_agents: int = MAX_AGENTS) -> List[str]:
    return [f"A{idx + 1}" for idx in range(max_agents)]


def action_idx_to_deg(action_idx: int, action_bins: Sequence[int] = None) -> int:
    bins = list(action_bins) if action_bins is not None else [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
    if not 0 <= int(action_idx) < len(bins):
        raise ValueError(f"Action index out of range: {action_idx}")
    return int(bins[int(action_idx)])


def deg_to_action_idx(degrees: int, action_bins: Sequence[int] = None) -> int:
    bins = list(action_bins) if action_bins is not None else [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
    snapped = min(bins, key=lambda candidate: abs(int(candidate) - int(round(degrees))))
    return int(bins.index(snapped))


def _first_inside_index(
    waypoint_names: Sequence[str],
    waypoint_map: Dict[str, Tuple[float, float]],
    sector: Polygon,
) -> Optional[int]:
    for idx, name in enumerate(waypoint_names):
        if sector.covers(Point(waypoint_map[name])):
            return idx
    return None


def _extract_entry_distance(line: LineString, sector: Polygon, fallback_distance: float) -> float:
    boundary = line.intersection(sector.boundary)
    entry_points: List[Point] = []
    if boundary.is_empty:
        return fallback_distance
    if boundary.geom_type == "Point":
        entry_points = [boundary]
    elif hasattr(boundary, "geoms"):
        for geom in boundary.geoms:
            if geom.geom_type == "Point":
                entry_points.append(geom)
    if not entry_points:
        return fallback_distance
    return min(float(line.project(point)) for point in entry_points)


def _route_spec_from_waypoints(
    route_id: str,
    canonical_route_id: str,
    waypoint_names: Sequence[str],
    *,
    is_reversed: bool,
    waypoint_map: Dict[str, Tuple[float, float]],
    sector: Polygon,
) -> RouteSpec:
    points = tuple(waypoint_map[name] for name in waypoint_names)
    line = LineString(points)
    first_inside_idx = _first_inside_index(waypoint_names, waypoint_map, sector)
    if first_inside_idx is None:
        prefix_names = tuple(waypoint_names)
        fallback_entry_distance = float(line.length)
    else:
        prefix_names = tuple(waypoint_names[: first_inside_idx + 1])
        prefix_line = LineString(points[: first_inside_idx + 1])
        fallback_entry_distance = float(prefix_line.length)
    entry_distance = _extract_entry_distance(line, sector, fallback_entry_distance)
    entry_step = int(math.ceil(entry_distance / AGENT_SPEED))
    return RouteSpec(
        route_id=route_id,
        canonical_route_id=canonical_route_id,
        is_reversed=bool(is_reversed),
        origin=str(waypoint_names[0]),
        destination=str(waypoint_names[-1]),
        waypoint_names=tuple(str(name) for name in waypoint_names),
        points=points,
        linestring=line,
        shared_prefix_key="->".join(prefix_names),
        entry_distance=float(entry_distance),
        entry_step=entry_step,
    )


@lru_cache(maxsize=1)
def build_route_catalog() -> Tuple[RouteSpec, ...]:
    """Build forward and mirrored route signatures from the shared catalog."""
    path_df = pd.read_csv(ROUTES_CSV)
    waypoint_map = load_waypoint_map()
    sector = load_sector_polygon()
    catalog: List[RouteSpec] = []

    for _, row in path_df.iterrows():
        canonical_route_id = str(row["ROUTE_NAME"]).strip()
        waypoint_names: List[str] = []
        for col, value in row.items():
            if col in {"ROUTE_NAME", "SID", "STAR"} or pd.isna(value):
                continue
            name = str(value).strip()
            if name:
                waypoint_names.append(name)
        if len(waypoint_names) < 2:
            continue

        catalog.append(
            _route_spec_from_waypoints(
                canonical_route_id,
                canonical_route_id,
                waypoint_names,
                is_reversed=False,
                waypoint_map=waypoint_map,
                sector=sector,
            )
        )
        catalog.append(
            _route_spec_from_waypoints(
                f"{canonical_route_id}{ROUTE_MIRROR_SUFFIX}",
                canonical_route_id,
                list(reversed(waypoint_names)),
                is_reversed=True,
                waypoint_map=waypoint_map,
                sector=sector,
            )
        )

    signatures = {(spec.points, spec.origin, spec.destination) for spec in catalog}
    if len(signatures) != len(catalog):
        raise RuntimeError("Route catalog contains duplicate route signatures.")
    return tuple(catalog)


def max_agents_possible() -> int:
    return len(build_route_catalog())


def resolve_route_specs(route_ids: Optional[Sequence[str]] = None) -> List[RouteSpec]:
    catalog = {route.route_id: route for route in build_route_catalog()}
    if route_ids is None:
        return list(catalog.values())
    missing = [route_id for route_id in route_ids if route_id not in catalog]
    if missing:
        raise KeyError(f"Unknown route ids: {missing}")
    return [catalog[route_id] for route_id in route_ids]


def assign_routes(
    num_agents: int,
    *,
    seed: Optional[int] = None,
    route_ids: Optional[Sequence[str]] = None,
) -> List[AssignedRoute]:
    """Select routes without replacement and compute staggered start steps."""
    capacity = max_agents_possible()
    if not 1 <= int(num_agents) <= capacity:
        raise ValueError(f"num_agents must be within [1, {capacity}], got {num_agents}")

    rng = random.Random(seed)
    catalog = resolve_route_specs(route_ids)
    if route_ids is None:
        selected = rng.sample(catalog, int(num_agents))
    else:
        if len(catalog) < int(num_agents):
            raise ValueError(
                f"Need at least {num_agents} explicit route ids, got {len(catalog)}"
            )
        selected = list(catalog[: int(num_agents)])

    palette = stable_agent_palette(MAX_AGENTS)
    agent_ids = possible_agent_ids(MAX_AGENTS)
    prefix_state: Dict[str, Tuple[int, int]] = {}
    assignments: List[AssignedRoute] = []

    for idx, route in enumerate(selected):
        if route.shared_prefix_key in prefix_state:
            prev_start, prev_entry = prefix_state[route.shared_prefix_key]
            start_step = prev_start + prev_entry + 1
        else:
            start_step = 0
        prefix_state[route.shared_prefix_key] = (start_step, route.entry_step)
        assignments.append(
            AssignedRoute(
                agent_id=agent_ids[idx],
                route=route,
                color=palette[idx],
                start_step=start_step,
            )
        )
    return assignments


def heading_from_line(line: LineString, distance: float, eps: float = 1e-3) -> float:
    """Approximate the tangent heading of a route at the given distance."""
    distance = max(0.0, min(float(distance), float(line.length)))
    delta = max(eps, min(1.0, 0.001 * max(float(line.length), 1.0)))
    p0 = line.interpolate(max(0.0, distance - delta))
    p1 = line.interpolate(min(float(line.length), distance + delta))
    return math.atan2(p1.y - p0.y, p1.x - p0.x)


def line_signed_cross_track(line: LineString, point: Tuple[float, float]) -> Tuple[float, float, float]:
    """Return signed cross-track distance, along-track position, and route heading."""
    cur = Point(point)
    s = float(line.project(cur))
    foot = line.interpolate(s)
    route_heading = heading_from_line(line, s)
    tx, ty = math.cos(route_heading), math.sin(route_heading)
    dx, dy = cur.x - foot.x, cur.y - foot.y
    cross = tx * dy - ty * dx
    signed_xtrk = math.copysign(math.hypot(dx, dy), cross)
    return signed_xtrk, s, route_heading


def clip_position(x_val: float, y_val: float) -> Tuple[float, float]:
    """Clamp free-flight positions to the same broad play area as the current env."""
    return float(np.clip(x_val, 15.0, 100.0)), float(np.clip(y_val, 15.0, 100.0))


def plot_sector() -> plt.Figure:
    """Reproduce the current sector base plot with a fixed 12x8 size."""
    waypoint_map = load_waypoint_map()
    sector = load_sector_polygon()
    fig, ax = plt.subplots(figsize=(12, 8))

    sector_x, sector_y = sector.exterior.xy
    ax.plot(sector_x, sector_y, color="black")

    waypoint_x = [coords[0] for coords in waypoint_map.values()]
    waypoint_y = [coords[1] for coords in waypoint_map.values()]
    ax.scatter(waypoint_x, waypoint_y, c="maroon", s=5, alpha=1.0)
    for name, coords in waypoint_map.items():
        ax.text(
            coords[0] + 0.25,
            coords[1] + 0.25,
            name,
            color="black",
            fontsize=8,
            alpha=0.85,
        )

    for route in build_route_catalog():
        xs = [point[0] for point in route.points]
        ys = [point[1] for point in route.points]
        color = "#3b82f6" if not route.is_reversed else "#0ea5e9"
        ax.plot(xs, ys, color=color, alpha=0.18, linewidth=0.8)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("auto")
    return fig


def save_rendered_figure(figure: plt.Figure, output_path: Path) -> None:
    """Save a rendered frame at exactly 1200x800."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=100)
    plt.close(figure)
