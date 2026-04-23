"""Shared route, sector, and rendering helpers for the multi-agent setup."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon

from configs import AGENT_SPEED, MAX_AGENTS, ROUTES_CSV, SAFE_R, SECTOR_GEOJSON, WAYPOINTS_CSV


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
    """Return a stable high-contrast palette for up to six agents."""
    base = [
        "#111827",
        "#b91c1c",
        "#0f766e",
        "#7c3aed",
        "#b45309",
        "#2563eb",
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


@lru_cache(maxsize=1)
def build_route_catalog() -> Tuple[RouteSpec, ...]:
    """Build the six unique route signatures from the shared route catalog."""
    path_df = pd.read_csv(ROUTES_CSV)
    waypoint_map = load_waypoint_map()
    sector = load_sector_polygon()
    catalog: List[RouteSpec] = []

    for _, row in path_df.iterrows():
        route_id = str(row["ROUTE_NAME"]).strip()
        waypoint_names: List[str] = []
        points: List[Tuple[float, float]] = []
        for col, value in row.items():
            if col in {"ROUTE_NAME", "SID", "STAR"} or pd.isna(value):
                continue
            name = str(value).strip()
            if not name:
                continue
            waypoint_names.append(name)
            points.append(waypoint_map[name])

        if len(points) < 2:
            continue

        line = LineString(points)
        first_inside_idx = _first_inside_index(waypoint_names, waypoint_map, sector)
        if first_inside_idx is None:
            prefix_names = waypoint_names
            fallback_entry_distance = float(line.length)
        else:
            prefix_names = waypoint_names[: first_inside_idx + 1]
            prefix_line = LineString(points[: first_inside_idx + 1])
            fallback_entry_distance = float(prefix_line.length)
        entry_distance = _extract_entry_distance(line, sector, fallback_entry_distance)
        entry_step = int(math.ceil(entry_distance / AGENT_SPEED))
        catalog.append(
            RouteSpec(
                route_id=route_id,
                origin=waypoint_names[0],
                destination=waypoint_names[-1],
                waypoint_names=tuple(waypoint_names),
                points=tuple(points),
                linestring=line,
                shared_prefix_key="->".join(prefix_names),
                entry_distance=float(entry_distance),
                entry_step=entry_step,
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
    if not 1 <= int(num_agents) <= max_agents_possible():
        raise ValueError(
            f"num_agents must be within [1, {max_agents_possible()}], got {num_agents}"
        )
    rng = random.Random(seed)
    catalog = resolve_route_specs(route_ids)
    if route_ids is None:
        selected = rng.sample(catalog, int(num_agents))
    else:
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

    for route in build_route_catalog():
        xs = [point[0] for point in route.points]
        ys = [point[1] for point in route.points]
        ax.plot(xs, ys, color="blue", alpha=0.3)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("auto")
    return fig


def save_rendered_figure(figure: plt.Figure, output_path: Path) -> None:
    """Save a rendered frame at exactly 1200x800."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=100)
    plt.close(figure)
