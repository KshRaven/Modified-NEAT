import os
import json
import warnings

from typing import Any

from .constants import (
    COLOR_ROAD, COLOR_GRASS, COLOR_CURB_PRI, COLOR_CURB_SEC, COLOR_START_SEC, 
    COLOR_START_PRI, COLOR_GRAVEL, COLOR_EMPTY
)


class Configuration:
    """Base configuration class for loading parameters from file or dict."""
    
    def __init__(self, *, params: dict[str, Any] | None = None, file: str = None) -> None:
        # Load from json file provided or parameter dictionary
        if file is not None:
            if params is not None:
                warnings.warn(
                    "Overloading parameters with file input. "
                    "File input takes precedence over provided params."
                )
            params = self.load(file)
        
        if params is None:
            params = dict()
        
        self.params = params

    def load(self, file: str) -> dict[str, Any]:
        config_type = self.__class__.__name__
        # Check if file exists
        if not os.path.isfile(file):
            raise FileNotFoundError(f"{config_type} parameter file not found: {file}")
        
        with open(file, 'r') as f:
            print(f"Loaded {config_type} parameters from file: {file}")
            return json.load(f)
        
    def save(self, file: str) -> bool:
        # TODO: Implement
        pass


class TileConfig(Configuration):
    """Tile rendering configuration."""
    
    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)
        
        # Surface ratios
        self.gravel: float = self.params.get("gravel", 0.050)
        self.grass: float = self.params.get("grass", 0.100)
        self.curb: float = self.params.get("curb", 0.075)
        self.curb_pattern_splits: int = abs(int(self.params.get("curb_pattern_splits", 8)))
        self.start_checker_size: int | None = self.params.get("start_checker_size", None)
        self.finish_line_back_offset_ratio: float = self.params.get("finish_line_back_offset_ratio", 0.10)
        
        # Surface colors
        self.road_color: tuple[int, int, int] = tuple(self.params.get("road_color", COLOR_ROAD))
        self.grass_color: tuple[int, int, int] = tuple(self.params.get("grass_color", COLOR_GRASS))
        self.curb_red: tuple[int, int, int] = tuple(self.params.get("curb_red", COLOR_CURB_PRI))
        self.curb_white: tuple[int, int, int] = tuple(self.params.get("curb_white", COLOR_CURB_SEC))
        self.gravel_color: tuple[int, int, int] = tuple(self.params.get("gravel_color", COLOR_GRAVEL))


class GridConfig(Configuration):
    """Grid configuration for track layout."""
    
    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)
        
        self.grid_size: tuple[int, int] = tuple(self.params.get("grid_size", (5, 5)))
        self.cell_size: int = self.params.get("cell_size", 100)
        self.granularity: str = self.params.get("granularity", "pixels")  # 'pixels' or 'cell'
        
        self.tile: TileConfig = TileConfig(params=self.params.get("tile", {}))


class CarConfig(Configuration):
    """Car/Vehicle physics and control configuration."""
    
    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)
        
        # Movement physics
        self.max_linear_velocity: float  = self.params.get("max_linear_velocity", 5.0)
        self.max_angular_velocity: float = self.params.get("max_angular_velocity", 5.0)
        self.linear_acceleration: float  = self.params.get("linear_acceleration", 0.1)
        self.angular_acceleration: float = self.params.get("angular_acceleration", 1.0)
        
        # Control coefficients
        self.reverse_coeff: float = self.params.get("reverse_coeff", 0.50) # TODO: Label these
        self.brake_coeff: float   = self.params.get("brake_coeff",   2.00)
        self.checker_coeff: float = self.params.get("checker_coeff", 2.00)
        
        # Visual/behavioral
        self.scale: float = self.params.get("car_scale", 0.20)
        # Visual front-wheel steering (degrees): full lock at standstill, smaller lock at top speed,
        # rendered in steps of wheel_angle_step (sprite cache granularity)
        self.wheel_lock_low_speed: float = self.params.get("wheel_lock_low_speed", 32.0)
        self.wheel_lock_top_speed: float = self.params.get("wheel_lock_top_speed", 16.0)
        self.wheel_angle_step: float = self.params.get("wheel_angle_step", 3.0)
        self.state_tiles: int = self.params.get("state_tiles", 3) # 3)  # Tiles used in state distance/phase features (min 1)
        self.use_surface: bool = self.params.get("use_surface", True)  # Surface code under the car centre
        self.use_lidar: bool = self.params.get("use_lidar", True)
        self.use_tile_lidar: bool = self.params.get("use_tile_lidar", True)  # 4 beams (every 90 deg) to the current tile boundary
        
        # Control settings
        self.toggle_reverse: bool = self.params.get("toggle_reverse", True)
        self.restrict_movement: bool | int = self.params.get("restrict_movement", True)
        self.discrete: bool = self.params.get("discrete", False)
        self.full_restart: bool = self.params.get("full_restart", False)
        
        # Thresholds and limits
        self.min_brake_steps: int = self.params.get("min_brake_steps", 3)
        self.min_straight_steps: int = self.params.get("min_straight_steps", 5)
        self.straight_alignment_tolerance: float = self.params.get("straight_alignment_tolerance", 10.0)
        self.cutoff: float = self.params.get("cutoff", 0.5)
        self.frames_per_tile: int = self.params.get("frames_per_tile", 100)
        
        # Surface traction: scales the engine / brake force AND the top speed reachable on that surface
        # (1.0 = full). It never decays the velocity itself: a car that is over the surface's top speed
        # sheds the excess at engine-brake rate, and a coasting car keeps its momentum.
        self.friction_road:   float = self.params.get("friction_road",   1.00)
        self.friction_curb:   float = self.params.get("friction_curb",   0.98)
        self.friction_grass:  float = self.params.get("friction_grass",  0.96)
        self.friction_gravel: float = self.params.get("friction_gravel", 0.94)

        # Surface handling: steering authority (yaw acceleration AND max yaw rate). > 1 = the body rotates
        # more eagerly. Rotation is not the same as turning: how much the car's path follows its nose is
        # decided by slip below.
        self.handling_road:   float = self.params.get("handling_road",   1.00)
        self.handling_curb:   float = self.params.get("handling_curb",   1.25)
        self.handling_grass:  float = self.params.get("handling_grass",  1.50)
        self.handling_gravel: float = self.params.get("handling_gravel", 1.50)

        # Surface slip: how much of last frame's velocity vector survives into this frame (0 = the path
        # follows the nose instantly, 1 = it ignores the nose). It is a blend, never an addition, so it
        # cannot add speed. The effect is scaled by speed, so it vanishes when the car is nearly stopped.
        self.slip_road:   float = self.params.get("slip_road",   0.05)
        self.slip_curb:   float = self.params.get("slip_curb",   0.10)
        self.slip_grass:  float = self.params.get("slip_grass",  0.45)
        self.slip_gravel: float = self.params.get("slip_gravel", 0.50)

        # Lateral grip: the sideways acceleration (px/frame^2) the tyres hold on a perfect road, reduced by the
        # surface's slip. The car's PATH can only turn towards its nose at grip / speed radians per frame, so
        # a corner of radius R has a top speed of ~ sqrt(grip x R): that is what forces braking, and why a
        # rotated nose on grass barely changes where the car goes.
        self.lateral_grip: float = self.params.get("lateral_grip", 0.33)
        # Below this fraction of max speed grip is progressively restored (slip fades out at crawling speed)
        self.grip_relief: float = self.params.get("grip_relief", 0.33)
        # Speed lost while the car travels sideways to its nose (0 = none, 1 = all of it at 90 degrees)
        self.slide_drag: float = self.params.get("slide_drag", 0.35)
        # Gravel randomly nudges the heading by up to this many degrees per frame (scaled by speed)
        self.gravel_noise: float = self.params.get("gravel_noise", 2.0)

        # LiDAR configuration
        self.lidar_angles: int = self.params.get("lidar_angles", 8)
        self.lidar_max_dist: float | None = self.params.get("lidar_max_dist", None)  # If None, uses grid.cell_size * 2.5
        self.lidar_step: int = self.params.get("lidar_step", 4)
        
        # Verbose overlay panels (debug text / race tower): translucent backdrop
        self.overlay_color: tuple[int, int, int] = tuple(self.params.get("overlay_color", (0, 0, 0)))
        self.overlay_alpha: int = int(self.params.get("overlay_alpha", 120))  # 0 = none, 255 = opaque
        
        # Additional behavior flags
        self.endless: bool = self.params.get("endless", True)


class StartGridConfig(Configuration):
    """Staggered starting-grid layout. The grid is laid out *backwards* along the track,
    beginning at the start tile and continuing into the tiles that precede it."""

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        # Racers per tile, starting with the start tile and then the tile behind it, etc.
        # e.g. [4, 7] -> 4 cars on the start tile, 7 on the tile behind it.
        # If more racers are requested than the list covers, the last value is repeated.
        # null -> auto: every tile is filled up to what physically fits (see `row_gap`).
        self.cars_per_tile: list[int] | None = self.params.get("cars_per_tile", [4, 7])
        # Cars side by side in one row (null -> auto from road width / car width)
        self.lanes: int | None = self.params.get("lanes", 2)
        # Longitudinal clearance per row, as a multiple of the car length. A tile that
        # cannot fit its requested count at this spacing is capped (and a warning is shown).
        self.row_gap: float = self.params.get("row_gap", 1.35)
        # Minimum lane clearance, as a multiple of the car width (used by lanes=null)
        self.lane_gap: float = self.params.get("lane_gap", 1.8)
        # Fraction of the road width the lanes spread across
        self.lane_spread: float = self.params.get("lane_spread", 0.5)
        # Fraction of a row's spacing by which odd lanes sit further back (F1-style stagger)
        self.stagger: float = self.params.get("stagger", 0.35)
        # Fraction of the tile length left empty at each tile edge
        self.edge_margin: float = self.params.get("edge_margin", 0.08)
        # How many tiles (start tile included) the grid may span. Keeps the grid compact.
        self.max_grid_tiles: int = self.params.get("max_grid_tiles", 3)


class SlipstreamConfig(Configuration):
    """Simplified slipstream (drafting) model: a cheap pairwise cone test, no wake simulation."""

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        self.enabled: bool = self.params.get("enabled", True)
        # Wake length in car lengths (distance from the leader's centre)
        self.range_cars: float = self.params.get("range_cars", 8.0)
        # Half-angle (deg) of the wake cone seen from the follower; the closer to the
        # leader's line, the stronger the effect
        self.cone_deg: float = self.params.get("cone_deg", 30.0)
        # Leader must head roughly the same way (max heading difference, deg)
        self.max_heading_diff_deg: float = self.params.get("max_heading_diff_deg", 35.0)
        # Leader must be at least this fast (fraction of max speed) to create a wake
        self.min_leader_speed: float = self.params.get("min_leader_speed", 0.30)
        # Falloff exponents (1 = linear)
        self.distance_falloff: float = self.params.get("distance_falloff", 1.0)
        self.angle_falloff: float = self.params.get("angle_falloff", 1.5)
        # Extra acceleration at full draft, as a multiple of linear_acceleration (throttle on)
        self.push: float = self.params.get("push", 0.40)
        # Extra top speed at full draft, as a fraction of max_linear_velocity
        self.top_speed_bonus: float = self.params.get("top_speed_bonus", 0.25)
        # Per-frame blend rates: the tow builds up gradually and bleeds away once you leave it
        self.ramp: float = self.params.get("ramp", 0.12)
        self.decay: float = self.params.get("decay", 0.04)
        # --- Rendering ---
        # Draw particles/dots (True) instead of a continuous curved line (False)
        # Trail style: "line" (continuous curved streak), "particles" (dots) or "flame" (layered glow)
        legacy = "particles" if self.params.get("particles", False) else ("flame" if self.params.get("flame", False) else "line")
        self.trail_type: str = str(self.params.get("trail_type", legacy)).lower()
        if self.trail_type not in ("line", "particles", "flame"):
            raise ValueError(f"slipstream.trail_type must be 'line', 'particles' or 'flame', got {self.trail_type!r}")
        self.particles: bool = self.trail_type == "particles"   # kept for existing call sites
        self.flame: bool = self.trail_type == "flame"
        # Number of past frames remembered per car; the drawn streak is (draft level * this) long
        self.trail_length: int = max(2, int(self.params.get("trail_length", 20)))
        # Colour while the tow is building up / holding, and while it is wearing off
        self.color_gain: tuple[int, int, int] = tuple(self.params.get("color_gain", (140, 215, 255)))
        self.color_decay: tuple[int, int, int] = tuple(self.params.get("color_decay", (255, 140, 40)))
        # Streak lateral offset from the car centre, as a fraction of the car width
        self.side_offset: float = self.params.get("side_offset", 0.35)
        # Particle spacing in trail frames, and random sideways spread (px) at the tail end
        self.particle_step: int = max(1, int(self.params.get("particle_step", 2)))
        self.particle_spread: float = self.params.get("particle_spread", 3.0)
        # --- Trail opacity (every trail_type) ---
        # Opacity is alpha_head at the car and fades linearly to a tail opacity at the far end. That tail
        # opacity depends on how long the streak currently is relative to trail_length: a short streak
        # ends at alpha_tail_short, a streak of maximum length ends at alpha_tail_long.
        self.alpha_head: float = min(max(self.params.get("alpha_head", 1.0), 0.0), 1.0)
        self.alpha_tail_short: float = min(max(self.params.get("alpha_tail_short", 0.80), 0.0), 1.0)
        self.alpha_tail_long: float = min(max(self.params.get("alpha_tail_long", 0.05), 0.0), 1.0)
        # --- Flame trail (trail_type = "flame"): layered ribbons, hot core -> cooler glow ---
        self.flame_layers: int = max(1, int(self.params.get("flame_layers", 3)))
        self.flame_width: float = max(1.0, self.params.get("flame_width", 6.0))    # outer layer width (px) at the car
        self.flame_taper: float = min(max(self.params.get("flame_taper", 0.6), 0.0), 0.95)  # width lost by the tail
        self.flame_flicker: float = min(max(self.params.get("flame_flicker", 0.25), 0.0), 1.0)
        self.flame_hot: tuple[int, int, int] = tuple(self.params.get("flame_hot", (255, 245, 210)))


class CollisionConfig(Configuration):
    """Simplified car-to-car collisions: every car is two circles, equal masses."""

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        self.enabled: bool = self.params.get("enabled", True)
        # Circle radius as a fraction of the car width
        self.radius_scale: float = self.params.get("radius_scale", 0.55)
        # Bounciness along the contact normal (0 = no rebound)
        self.restitution: float = self.params.get("restitution", 0.15)
        # Fraction of overlap resolved per frame (1 = fully, may feel stiff)
        self.separation: float = self.params.get("separation", 0.8)
        # Cap on how far one frame may shove a car (fraction of car width)
        self.max_shove: float = self.params.get("max_shove", 0.6)
        # Share of the sideways impulse that becomes a slide
        self.lateral_transfer: float = self.params.get("lateral_transfer", 0.6)
        # Yaw kick from off-centre hits (deg/frame per unit of impulse) and its cap
        self.spin: float = self.params.get("spin", 0.8)
        self.max_spin: float = self.params.get("max_spin", 2.0)
        # A hit never slows a moving car below this fraction of max speed. Players.update()
        # costs a life for every standstill frame, so races must not stall on contact.
        self.min_speed: float = self.params.get("min_speed", 0.08)


class CountdownConfig(Configuration):
    """Start-of-race countdown (only shown with render_mode='human')."""

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        self.enabled: bool = self.params.get("enabled", True)
        self.steps: int = max(1, int(self.params.get("steps", 3)))
        self.step_seconds: float = self.params.get("step_seconds", 0.9)
        self.go_seconds: float = self.params.get("go_seconds", 0.7)
        self.text_go: str = self.params.get("text_go", "GO!")
        self.dim: int = self.params.get("dim", 110)  # backdrop darkness 0-255


class RaceConfig(Configuration):
    """Race mode: starting grid, slipstream, collisions and countdown."""

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        self.enabled: bool = self.params.get("enabled", False)
        # Default number of racers when reset() is not given an explicit count
        self.racers: int | None = self.params.get("racers", None)
        # Realistic cap so the grid is never wall-to-wall cars
        self.max_racers: int = self.params.get("max_racers", 12)
        # When True (and race mode is on) LiDAR beams also detect other active cars, treated
        # like the curb / boundaries: a beam stops at the first car it hits.
        self.lidar_detect_cars: bool = self.params.get("lidar_detect_cars", False)

        self.start_grid: StartGridConfig = StartGridConfig(params=self.params.get("start_grid", {}))
        self.slipstream: SlipstreamConfig = SlipstreamConfig(params=self.params.get("slipstream", {}))
        self.collision: CollisionConfig = CollisionConfig(params=self.params.get("collision", {}))
        self.countdown: CountdownConfig = CountdownConfig(params=self.params.get("countdown", {}))


class RewardConfig(Configuration):
    """Time-trial reward shaping (all weights are per frame unless stated, before the lives/deaths scaling)."""

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        # False -> the original flat reward (+5/frame on road, -2/frame off road, tiny crash cost)
        self.shaping: bool = self.params.get("shaping", True)

        # --- Progress / time ---
        # Reward for driving along the centre line: a full tile of progress is worth this much.
        # Final scaling of the per-frame reward handed to the learner (Players.get_reward). The raw fitness
        # keeps the big weights below; this keeps what the learner sees (and its sum) at a sane size.
        self.reward_scale: float = self.params.get("reward_scale", 1.0e-3)
        # Flat amount added to every living player's raw reward each frame (raw units, before the scale).
        # Negative = a standing cost, positive = a survival bonus, 0 = off.
        self.reward_offset: float = self.params.get("reward_offset", 0.0)
        self.progress_weight: float = self.params.get("progress_weight", 2000.0)
        # Constant cost per frame so standing still / crawling is never neutral
        self.time_penalty: float = self.params.get("time_penalty", 4.0)
        # Optional flat bonus per frame on the road (0 = off; it rewards slow survival)
        self.survival_bonus: float = self.params.get("survival_bonus", 0.0)

        # --- Look-ahead target speed (corner anticipation) ---
        # Tiles ahead inspected for corners. Keep it in line with how far the observation sees.
        self.lookahead_tiles: int = max(1, min(int(self.params.get("lookahead_tiles", 2)), 8))
        # Tiles grouped into one corner. 2 lets a pair of opposite bends (S) cancel out into a
        # diagonal, while two same-direction bends read as a U-turn.
        self.corner_window: int = max(1, min(int(self.params.get("corner_window", 2)), 4))
        # Speed (fraction of max speed) a 90 degree corner can be taken at. None = derived from the
        # physics: corner_safety * max_angular_velocity * (corner_line_radius * tile_size / 2)
        self.corner_speed_ratio: float | None = self.params.get("corner_speed_ratio", None)
        self.corner_line_radius: float = self.params.get("corner_line_radius", 1.3)  # racing line vs centre line
        self.corner_safety: float = self.params.get("corner_safety", 0.85)
        # A U-turn (two same-direction bends) is taken this much slower than a single 90 degree bend
        self.uturn_ratio: float = self.params.get("uturn_ratio", 0.85)
        # Share of the car's full braking power assumed when planning the slow-down
        self.braking_margin: float = self.params.get("braking_margin", 0.8)
        # Progress is only credited up to the target speed; beyond it this is charged per unit of excess
        self.overspeed_weight: float = self.params.get("overspeed_weight", 600.0)

        # --- Smoothness on straights ---
        self.steer_weight: float = self.params.get("steer_weight", 40.0)    # |steering| while no corner is near
        self.jitter_weight: float = self.params.get("jitter_weight", 20.0)  # |change in steering| anywhere

        # --- Surfaces ---
        # Curbs are cheap (cheaper still on corner tiles) so clipping an apex can pay off
        self.curb_weight: float = self.params.get("curb_weight", 12.0)
        self.curb_corner_discount: float = self.params.get("curb_corner_discount", 0.7)
        self.grass_weight: float = self.params.get("grass_weight", 250.0)
        self.gravel_weight: float = self.params.get("gravel_weight", 300.0)
        # Progress made on grass / gravel is credited at only this fraction, and the first contact costs a
        # one-off, so recovering from a mistake never pays as well as never leaving the road
        self.offroad_progress: float = self.params.get("offroad_progress", 0.75) # 0.2)
        self.offroad_entry_penalty: float = self.params.get("offroad_entry_penalty", 500) # 1500.0)

        # --- One-offs ---
        # Charged on elimination, scaled by (1 + speed at the crash), so failing a corner never pays
        self.crash_penalty: float = self.params.get("crash_penalty", 3000.0)
        # Race mode only: per frame in car-to-car contact (cars then learn to go around each other)
        self.bump_weight: float = self.params.get("bump_weight", 400.0)

        # --- Traffic (race mode only) ---
        # In race mode the LiDAR reports other cars as obstacles, exactly like walls, so a policy that
        # was trained alone tends to swerve off the track to dodge them. These terms use the cars'
        # true positions (not the beams), so they work however few beams there are or however dense
        # the grid is, and they push the car to go round on the road instead of leaving it.
        # How far ahead (in car lengths) and how wide (in car widths) a car counts as "in my way"
        self.traffic_range: float = self.params.get("traffic_range", 6.0)
        self.traffic_gate: float = self.params.get("traffic_gate", 1.6)
        # Off-road / curb cost is multiplied up to this much when a car is right ahead (no wall-style swerve)
        self.edge_guard: float = self.params.get("edge_guard", 2.5)
        # Charged for turning away from the track direction on a straight while blocked
        self.veer_weight: float = self.params.get("veer_weight", 150.0)
        # Charged for closing on a car in the same lane (scaled by closing speed and proximity)
        self.closing_weight: float = self.params.get("closing_weight", 300.0)
        # Paid per frame while level with / just behind a car in the next lane, on the road and not slower
        self.lane_weight: float = self.params.get("lane_weight", 60.0)
        # One-off for a clean pass (moving ahead of a nearby car while on the road)
        self.overtake_bonus: float = self.params.get("overtake_bonus", 1500.0)


class WindowConfig(Configuration):
    """Window/Display configuration."""
    
    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)
        
        self.title: str = self.params.get("title", "Car Racer")
        self.fps: int | None = self.params.get("fps", 60)
        self.background_color: tuple[int, int, int] = tuple(self.params.get("background_color", (20, 20, 20)))
        self.font_size: int = self.params.get("font_size", 24)
        # Overlay text (HUD, leaderboard, labels, countdown) is drawn at font_size * font_scale
        self.font_scale: float = self.params.get("font_scale", 0.7)


class GenConfig(Configuration):
    """Track generation configuration."""
    
    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)
        
        # Generation parameters
        self.seed: int | None = self.params.get("seed", None)
        self.max_attempts: int = self.params.get("max_attempts", 1000)
        self.min_tiles: int = self.params.get("min_tiles", 0)


class PlayerConfig(Configuration):
    """Player initialization and behavior configuration."""
    
    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)
        
        self.lives: int = self.params.get("lives", 5)
        self.max_hiatus: int = self.params.get("max_hiatus", 100)
        self.max_frames: int = self.params.get("max_frames", 1000)


class EnvironmentConfig(Configuration):
    """Environment configuration for RacerGame (Gym-like environment)."""
    
    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)
        
        # Track layout
        self.grid_size: tuple[int, int] = tuple(self.params.get("grid_size", (6, 6)))
        self.cell_size: int = self.params.get("cell_size", 110)
        self.layout: list[str] | None = self.params.get("layout", None)
        
        # Track generation
        self.max_attempts: int = self.params.get("max_attempts", 1000)
        self.min_tiles: int = self.params.get("min_tiles", 0)
        self.seed: int | None = self.params.get("seed", None)
        
        # Gameplay
        self.lives: int = self.params.get("lives", 10)
        self.max_hiatus: int = self.params.get("max_hiatus", 100)
        self.max_frames: int = self.params.get("max_frames", 1000)
        self.min_laps: int = self.params.get("min_laps", 3)
        
        # Nested configurations
        self.tile: TileConfig = TileConfig(params=self.params.get("tile", {}))
        self.grid: GridConfig = GridConfig(params=self.params.get("grid", {}))
        self.car: CarConfig = CarConfig(params=self.params.get("car", {}))
        self.window: WindowConfig = WindowConfig(params=self.params.get("window", {}))
        self.generation: GenConfig = GenConfig(params=self.params.get("generation", {})) # NOTE: parameters used for track generation
        self.player: PlayerConfig = PlayerConfig(params=self.params.get("player", {})) # NOTE: parameters used to initalize Players class
        self.race: RaceConfig = RaceConfig(params=self.params.get("race", {})) # NOTE: opt-in race mode (grid, slipstream, collisions, countdown)
        self.reward: RewardConfig = RewardConfig(params=self.params.get("reward", {})) # NOTE: time-trial reward shaping


def overwrite_config(config: Configuration, params: dict[str, Any]):
    for attr, value in params.items():
        if hasattr(config, attr):
            setattr(config, attr, value)