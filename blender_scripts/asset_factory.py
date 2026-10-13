"""AeroWatch AI — Procedural 3D Asset Factory for Blender 5.2.

Constructs detailed procedural 3D geometry and PBR materials for canonical civilian classes:
  0: drone    (multirotor UAV with arms, motors, spinning rotors, camera gimbal)
  1: airplane (civilian fixed-wing commuter / utility aircraft)
  2: bird     (avian wildlife with volumetric body, head, beak, angled wings)
  3: balloon  (hot-air balloon with gore panels & basket, or weather balloon)
"""
from __future__ import annotations

import math
import random
import bpy
from mathutils import Vector, Euler

CANONICAL_CLASSES = {
    0: "drone",
    1: "aircraft",
    2: "bird",
    3: "balloon",
    4: "military_jet",
    5: "kamikaze_drone",
}



def create_pbr_material(
    name: str,
    base_color: tuple[float, float, float],
    metallic: float = 0.0,
    roughness: float = 0.5,
    specular: float = 0.5,
) -> bpy.types.Material:
    """Creates a Principled BSDF material compatible with Blender 5.2 EEVEE."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.diffuse_color = (*base_color, 1.0)
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = (*base_color, 1.0)
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
        # Blender 4+ renamed or updated specular input
        for spec_key in ("Specular IOR Level", "Specular"):
            if spec_key in bsdf.inputs:
                bsdf.inputs[spec_key].default_value = specular
                break
    return mat


def add_mesh_cube(name: str, dimensions: tuple[float, float, float], location: tuple[float, float, float], material: bpy.types.Material | None = None) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if material:
        obj.data.materials.append(material)
    return obj


def add_mesh_cylinder(
    name: str,
    radius: float,
    depth: float,
    location: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    material: bpy.types.Material | None = None,
    vertices: int = 16,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices, radius=radius, depth=depth, location=location, rotation=rotation
    )
    obj = bpy.context.object
    obj.name = name
    if material:
        obj.data.materials.append(material)
    for p in obj.data.polygons:
        p.use_smooth = True
    return obj


def add_mesh_sphere(
    name: str,
    scale: tuple[float, float, float],
    location: tuple[float, float, float],
    material: bpy.types.Material | None = None,
    segments: int = 16,
    rings: int = 12,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=segments, ring_count=rings, radius=1.0, location=location
    )
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if material:
        obj.data.materials.append(material)
    for p in obj.data.polygons:
        p.use_smooth = True
    return obj


def join_components(name: str, parts: list[bpy.types.Object], pass_index: int = 0) -> bpy.types.Object:
    """Joins sub-objects into a single parent mesh object with unified transforms."""
    bpy.ops.object.select_all(action="DESELECT")
    for part in parts:
        part.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    root = bpy.context.object
    root.name = name
    root.pass_index = pass_index
    root.rotation_euler = (0.0, 0.0, 0.0)
    root.scale = (1.0, 1.0, 1.0)
    return root


class AssetFactory:
    """Generates procedural civilian aerial objects with varied configurations."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.materials = self._build_material_library()

    def _build_material_library(self) -> dict[str, bpy.types.Material]:
        return {
            "carbon_dark": create_pbr_material("CarbonDark", (0.04, 0.05, 0.06), metallic=0.3, roughness=0.35),
            "carbon_grey": create_pbr_material("CarbonGrey", (0.12, 0.13, 0.15), metallic=0.2, roughness=0.45),
            "matte_black": create_pbr_material("MatteBlack", (0.02, 0.02, 0.02), metallic=0.1, roughness=0.8),
            "aviation_white": create_pbr_material("AviationWhite", (0.92, 0.93, 0.95), metallic=0.1, roughness=0.25),
            "aluminum": create_pbr_material("AircraftAluminum", (0.55, 0.58, 0.62), metallic=0.75, roughness=0.3),
            "safety_orange": create_pbr_material("SafetyOrange", (0.95, 0.25, 0.05), metallic=0.1, roughness=0.4),
            "aviation_red": create_pbr_material("AviationRed", (0.85, 0.08, 0.08), metallic=0.1, roughness=0.35),
            "aviation_blue": create_pbr_material("AviationBlue", (0.05, 0.35, 0.85), metallic=0.15, roughness=0.35),
            "glass_cockpit": create_pbr_material("GlassCockpit", (0.08, 0.14, 0.20), metallic=0.9, roughness=0.1),
            "rotor_blur": create_pbr_material("RotorBlur", (0.15, 0.15, 0.15), metallic=0.1, roughness=0.6),
            "bird_slate": create_pbr_material("BirdSlate", (0.18, 0.20, 0.22), metallic=0.0, roughness=0.85),
            "bird_brown": create_pbr_material("BirdBrown", (0.28, 0.19, 0.12), metallic=0.0, roughness=0.9),
            "bird_white": create_pbr_material("BirdWhite", (0.85, 0.84, 0.80), metallic=0.0, roughness=0.8),
            "bird_beak": create_pbr_material("BirdBeak", (0.92, 0.65, 0.08), metallic=0.0, roughness=0.4),
            "balloon_yellow": create_pbr_material("BalloonYellow", (0.96, 0.82, 0.12), metallic=0.05, roughness=0.5),
            "balloon_cyan": create_pbr_material("BalloonCyan", (0.08, 0.68, 0.85), metallic=0.05, roughness=0.5),
            "balloon_magenta": create_pbr_material("BalloonMagenta", (0.85, 0.12, 0.55), metallic=0.05, roughness=0.5),
            "basket_wicker": create_pbr_material("BasketWicker", (0.35, 0.24, 0.15), metallic=0.0, roughness=0.9),
        }

    # -------------------------------------------------------------
    # CLASS 0: DRONE (Civilian Multirotor UAV)
    # -------------------------------------------------------------
    def create_drone(self, subtype: int | None = None) -> bpy.types.Object:
        """Builds a civilian multirotor drone (quadcopter or hexacopter)."""
        st = subtype if subtype is not None else self.rng.choice([0, 1, 2])
        accent_color = self.rng.choice(["safety_orange", "aviation_blue", "aviation_red"])
        mat_body = self.materials["carbon_dark"]
        mat_arms = self.materials["carbon_grey"]
        mat_accent = self.materials[accent_color]
        mat_rotor = self.materials["rotor_blur"]
        mat_metal = self.materials["aluminum"]

        parts = []
        if st == 0:  # X-Frame Quadcopter with landing skids & camera pod
            body = add_mesh_cube("drone_body", (0.38, 0.52, 0.16), (0.0, 0.0, 0.0), mat_body)
            parts.append(body)
            # Center top dome
            parts.append(add_mesh_sphere("gps_dome", (0.09, 0.09, 0.05), (0.0, 0.05, 0.10), mat_accent))
            # Camera gimbal pod
            parts.append(add_mesh_sphere("gimbal", (0.08, 0.08, 0.08), (0.0, 0.24, -0.09), self.materials["glass_cockpit"]))
            # Cross arms
            arm_span_x = self.rng.uniform(1.35, 1.55)
            arm_span_y = self.rng.uniform(1.05, 1.25)
            parts.append(add_mesh_cube("arm_x1", (arm_span_x, 0.065, 0.05), (0.0, 0.0, 0.02), mat_arms))
            parts.append(add_mesh_cube("arm_y1", (0.065, arm_span_y, 0.05), (0.0, 0.0, 0.02), mat_arms))
            # 4 Motors & Rotors
            half_x, half_y = arm_span_x * 0.44, arm_span_y * 0.42
            rotor_rad = self.rng.uniform(0.24, 0.30)
            for x in (-half_x, half_x):
                for y in (-half_y, half_y):
                    parts.append(add_mesh_cylinder("motor", 0.075, 0.09, (x, y, 0.07), material=mat_metal))
                    parts.append(add_mesh_cylinder("rotor", rotor_rad, 0.015, (x, y, 0.12), material=mat_rotor, vertices=16))
            # Landing skids
            for x in (-0.22, 0.22):
                parts.append(add_mesh_cube("skid_strut", (0.03, 0.03, 0.22), (x, 0.0, -0.11), mat_arms))
                parts.append(add_mesh_cylinder("skid_rail", 0.022, 0.55, (x, 0.0, -0.22), rotation=(math.radians(90), 0, 0), material=mat_arms))

        elif st == 1:  # Hexacopter (6-arm commercial inspection UAV)
            body = add_mesh_cylinder("drone_body_hex", 0.30, 0.18, (0.0, 0.0, 0.0), material=mat_body, vertices=12)
            parts.append(body)
            parts.append(add_mesh_sphere("lidar_pod", (0.11, 0.11, 0.10), (0.0, 0.0, -0.12), self.materials["matte_black"]))
            arm_length = 0.72
            for i in range(6):
                angle = i * (math.pi / 3.0)
                arm_x = math.cos(angle) * (arm_length * 0.5)
                arm_y = math.sin(angle) * (arm_length * 0.5)
                parts.append(add_mesh_cylinder(f"hex_arm_{i}", 0.035, arm_length, (arm_x, arm_y, 0.03), rotation=(0, math.radians(90), angle), material=mat_arms))
                tip_x = math.cos(angle) * arm_length
                tip_y = math.sin(angle) * arm_length
                parts.append(add_mesh_cylinder(f"hex_motor_{i}", 0.07, 0.08, (tip_x, tip_y, 0.08), material=mat_metal))
                parts.append(add_mesh_cylinder(f"hex_rotor_{i}", 0.22, 0.014, (tip_x, tip_y, 0.13), material=mat_rotor, vertices=16))

        else:  # Compact Consumer Quadcopter (aerodynamic curved shell)
            body = add_mesh_sphere("drone_body_curved", (0.24, 0.38, 0.14), (0.0, 0.0, 0.0), material=self.materials["aviation_white"])
            parts.append(body)
            parts.append(add_mesh_cube("nose_sensor", (0.16, 0.08, 0.08), (0.0, 0.32, -0.02), self.materials["glass_cockpit"]))
            for x, y, rot_z in [(-0.45, 0.35, -25), (0.45, 0.35, 25), (-0.50, -0.38, -155), (0.50, -0.38, 155)]:
                parts.append(add_mesh_cube("arm_diag", (0.35, 0.05, 0.04), (x * 0.6, y * 0.6, -0.02), self.materials["aviation_white"]))
                parts.append(add_mesh_cylinder("motor", 0.065, 0.08, (x, y, 0.04), material=mat_metal))
                parts.append(add_mesh_cylinder("rotor", 0.20, 0.012, (x, y, 0.09), material=mat_accent, vertices=16))

        return join_components("synthetic_drone", parts, pass_index=0)

    # -------------------------------------------------------------
    # CLASS 1: AIRPLANE (Civilian Fixed-Wing Aircraft)
    # -------------------------------------------------------------
    def create_airplane(self, subtype: int | None = None) -> bpy.types.Object:
        """Builds a civilian fixed-wing aircraft (regional passenger or single-prop utility)."""
        st = subtype if subtype is not None else self.rng.choice([0, 1, 2])
        mat_white = self.materials["aviation_white"]
        mat_metal = self.materials["aluminum"]
        mat_stripe = self.materials[self.rng.choice(["aviation_blue", "aviation_red", "safety_orange"])]
        mat_glass = self.materials["glass_cockpit"]

        parts = []
        if st == 0:  # Twin-Engine Regional Jet / Airliner
            # Streamlined cylindrical fuselage
            fuse_len = self.rng.uniform(2.4, 2.8)
            parts.append(add_mesh_cylinder("fuselage", 0.22, fuse_len, (0.0, 0.0, 0.0), rotation=(math.radians(90), 0, 0), material=mat_white, vertices=20))
            # Nose cone
            parts.append(add_mesh_sphere("nose_cone", (0.21, 0.45, 0.20), (0.0, fuse_len * 0.5 + 0.15, 0.0), material=mat_white))
            # Cockpit windscreen
            parts.append(add_mesh_cube("cockpit_glass", (0.24, 0.25, 0.10), (0.0, fuse_len * 0.45, 0.14), mat_glass))
            # Wing sweep
            wing_span = self.rng.uniform(2.8, 3.4)
            parts.append(add_mesh_cube("wings", (wing_span, 0.46, 0.06), (0.0, 0.05, -0.06), mat_metal))
            # Winglets
            for side, x in (("left", -wing_span * 0.5), ("right", wing_span * 0.5)):
                parts.append(add_mesh_cube(f"winglet_{side}", (0.03, 0.16, 0.18), (x, 0.05, 0.04), mat_stripe))
            # Twin engine nacelles under wings
            for x in (-wing_span * 0.25, wing_span * 0.25):
                parts.append(add_mesh_cylinder("jet_engine", 0.11, 0.52, (x, 0.15, -0.18), rotation=(math.radians(90), 0, 0), material=mat_metal))
            # Tail section: horizontal stabilizer and vertical tail fin
            tail_y = -fuse_len * 0.46
            parts.append(add_mesh_cube("tail_h", (1.1, 0.22, 0.04), (0.0, tail_y, 0.08), mat_white))
            parts.append(add_mesh_cube("tail_v", (0.05, 0.38, 0.48), (0.0, tail_y, 0.32), mat_stripe))

        elif st == 1:  # Single-Engine Propeller Utility / Trainer Plane (Cessna style)
            fuse_len = 2.0
            parts.append(add_mesh_cylinder("fuselage_prop", 0.18, fuse_len, (0.0, 0.0, 0.0), rotation=(math.radians(90), 0, 0), material=mat_white, vertices=16))
            parts.append(add_mesh_sphere("prop_spinner", (0.08, 0.18, 0.08), (0.0, fuse_len * 0.5 + 0.08, 0.0), mat_stripe))
            # Propeller blades
            parts.append(add_mesh_cube("prop_blade", (0.55, 0.03, 0.06), (0.0, fuse_len * 0.5 + 0.05, 0.0), self.materials["matte_black"]))
            # High wing
            wing_span = 2.7
            parts.append(add_mesh_cube("high_wing", (wing_span, 0.42, 0.05), (0.0, 0.12, 0.19), mat_stripe))
            # Wing struts
            for x in (-0.65, 0.65):
                parts.append(add_mesh_cube(f"strut_{x}", (0.025, 0.025, 0.32), (x, 0.12, 0.02), mat_metal))
            # Cockpit cabin windows
            parts.append(add_mesh_cube("cabin_glass", (0.24, 0.42, 0.16), (0.0, 0.25, 0.08), mat_glass))
            # Tail assembly
            tail_y = -fuse_len * 0.46
            parts.append(add_mesh_cube("tail_h", (0.85, 0.18, 0.035), (0.0, tail_y, 0.04), mat_white))
            parts.append(add_mesh_cube("tail_v", (0.04, 0.32, 0.36), (0.0, tail_y, 0.22), mat_stripe))

        else:  # Twin-Turboprop Commuter
            fuse_len = 2.6
            parts.append(add_mesh_cylinder("fuselage_turbo", 0.20, fuse_len, (0.0, 0.0, 0.0), rotation=(math.radians(90), 0, 0), material=mat_white, vertices=18))
            parts.append(add_mesh_sphere("nose_cone", (0.19, 0.38, 0.18), (0.0, fuse_len * 0.5 + 0.12, 0.0), mat_white))
            parts.append(add_mesh_cube("cockpit_glass", (0.22, 0.22, 0.09), (0.0, fuse_len * 0.42, 0.13), mat_glass))
            wing_span = 3.2
            parts.append(add_mesh_cube("wing_main", (wing_span, 0.44, 0.055), (0.0, 0.08, 0.10), mat_stripe))
            # Twin turboprop engines with spinning prop discs
            for x in (-wing_span * 0.26, wing_span * 0.26):
                parts.append(add_mesh_cylinder("turbo_nacelle", 0.09, 0.58, (x, 0.20, 0.10), rotation=(math.radians(90), 0, 0), material=mat_metal))
                parts.append(add_mesh_cylinder("prop_disc", 0.32, 0.015, (x, 0.52, 0.10), rotation=(math.radians(90), 0, 0), material=self.materials["rotor_blur"], vertices=16))
            # T-Tail
            tail_y = -fuse_len * 0.47
            parts.append(add_mesh_cube("fin_v", (0.05, 0.36, 0.50), (0.0, tail_y, 0.34), mat_white))
            parts.append(add_mesh_cube("tail_t_top", (0.95, 0.20, 0.04), (0.0, tail_y - 0.04, 0.58), mat_stripe))

        return join_components("synthetic_airplane", parts, pass_index=1)

    # -------------------------------------------------------------
    # CLASS 2: BIRD (Avian Wildlife)
    # -------------------------------------------------------------
    def create_bird(self, subtype: int | None = None) -> bpy.types.Object:
        """Builds volumetric 3D bird geometry with body, head, beak, and angled wings."""
        st = subtype if subtype is not None else self.rng.choice([0, 1, 2])
        color_theme = self.rng.choice(["bird_slate", "bird_brown", "bird_white"])
        mat_feathers = self.materials[color_theme]
        mat_beak = self.materials["bird_beak"]
        mat_secondary = self.materials["bird_slate"] if color_theme != "bird_slate" else self.materials["bird_brown"]

        parts = []
        # Volumetric streamlined bird torso
        body_len = self.rng.uniform(0.55, 0.75)
        torso = add_mesh_sphere("bird_torso", (0.16, body_len * 0.5, 0.14), (0.0, 0.0, 0.0), mat_feathers)
        parts.append(torso)

        # Bird Head and Beak
        head_y = body_len * 0.48
        head = add_mesh_sphere("bird_head", (0.11, 0.13, 0.11), (0.0, head_y, 0.06), mat_feathers)
        parts.append(head)
        beak = add_mesh_cylinder("bird_beak", 0.035, 0.16, (0.0, head_y + 0.15, 0.04), rotation=(math.radians(90), 0, 0), material=mat_beak, vertices=8)
        parts.append(beak)

        # Tail fan
        tail_len = self.rng.uniform(0.35, 0.50)
        tail = add_mesh_cube("bird_tail", (0.28, tail_len, 0.025), (0.0, -body_len * 0.55, -0.02), mat_secondary)
        parts.append(tail)

        # Wings (Left & Right) with natural dihedral and sweep
        wing_span = self.rng.uniform(1.35, 1.85)
        wing_chord = self.rng.uniform(0.28, 0.38)
        dihedral_angle = self.rng.uniform(-10.0, 22.0) if st != 0 else 5.0  # Gliding vs flapping

        # Left wing segments (inner and outer tip)
        parts.append(add_mesh_cube("wing_inner_L", (wing_span * 0.28, wing_chord, 0.03), (-wing_span * 0.20, 0.04, 0.05), mat_feathers))
        parts.append(add_mesh_cube("wing_tip_L", (wing_span * 0.24, wing_chord * 0.65, 0.02), (-wing_span * 0.42, 0.0, 0.05 + math.sin(math.radians(dihedral_angle)) * 0.15), mat_secondary))

        # Right wing segments
        parts.append(add_mesh_cube("wing_inner_R", (wing_span * 0.28, wing_chord, 0.03), (wing_span * 0.20, 0.04, 0.05), mat_feathers))
        parts.append(add_mesh_cube("wing_tip_R", (wing_span * 0.24, wing_chord * 0.65, 0.02), (wing_span * 0.42, 0.0, 0.05 + math.sin(math.radians(dihedral_angle)) * 0.15), mat_secondary))

        return join_components("synthetic_bird", parts, pass_index=2)

    # -------------------------------------------------------------
    # CLASS 3: BALLOON (Civilian Hot-Air / Weather Balloon)
    # -------------------------------------------------------------
    def create_balloon(self, subtype: int | None = None) -> bpy.types.Object:
        """Builds a civilian hot-air balloon or high-altitude weather balloon."""
        st = subtype if subtype is not None else self.rng.choice([0, 1])
        color_choices = ["balloon_yellow", "balloon_cyan", "balloon_magenta", "aviation_red", "safety_orange"]
        col1 = self.rng.choice(color_choices)
        col2 = self.rng.choice([c for c in color_choices if c != col1])
        mat_env1 = self.materials[col1]
        mat_env2 = self.materials[col2]
        mat_basket = self.materials["basket_wicker"]
        mat_metal = self.materials["aluminum"]

        parts = []
        if st == 0:  # Hot-Air Balloon with teardrop envelope, burner, cables, & wicker basket
            env_scale = self.rng.uniform(1.0, 1.3)
            # Main upper sphere of envelope
            parts.append(add_mesh_sphere("balloon_dome", (0.85 * env_scale, 0.85 * env_scale, 0.95 * env_scale), (0.0, 0.0, 0.70 * env_scale), mat_env1, segments=24, rings=16))
            # Lower tapered skirt / cone
            parts.append(add_mesh_cylinder("balloon_skirt", 0.50 * env_scale, 0.55 * env_scale, (0.0, 0.0, 0.05 * env_scale), material=mat_env2, vertices=20))
            # Accent equator belt
            parts.append(add_mesh_cylinder("balloon_belt", 0.87 * env_scale, 0.12 * env_scale, (0.0, 0.0, 0.65 * env_scale), material=mat_env2, vertices=24))
            # Burner ring
            parts.append(add_mesh_cylinder("burner", 0.12, 0.08, (0.0, 0.0, -0.32 * env_scale), material=mat_metal))
            # Rigging cables
            for x, y in [(-0.16, -0.16), (0.16, -0.16), (-0.16, 0.16), (0.16, 0.16)]:
                parts.append(add_mesh_cube("cable", (0.015, 0.015, 0.38 * env_scale), (x * env_scale, y * env_scale, -0.42 * env_scale), mat_metal))
            # Basket
            parts.append(add_mesh_cube("basket", (0.32 * env_scale, 0.32 * env_scale, 0.30 * env_scale), (0.0, 0.0, -0.72 * env_scale), mat_basket))

        else:  # High-Altitude Scientific / Weather Balloon
            balloon_rad = self.rng.uniform(0.9, 1.2)
            parts.append(add_mesh_sphere("weather_balloon", (balloon_rad, balloon_rad, balloon_rad * 1.05), (0.0, 0.0, 0.6), self.materials["aviation_white"], segments=20, rings=16))
            # Sensor payload line
            parts.append(add_mesh_cylinder("tether", 0.012, 0.85, (0.0, 0.0, -0.1), material=mat_metal))
            # Sensor package / radiosonde box
            parts.append(add_mesh_cube("radiosonde", (0.18, 0.16, 0.14), (0.0, 0.0, -0.65), self.materials["safety_orange"]))
            # Miniature antenna
            parts.append(add_mesh_cylinder("antenna", 0.008, 0.22, (0.05, 0.0, -0.82), material=mat_metal))

        return join_components("synthetic_balloon", parts, pass_index=3)

    def get_asset_for_class(self, class_id: int) -> bpy.types.Object:
        """Instantiates a fresh procedural model for the requested canonical class ID."""
        if class_id == 0:
            return self.create_drone()
        elif class_id == 1:
            return self.create_airplane()
        elif class_id == 2:
            return self.create_bird()
        elif class_id == 3:
            return self.create_balloon()
        else:
            raise ValueError(f"Unknown canonical class ID: {class_id}")
