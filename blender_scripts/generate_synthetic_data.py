"""AeroWatch AI — Unified Advanced Multi-Drone, Warplane & Aerial Scene Generator.

Blender 5.2.2 LTS compatible.
Generates realistic outdoor scenes with procedural city/town, layered mountain ranges,
procedural sky/clouds, and diverse aerial objects:
  - Drones: Multirotors (quad/hex), FPV strike drones, Twin-boom reconnaissance UAVs, Heavy hexacopters
  - Kamikaze / Loitering Drones (طائرات مسيرة انتحارية / جوالة): Delta-wing pusher-prop UAVs (Shahed/Harop layout)
  - Warplanes (طائرات مقاتلة حربية): Twin-tail air superiority jets, Delta-canard multirole fighters, Stealth fighters, Swept-wing jets
  - Civilian Aircraft: High-wing trainer, Low-wing prop, Twin-turboprop, Commercial jet airliner
  - Avian Wildlife: Seagulls, Hawks
  - Balloons: Hot-air balloons, Scientific weather balloons

Includes ray-cast line-of-sight visibility checks, realistic real-world scaling,
dynamic camera tracking, YOLO bounding box annotations, and contact sheet generation.

Run from Blender GUI (Scripting workspace -> Run Script) or PowerShell:
  & "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe" --background --factory-startup --python .\\blender_scripts\\generate_synthetic_data.py -- --samples 5 --quality draft --resolution 640x360
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import bpy
import bmesh
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector, noise

# ---------------- Canonical Class Mappings ----------------
# By default, kamikaze/loitering delta drones are labeled as 'drone' (or 'kamikaze_drone' if --separate-kamikaze is active)
DEFAULT_CLASSES = ["drone", "aircraft", "bird", "balloon", "military_jet"]

DRONE_VARIANTS = (
    "survey_quad", "night_quad", "field_hex", "coast_quad", "rescue_quad",
    "fpv_strike", "twin_boom_recon", "heavy_hex", "delta_kamikaze"
)
WARPLANE_VARIANTS = (
    "fighter_twin_tail", "fighter_delta_canard", "fighter_stealth", "fighter_swept"
)

CFG = {
    "samples": 4,
    "seed": 20261010,
    "quality": "draft",
    "width": 1280,
    "height": 720,
    "output": "",
    "save_blend": True,
    "separate_kamikaze": False,
}

CLASS_NAMES = list(DEFAULT_CLASSES)
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

SCENE = None
CITY_ZONE = None
CITY_FOOTPRINTS = []
CITY_ROAD_LINES = []
CITY_TOPS = []
CURRENT_PRESET = {}
WORLD_RNG = random.Random(1)
AW_HAZE = True

HAZE_START = 40.0
HAZE_FAR_BOOST = 300.0
MIN_BOX_W, MIN_BOX_H, MIN_VISIBLE = 0.020, 0.010, 0.55

MIN_Z = {"drone": 3.0, "bird": 3.5, "balloon": 4.0, "aircraft": 9.0, "military_jet": 10.0, "kamikaze_drone": 4.0}
SEPARATION = {"drone": 1.6, "bird": 1.2, "balloon": 4.0, "aircraft": 14.0, "military_jet": 16.0, "kamikaze_drone": 2.5}
DIST_RANGE = {
    "drone": (8.0, 32.0),
    "kamikaze_drone": (12.0, 48.0),
    "bird": (7.0, 26.0),
    "balloon": (14.0, 48.0),
    "aircraft": (65.0, 140.0),
    "military_jet": (60.0, 140.0),
}


def parse_args():
    argv = sys.argv
    extra = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(prog="AeroWatch")
    parser.add_argument("--samples", type=int, default=4, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=20261010, help="Random seed")
    parser.add_argument("--quality", choices=("draft", "high"), default="draft", help="Render quality")
    parser.add_argument("--resolution", default="1280x720", help="Resolution WxH (e.g. 640x360 or 1280x720)")
    parser.add_argument("--output", default="", help="Output directory")
    parser.add_argument("--no-save-blend", action="store_true", help="Do not save .blend scene file")
    parser.add_argument("--separate-kamikaze", action="store_true", help="Label kamikaze/loitering drones as separate class")
    args = parser.parse_args(extra)

    if args.samples < 1:
        parser.error("--samples must be >= 1")
    try:
        w_txt, h_txt = args.resolution.lower().split("x")
        width, height = int(w_txt), int(h_txt)
    except ValueError:
        parser.error("--resolution must look like 640x360 or 1280x720")
        return

    CFG["samples"] = args.samples
    CFG["seed"] = args.seed
    CFG["quality"] = args.quality
    CFG["width"] = width
    CFG["height"] = height
    CFG["output"] = args.output
    CFG["save_blend"] = not args.no_save_blend
    CFG["separate_kamikaze"] = args.separate_kamikaze

    global CLASS_NAMES, CLASS_TO_ID
    CLASS_NAMES = list(DEFAULT_CLASSES)
    if args.separate_kamikaze and "kamikaze_drone" not in CLASS_NAMES:
        CLASS_NAMES.append("kamikaze_drone")
    CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def resolve_project_root() -> Path:
    # 1. Check parent of this script file
    script_dir = Path(__file__).resolve().parent
    if (script_dir.parent / "data").exists():
        return script_dir.parent

    # 2. Check current working directory
    cwd = Path.cwd()
    if (cwd / "data").exists():
        return cwd

    # 3. Check home directory
    home = Path.home()
    for candidate in [home / "AeroWatchAI", home / "Desktop" / "AeroWatchAI"]:
        if candidate.exists():
            return candidate

    return cwd


def prepare_output_dir(project_root: Path) -> Path:
    import shutil
    if CFG["output"]:
        requested = Path(CFG["output"]).expanduser().resolve()
        if requested.exists():
            shutil.rmtree(requested)
        requested.mkdir(parents=True, exist_ok=True)
        return requested
    data_root = project_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    target = data_root / "synthetic_v4"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def assign_split(index: int, total: int) -> str:
    if total < 3:
        return "train"
    if total >= 10:
        if index % 10 == 0:
            return "test"
        elif index % 10 in (3, 7):
            return "val"
        return "train"
    if index == total - 1:
        return "test"
    if index == total - 2 or (total >= 8 and index % 5 == 0):
        return "val"
    return "train"


def make_plan(total: int):
    plan_order = ["drone", "military_jet", "aircraft", "drone", "bird", "balloon"]
    if CFG["separate_kamikaze"]:
        plan_order.insert(2, "kamikaze_drone")
    return [(plan_order[i % len(plan_order)], assign_split(i, total)) for i in range(total)]


def active_scene():
    global SCENE
    return SCENE or bpy.context.scene


def clear_scene(scene):
    scene.camera = None
    scene.world = None
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for datablocks in (
        bpy.data.meshes, bpy.data.curves, bpy.data.cameras,
        bpy.data.lights, bpy.data.materials, bpy.data.worlds,
    ):
        for block in list(datablocks):
            if block.users == 0:
                datablocks.remove(block)


def link_object(obj, scene=None):
    (scene or active_scene()).collection.objects.link(obj)
    return obj


def set_if_present(node, socket_name, value):
    socket = node.inputs.get(socket_name)
    if socket is not None:
        socket.default_value = value


def smooth(obj):
    mesh = obj.data
    try:
        mesh.shade_smooth()
    except Exception:
        for poly in mesh.polygons:
            poly.use_smooth = True


def apply_scale(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


def delete_hierarchy(root):
    children = list(root.children_recursive)
    for obj in children:
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.objects.remove(root, do_unlink=True)


# ---------------- Materials Library ----------------
def make_material(name, color, metallic=0.0, roughness=0.55, emission=0.0, emission_color=None):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1.0)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    set_if_present(bsdf, "Base Color", (*color, 1.0))
    set_if_present(bsdf, "Metallic", metallic)
    set_if_present(bsdf, "Roughness", roughness)
    if emission > 0.0:
        set_if_present(bsdf, "Emission Color", (*(emission_color or color), 1.0))
        set_if_present(bsdf, "Emission Strength", emission)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def build_materials_library():
    mats = {
        "carbon": make_material("AW_Carbon", (0.06, 0.07, 0.08), 0.35, 0.35),
        "dark": make_material("AW_Dark", (0.03, 0.035, 0.04), 0.25, 0.40),
        "metal": make_material("AW_Metal", (0.55, 0.58, 0.60), 0.80, 0.25),
        "rubber": make_material("AW_Rubber", (0.02, 0.02, 0.02), 0.0, 0.8),
        "glass": make_material("AW_Glass", (0.04, 0.09, 0.14), 0.85, 0.10),
        "glass_gold": make_material("AW_GlassGold", (0.28, 0.22, 0.08), 0.90, 0.12),
        "basket": make_material("AW_Basket", (0.35, 0.22, 0.12), 0.0, 0.85),
        "beak": make_material("AW_Beak", (0.92, 0.62, 0.10), 0.05, 0.4),
        "gull": make_material("AW_Gull", (0.88, 0.89, 0.88), 0.0, 0.6),
        "gull_tip": make_material("AW_GullTip", (0.16, 0.18, 0.20), 0.0, 0.6),
        "hawk": make_material("AW_Hawk", (0.32, 0.22, 0.14), 0.0, 0.7),
        "led_red": make_material("AW_LedRed", (1.0, 0.05, 0.02), 0.0, 0.3, emission=8.0),
        "led_green": make_material("AW_LedGreen", (0.05, 1.0, 0.15), 0.0, 0.3, emission=8.0),
        "cloud": make_material("AW_Cloud", (0.94, 0.96, 0.98), 0.0, 0.95),
        # Tactical & Camo Materials
        "camo_desert": make_material("AW_CamoDesert", (0.58, 0.48, 0.34), 0.10, 0.65),
        "camo_olive": make_material("AW_CamoOlive", (0.22, 0.28, 0.16), 0.10, 0.65),
        "ram_charcoal": make_material("AW_RamCharcoal", (0.08, 0.085, 0.09), 0.30, 0.40),
        "jet_ghost_grey": make_material("AW_JetGhost", (0.35, 0.38, 0.42), 0.25, 0.38),
        "jet_dark_grey": make_material("AW_JetDark", (0.16, 0.18, 0.20), 0.30, 0.36),
        "jet_navy": make_material("AW_JetNavy", (0.10, 0.15, 0.24), 0.25, 0.36),
        "fpv_neon_yellow": make_material("AW_FpvYellow", (0.95, 0.85, 0.05), 0.05, 0.4),
        "fpv_neon_cyan": make_material("AW_FpvCyan", (0.05, 0.85, 0.95), 0.05, 0.4),
    }

    # Drone paint themes
    paints = {
        "survey_quad": {"body": mats["metal"], "arm": make_material("AW_SurvArm", (0.92, 0.32, 0.05), 0.15, 0.35), "motor": mats["dark"]},
        "night_quad": {"body": mats["ram_charcoal"], "arm": mats["carbon"], "motor": mats["dark"]},
        "field_hex": {"body": mats["camo_olive"], "arm": mats["carbon"], "motor": mats["metal"]},
        "coast_quad": {"body": mats["jet_ghost_grey"], "arm": make_material("AW_CoastBlue", (0.05, 0.30, 0.70), 0.2, 0.35), "motor": mats["metal"]},
        "rescue_quad": {"body": make_material("AW_RescRed", (0.85, 0.08, 0.06), 0.15, 0.35), "arm": mats["metal"], "motor": mats["dark"]},
        "fpv_strike": {"body": mats["carbon"], "arm": mats["fpv_neon_yellow"], "motor": mats["dark"]},
        "twin_boom_recon": {"body": mats["jet_ghost_grey"], "arm": mats["carbon"], "motor": mats["metal"]},
        "heavy_hex": {"body": mats["dark"], "arm": mats["carbon"], "motor": mats["metal"]},
        "delta_kamikaze": {"body": mats["camo_desert"], "arm": mats["ram_charcoal"], "motor": mats["metal"]},
    }

    liveries = {
        "blue": {"fuselage": make_material("AW_AcWhite", (0.85, 0.87, 0.90), 0.15, 0.32), "stripe": make_material("AW_AcBlue", (0.05, 0.28, 0.78), 0.15, 0.32), "tail": mats["metal"]},
        "green": {"fuselage": make_material("AW_AcCream", (0.85, 0.82, 0.72), 0.10, 0.38), "stripe": mats["camo_olive"], "tail": mats["dark"]},
        "orange": {"fuselage": mats["jet_ghost_grey"], "stripe": make_material("AW_AcOrange", (0.92, 0.38, 0.05), 0.12, 0.34), "tail": mats["ram_charcoal"]},
    }

    balloons = {
        "red": make_material("AW_BalRed", (0.86, 0.08, 0.06), 0.02, 0.28),
        "yellow": make_material("AW_BalYellow", (0.95, 0.78, 0.08), 0.02, 0.28),
        "teal": make_material("AW_BalTeal", (0.05, 0.55, 0.60), 0.02, 0.28),
        "orange": make_material("AW_BalOrange", (0.94, 0.36, 0.05), 0.02, 0.28),
        "white": make_material("AW_BalWhite", (0.92, 0.93, 0.95), 0.02, 0.30),
    }

    mats["paints"] = paints
    mats["liveries"] = liveries
    mats["balloons"] = balloons
    return mats


# ---------------- Primitive 3D Helpers ----------------
def add_cube(name, dimensions, location, material, bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    apply_scale(obj)
    if material:
        obj.data.materials.append(material)
    if bevel > 0:
        mod = obj.modifiers.new("Bevel", "BEVEL")
        mod.width = bevel
        mod.segments = 2
        mod.limit_method = "ANGLE"
    obj["aw_target_part"] = True
    return obj


def add_uv_sphere(name, scale, location, material, segments=20, rings=12):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings, radius=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    apply_scale(obj)
    if material:
        obj.data.materials.append(material)
    smooth(obj)
    obj["aw_target_part"] = True
    return obj


def add_cylinder(name, radius, depth, location, material, vertices=16, rotation=(0.0, 0.0, 0.0)):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = rotation
    if material:
        obj.data.materials.append(material)
    smooth(obj)
    obj["aw_target_part"] = True
    return obj


def add_rod_between(name, start, end, radius, material, vertices=10):
    a, b = Vector(start), Vector(end)
    direction = b - a
    depth = max(direction.length, 0.001)
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=(a + b) / 2)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    if material:
        obj.data.materials.append(material)
    smooth(obj)
    obj["aw_target_part"] = True
    return obj


def parent_target(parts, kind, location, rotation, scale, variant="", role="hero"):
    root = bpy.data.objects.new("AW_TARGET_" + kind, None)
    link_object(root)
    root.empty_display_type = "PLAIN_AXES"
    root.empty_display_size = 0.45
    for part in parts:
        part.parent = root
        part.matrix_parent_inverse = root.matrix_world.inverted()
        part["aw_target_part"] = True
    root.location = location
    root.rotation_euler = rotation
    root.scale = (scale, scale, scale)
    root["aw_target_class"] = kind
    root["aw_variant"] = variant
    root["aw_role"] = role
    return root


# =====================================================================
# 3D PROCEDURAL ASSET BUILDERS: DRONES & KAMIKAZE / LOITERING UAVS
# =====================================================================

def build_multirotor(mats, rng, location, rotation, scale, variant, role="hero"):
    """Builds standard multirotor UAVs (quadcopters and hexacopters)."""
    paint = mats["paints"].get(variant, mats["paints"]["survey_quad"])
    parts = []
    is_hex = "hex" in variant
    count = 6 if is_hex else 4
    reach = 1.05 if is_hex else 0.95
    body_dim = (0.55, 0.65, 0.18)

    parts.append(add_uv_sphere(variant + " shell", (body_dim[0] * 0.48, body_dim[1] * 0.46, 0.12), (0, 0, 0.06), paint["body"], 24, 14))
    parts.append(add_cube(variant + " belly", (body_dim[0] * 0.75, body_dim[1] * 0.60, 0.08), (0, 0.02, -0.02), mats["dark"], bevel=0.02))
    parts.append(add_cube(variant + " deck", (body_dim[0] * 0.48, body_dim[1] * 0.35, 0.06), (0, -0.02, 0.15), paint["arm"], bevel=0.012))

    for index in range(count):
        angle = math.radians(45 + index * 90) if count == 4 else math.radians(-90 + index * 60)
        x, y = math.cos(angle) * reach, math.sin(angle) * reach
        parts.append(add_rod_between("%s arm %d" % (variant, index), (x * 0.2, y * 0.2, 0.04), (x, y, 0.05), 0.052, paint["arm"]))
        parts.append(add_cylinder("%s motor %d" % (variant, index), 0.09, 0.11, (x, y, 0.10), paint["motor"]))
        spin = math.radians(rng.uniform(0.0, 180.0))
        blade_a = add_cube("%s propA %d" % (variant, index), (reach * 0.95, 0.10, 0.025), (x, y, 0.19), mats["rubber"])
        blade_a.rotation_euler[2] = spin
        blade_b = add_cube("%s propB %d" % (variant, index), (reach * 0.95, 0.10, 0.025), (x, y, 0.20), mats["carbon"])
        blade_b.rotation_euler[2] = spin + math.radians(90)
        parts.extend((blade_a, blade_b))

    parts.append(add_cube(variant + " gimbal", (0.18, 0.15, 0.14), (0, -body_dim[1] * 0.60, -0.10), mats["dark"], bevel=0.015))
    parts.append(add_uv_sphere(variant + " lens", (0.055, 0.045, 0.055), (0, -body_dim[1] * 0.75, -0.10), mats["glass"], 16, 10))
    for side, name in ((-1, "L"), (1, "R")):
        parts.append(add_rod_between(variant + " leg " + name, (side * 0.18, -0.18, -0.04), (side * 0.28, -0.02, -0.28), 0.028, mats["metal"]))
        parts.append(add_rod_between(variant + " rail " + name, (side * 0.28, -0.35, -0.28), (side * 0.28, 0.30, -0.28), 0.025, mats["dark"]))

    target_class = "drone"
    return parent_target(parts, target_class, location, rotation, scale, variant, role)


def build_delta_kamikaze_drone(mats, rng, location, rotation, scale, role="hero", livery=None):
    """Delta-wing pusher loitering / kamikaze UAV (Shahed-136 / Harop silhouette).

    Triangular delta wings with tip winglets, blended center fuselage, and rear pusher propeller.
    """
    body_mat = mats["camo_desert"] if livery == "desert" else (mats["camo_olive"] if livery == "olive" else mats["ram_charcoal"])
    parts = []

    # 1. Main Delta Wing (swept triangular wing)
    wing_span = 2.6
    wing_chord = 3.2
    # Left and Right delta wing halves
    for side in (-1, 1):
        wing = add_cube("delta_wing", (wing_span * 0.48, wing_chord * 0.50, 0.08), (side * wing_span * 0.26, 0.20, 0.0), body_mat, bevel=0.02)
        wing.rotation_euler[2] = math.radians(side * 24)
        parts.append(wing)
        # Vertical stabilizer endplate at wingtip
        tip_fin = add_cube("delta_tip_fin", (0.05, 0.65, 0.75), (side * (wing_span * 0.52), wing_chord * 0.38, 0.15), body_mat, bevel=0.01)
        parts.append(tip_fin)

    # 2. Central Fuselage Body
    parts.append(add_uv_sphere("delta_fuselage", (0.28, 1.45, 0.22), (0.0, -0.10, 0.02), body_mat, 22, 12))
    # Pointed / ogival nose cone
    parts.append(add_uv_sphere("delta_nose", (0.18, 0.55, 0.16), (0.0, -1.55, 0.01), body_mat, 18, 10))
    # Front optical sensor port / pitot tube
    parts.append(add_cylinder("pitot_probe", 0.015, 0.35, (0.0, -2.15, 0.0), mats["metal"], rotation=(math.radians(90), 0, 0)))

    # 3. Rear Pusher Engine Nacelle & Propeller
    engine_y = 1.35
    parts.append(add_cylinder("pusher_engine", 0.14, 0.40, (0.0, engine_y, 0.02), mats["dark"], rotation=(math.radians(90), 0, 0)))
    parts.append(add_uv_sphere("pusher_spinner", (0.07, 0.12, 0.07), (0.0, engine_y + 0.25, 0.02), mats["metal"], 14, 8))
    # 2-blade pusher propeller
    spin = math.radians(rng.uniform(0.0, 180.0))
    blade = add_cube("pusher_blade", (0.85, 0.06, 0.025), (0.0, engine_y + 0.22, 0.02), mats["carbon"])
    blade.rotation_euler[1] = spin
    parts.append(blade)

    target_class = "kamikaze_drone" if CFG["separate_kamikaze"] else "drone"
    return parent_target(parts, target_class, location, rotation, scale, "delta_kamikaze", role)


def build_twin_boom_drone(mats, rng, location, rotation, scale, role="hero"):
    """Twin-boom pusher reconnaissance / surveillance UAV (Bayraktar / RQ-7 style)."""
    body_mat = mats["jet_ghost_grey"]
    parts = []

    # 1. High aspect ratio main wing
    span = 4.8
    parts.append(add_cube("tb_wing", (span, 0.52, 0.065), (0.0, -0.10, 0.22), body_mat, bevel=0.015))

    # 2. Central Pod Fuselage
    parts.append(add_uv_sphere("tb_pod", (0.24, 0.95, 0.25), (0.0, -0.45, 0.08), body_mat, 24, 14))
    # Underslung EO/IR Gimbal Turret under nose
    parts.append(add_uv_sphere("tb_gimbal", (0.13, 0.13, 0.13), (0.0, -1.05, -0.10), mats["dark"], 18, 10))
    parts.append(add_uv_sphere("tb_lens", (0.07, 0.05, 0.07), (0.0, -1.18, -0.10), mats["glass"], 14, 8))

    # 3. Twin Tail Booms & Inverted V-Tail
    boom_span = 1.35
    boom_len = 2.4
    for side in (-1, 1):
        bx = side * (boom_span * 0.5)
        parts.append(add_rod_between(f"tb_boom_{side}", (bx, -0.10, 0.20), (bx, boom_len, 0.20), 0.035, mats["carbon"]))
        # Inverted V-tail surface
        v_tail = add_cube(f"v_tail_{side}", (0.55, 0.24, 0.04), (bx * 0.72, boom_len + 0.05, 0.05), body_mat, bevel=0.01)
        v_tail.rotation_euler[1] = math.radians(side * 36)
        parts.append(v_tail)

    # 4. Rear Pusher Propeller
    prop_y = 0.55
    parts.append(add_cylinder("tb_motor", 0.11, 0.28, (0.0, prop_y, 0.12), mats["dark"], rotation=(math.radians(90), 0, 0)))
    prop_blade = add_cube("tb_prop", (0.80, 0.05, 0.02), (0.0, prop_y + 0.16, 0.12), mats["carbon"])
    prop_blade.rotation_euler[1] = math.radians(rng.uniform(0.0, 180.0))
    parts.append(prop_blade)

    # 5. Lightweight Landing Gear Legs
    for side in (-1, 1):
        parts.append(add_rod_between(f"tb_gear_{side}", (side * 0.18, -0.40, 0.0), (side * 0.38, -0.35, -0.35), 0.025, mats["metal"]))
        parts.append(add_cylinder(f"tb_wheel_{side}", 0.07, 0.04, (side * 0.40, -0.35, -0.38), mats["rubber"], rotation=(0, math.radians(90), 0)))
    parts.append(add_rod_between("tb_nose_gear", (0.0, -0.90, 0.0), (0.0, -0.85, -0.32), 0.022, mats["metal"]))
    parts.append(add_cylinder("tb_nose_wheel", 0.06, 0.035, (0.0, -0.85, -0.35), mats["rubber"], rotation=(0, math.radians(90), 0)))

    target_class = "drone"
    return parent_target(parts, target_class, location, rotation, scale, "twin_boom_recon", role)


def build_fpv_drone(mats, rng, location, rotation, scale, role="hero"):
    """Compact high-speed FPV racing / strike quad drone."""
    parts = []
    frame_mat = mats["carbon"]
    prop_mat = rng.choice([mats["fpv_neon_yellow"], mats["fpv_neon_cyan"], mats["led_red"]])

    # 1. Carbon fiber X-frame plates
    parts.append(add_cube("fpv_top_plate", (0.16, 0.42, 0.015), (0.0, 0.0, 0.04), frame_mat))
    parts.append(add_cube("fpv_bot_plate", (0.18, 0.45, 0.02), (0.0, 0.0, -0.02), frame_mat))
    # Top-mounted LiPo Battery
    parts.append(add_cube("fpv_battery", (0.14, 0.28, 0.10), (0.0, 0.04, 0.10), mats["dark"]))

    # 2. Four Diagonal Arms & High-RPM Motors
    arm_reach = 0.52
    for x, y, rot_deg in [(-arm_reach, arm_reach * 0.85, -35), (arm_reach, arm_reach * 0.85, 35),
                          (-arm_reach, -arm_reach * 0.85, -145), (arm_reach, -arm_reach * 0.85, 145)]:
        parts.append(add_rod_between("fpv_arm", (0.0, 0.0, 0.0), (x, y, 0.01), 0.028, frame_mat))
        parts.append(add_cylinder("fpv_motor", 0.065, 0.06, (x, y, 0.04), mats["metal"]))
        # Tri-blade props
        spin = math.radians(rng.uniform(0.0, 120.0))
        for a in (0, 120, 240):
            bl = add_cube("fpv_blade", (0.24, 0.04, 0.012), (x, y, 0.08), prop_mat)
            bl.rotation_euler[2] = spin + math.radians(a)
            parts.append(bl)

    # 3. High-angle FPV camera tilted upwards
    cam_box = add_cube("fpv_cam", (0.08, 0.08, 0.08), (0.0, -0.20, 0.04), mats["metal"])
    cam_box.rotation_euler[0] = math.radians(-38)
    parts.append(cam_box)
    parts.append(add_uv_sphere("fpv_lens", (0.03, 0.02, 0.03), (0.0, -0.25, 0.06), mats["glass"], 12, 8))

    # 4. Rear Lollipop Antenna
    parts.append(add_rod_between("ant_stalk", (0.0, 0.20, 0.05), (0.0, 0.28, 0.22), 0.012, mats["rubber"]))
    parts.append(add_uv_sphere("ant_dome", (0.04, 0.04, 0.04), (0.0, 0.28, 0.24), mats["led_red"], 12, 8))

    target_class = "drone"
    return parent_target(parts, target_class, location, rotation, scale, "fpv_strike", role)


# =====================================================================
# 3D PROCEDURAL ASSET BUILDERS: WARPLANES & FIGHTER JETS
# =====================================================================

def create_twin_tail_fighter(mats, rng, location, rotation, scale, role="hero", livery=0):
    """Air-Superiority Twin-Tail Twin-Engine Fighter (F-15 / Su-27 / MiG-29 silhouette)."""
    body_mat = mats["jet_ghost_grey"] if livery == 0 else (mats["jet_dark_grey"] if livery == 1 else mats["jet_navy"])
    parts = []

    # 1. Elongated Chined Fuselage
    parts.append(add_uv_sphere("tt_nose", (0.35, 2.8, 0.35), (0.0, -6.8, 0.04), body_mat, 20, 12))
    parts.append(add_uv_sphere("tt_fore", (0.95, 3.4, 0.75), (0.0, -3.2, 0.02), body_mat, 24, 14))
    parts.append(add_uv_sphere("tt_mid", (1.30, 4.2, 0.82), (0.0, 0.8, -0.02), body_mat, 26, 14))
    parts.append(add_uv_sphere("tt_aft", (1.20, 3.2, 0.72), (0.0, 4.4, 0.0), body_mat, 24, 12))

    # 2. Bubble Canopy
    parts.append(add_uv_sphere("tt_canopy", (0.44, 1.85, 0.38), (0.0, -3.6, 0.65), mats["glass_gold"], 20, 12))

    # 3. Swept Cropped-Delta Wings
    wing_span = 9.2
    for side in (-1, 1):
        wing = add_cube(f"tt_wing_{side}", (wing_span * 0.44, 3.6, 0.12), (side * 2.8, 1.2, -0.10), body_mat, bevel=0.02)
        wing.rotation_euler[2] = math.radians(side * 28)
        parts.append(wing)
        # Leading Edge Root Extension (LERX)
        lerx = add_cube(f"tt_lerx_{side}", (0.55, 2.6, 0.08), (side * 1.15, -1.8, 0.02), body_mat)
        lerx.rotation_euler[2] = math.radians(side * 14)
        parts.append(lerx)

    # 4. Twin Vertical Stabilizers (Twin Fins)
    for side in (-1, 1):
        fin = add_cube(f"tt_fin_{side}", (0.12, 2.6, 2.3), (side * 1.05, 4.6, 1.55), body_mat, bevel=0.015)
        fin.rotation_euler[1] = math.radians(side * 6)  # slight outward cant
        parts.append(fin)

    # 5. All-Moving Horizontal Tailplanes
    for side in (-1, 1):
        stab = add_cube(f"tt_stab_{side}", (2.2, 1.9, 0.09), (side * 2.2, 5.8, -0.02), body_mat)
        stab.rotation_euler[2] = math.radians(side * 26)
        parts.append(stab)

    # 6. Twin Engine Nacelles & Afterburner Nozzles
    for side in (-1, 1):
        parts.append(add_cylinder(f"tt_nacelle_{side}", 0.52, 3.6, (side * 0.75, 2.8, -0.15), body_mat, vertices=16, rotation=(math.radians(90), 0, 0)))
        parts.append(add_cylinder(f"tt_nozzle_{side}", 0.46, 1.2, (side * 0.75, 6.2, -0.15), mats["dark"], vertices=16, rotation=(math.radians(90), 0, 0)))
        # Underslung Air Intake
        parts.append(add_cube(f"tt_intake_{side}", (0.75, 2.4, 0.70), (side * 0.75, -0.8, -0.38), mats["dark"], bevel=0.08))

    return parent_target(parts, "military_jet", location, rotation, scale, "fighter_twin_tail", role)


def create_delta_canard_fighter(mats, rng, location, rotation, scale, role="hero", livery=0):
    """Delta-Canard Multirole Jet (Rafale / Eurofighter / Gripen silhouette)."""
    body_mat = mats["jet_ghost_grey"] if livery == 0 else mats["jet_dark_grey"]
    parts = []

    # 1. Sleek Aerodynamic Fuselage
    parts.append(add_uv_sphere("dc_nose", (0.32, 2.5, 0.32), (0.0, -6.0, 0.02), body_mat, 20, 12))
    parts.append(add_uv_sphere("dc_fore", (0.85, 3.2, 0.72), (0.0, -2.8, 0.02), body_mat, 22, 12))
    parts.append(add_uv_sphere("dc_mid", (1.10, 3.8, 0.78), (0.0, 0.9, 0.0), body_mat, 24, 14))
    parts.append(add_uv_sphere("dc_aft", (0.90, 2.6, 0.68), (0.0, 4.0, 0.02), body_mat, 20, 12))
    # Canopy
    parts.append(add_uv_sphere("dc_canopy", (0.38, 1.7, 0.35), (0.0, -3.1, 0.58), mats["glass"], 18, 10))

    # 2. Forward Canard Foreplanes (ahead of main wing)
    for side in (-1, 1):
        canard = add_cube(f"dc_canard_{side}", (1.4, 0.95, 0.07), (side * 1.15, -2.2, 0.28), body_mat, bevel=0.01)
        canard.rotation_euler[2] = math.radians(side * 35)
        canard.rotation_euler[1] = math.radians(-side * 5)
        parts.append(canard)

    # 3. Main Cropped Delta Wings
    wing_span = 8.6
    for side in (-1, 1):
        wing = add_cube(f"dc_delta_{side}", (wing_span * 0.45, 3.8, 0.11), (side * 2.6, 1.6, -0.05), body_mat, bevel=0.02)
        wing.rotation_euler[2] = math.radians(side * 36)
        parts.append(wing)
        # Wingtip missile rail pod
        parts.append(add_cylinder(f"dc_tip_{side}", 0.09, 1.6, (side * (wing_span * 0.50), 1.9, -0.05), mats["dark"], vertices=10, rotation=(math.radians(90), 0, 0)))

    # 4. Single Tall Swept Vertical Fin
    fin = add_cube("dc_fin", (0.12, 2.8, 2.5), (0.0, 3.8, 1.6), body_mat, bevel=0.015)
    parts.append(fin)

    # 5. Engine Nozzles (Twin or Single)
    for side in (-0.42, 0.42):
        parts.append(add_cylinder(f"dc_nozzle_{side}", 0.42, 1.2, (side, 5.8, 0.0), mats["dark"], vertices=16, rotation=(math.radians(90), 0, 0)))

    return parent_target(parts, "military_jet", location, rotation, scale, "fighter_delta_canard", role)


def create_stealth_fighter(mats, rng, location, rotation, scale, role="hero", livery=0):
    """Faceted Stealth Fighter (F-22 / F-35 faceted silhouette)."""
    body_mat = mats["ram_charcoal"]
    parts = []

    # 1. Faceted Chined Fuselage
    parts.append(add_uv_sphere("st_nose", (0.36, 2.6, 0.32), (0.0, -6.2, 0.02), body_mat, 18, 10))
    parts.append(add_uv_sphere("st_mid", (1.25, 4.4, 0.75), (0.0, 0.2, 0.0), body_mat, 24, 12))
    # Golden Radar-Reflective Canopy
    parts.append(add_uv_sphere("st_canopy", (0.42, 1.7, 0.36), (0.0, -3.2, 0.56), mats["glass_gold"], 18, 10))

    # 2. Trapezoidal Diamond Main Wings
    wing_span = 9.4
    for side in (-1, 1):
        wing = add_cube(f"st_wing_{side}", (wing_span * 0.44, 3.5, 0.10), (side * 2.8, 0.8, -0.08), body_mat, bevel=0.02)
        wing.rotation_euler[2] = math.radians(side * 34)
        parts.append(wing)

    # 3. Canted Twin Vertical Tails (tilted 28 deg)
    for side in (-1, 1):
        fin = add_cube(f"st_fin_{side}", (0.12, 2.5, 2.1), (side * 1.35, 4.4, 1.45), body_mat, bevel=0.015)
        fin.rotation_euler[1] = math.radians(side * 26)
        parts.append(fin)

    # 4. Tailplanes & Serrated Exhausts
    for side in (-1, 1):
        stab = add_cube(f"st_stab_{side}", (2.2, 1.8, 0.08), (side * 2.4, 5.5, -0.04), body_mat)
        stab.rotation_euler[2] = math.radians(side * 30)
        parts.append(stab)
        parts.append(add_cube(f"st_nozzle_{side}", (0.75, 1.2, 0.40), (side * 0.65, 5.8, -0.05), mats["dark"], bevel=0.05))

    return parent_target(parts, "military_jet", location, rotation, scale, "fighter_stealth", role)


# =====================================================================
# 3D PROCEDURAL ASSET BUILDERS: CIVILIAN AIRCRAFT, BIRDS, BALLOONS
# =====================================================================

def create_aircraft(mats, rng, location, rotation, scale, role="hero", model="highwing", livery="blue"):
    """Civilian aircraft: high-wing prop, low-wing prop, twin prop, passenger jet."""
    liv = mats["liveries"].get(livery, mats["liveries"]["blue"])
    parts = []
    glass = mats["glass"]
    dark = mats["dark"]
    metal = mats["metal"]
    rubber = mats["rubber"]

    def wing_half(side, span, chord, thick, y, z, dihedral, sweep):
        half = span * 0.5
        wing = add_cube("%s wing" % model, (half * 0.98, chord, thick), (side * half * 0.52, y, z), liv["fuselage"], bevel=0.015)
        wing.rotation_euler[1] = math.radians(-side * dihedral)
        wing.rotation_euler[2] = math.radians(side * sweep)
        parts.append(wing)
        tip = add_cube("%s tip" % model, (0.16, chord * 0.72, thick * 1.15), (side * (half - 0.02), y + sweep * 0.012, z + dihedral * 0.012), liv["tail"])
        parts.append(tip)

    def tail(y, fin_h, stab_span):
        parts.append(add_cube("%s fin" % model, (0.07, 0.55, fin_h), (0, y, fin_h * 0.45), liv["tail"], bevel=0.01))
        parts.append(add_cube("%s stab" % model, (stab_span, 0.32, 0.06), (0, y - 0.02, 0.08), liv["fuselage"], bevel=0.01))

    def prop_at(y, z, x=0.0, length=0.9):
        spin = math.radians(rng.uniform(0.0, 80.0))
        blade_a = add_cube("%s propA" % model, (0.07, 0.03, length), (x, y, z), mats["carbon"])
        blade_a.rotation_euler[1] = spin
        blade_b = add_cube("%s propB" % model, (0.07, 0.03, length), (x, y, z), dark)
        blade_b.rotation_euler[1] = spin + math.radians(90)
        parts.extend((blade_a, blade_b))
        parts.append(add_cylinder("%s spinner" % model, 0.07, 0.12, (x, y + 0.06, z), metal, rotation=(math.radians(90), 0, 0)))

    if model == "highwing":
        parts.append(add_uv_sphere("hw cabin", (0.32, 0.72, 0.34), (0, -0.15, 0.0), liv["fuselage"], 24, 14))
        parts.append(add_uv_sphere("hw nose", (0.20, 0.36, 0.22), (0, -0.95, -0.02), metal, 18, 10))
        parts.append(add_uv_sphere("hw boom", (0.18, 0.78, 0.18), (0, 0.85, 0.04), liv["fuselage"], 20, 10))
        parts.append(add_uv_sphere("hw cockpit", (0.22, 0.32, 0.14), (0, -0.55, 0.22), glass, 16, 10))
        wing_half(-1, 5.2, 0.78, 0.10, -0.05, 0.32, 6.0, 1.5)
        wing_half(1, 5.2, 0.78, 0.10, -0.05, 0.32, 6.0, 1.5)
        for side in (-1, 1):
            parts.append(add_rod_between("hw strut", (side * 0.28, 0.05, -0.22), (side * 1.25, -0.05, 0.28), 0.035, metal))
            parts.append(add_cylinder("hw wheel", 0.11, 0.07, (side * 0.38, -0.05, -0.58), rubber, rotation=(0.0, math.radians(90), 0.0)))
        tail(1.35, 0.62, 1.55)
        prop_at(-1.25, 0.0, length=1.05)
    elif model == "lowwing":
        parts.append(add_uv_sphere("lw cabin", (0.26, 0.85, 0.28), (0, -0.1, 0.05), liv["fuselage"], 24, 14))
        parts.append(add_uv_sphere("lw nose", (0.16, 0.32, 0.17), (0, -1.05, 0.02), metal, 18, 10))
        parts.append(add_uv_sphere("lw boom", (0.15, 0.7, 0.16), (0, 0.78, 0.08), liv["fuselage"], 18, 10))
        parts.append(add_uv_sphere("lw cockpit", (0.18, 0.34, 0.12), (0, -0.48, 0.22), glass, 16, 10))
        wing_half(-1, 4.6, 0.70, 0.09, -0.2, -0.08, 4.0, 4.0)
        wing_half(1, 4.6, 0.70, 0.09, -0.2, -0.08, 4.0, 4.0)
        tail(1.22, 0.5, 1.35)
        prop_at(-1.32, 0.02, length=0.92)
    elif model == "twin":
        parts.append(add_uv_sphere("tw cabin", (0.30, 1.05, 0.32), (0, -0.15, 0.05), liv["fuselage"], 24, 14))
        parts.append(add_uv_sphere("tw nose", (0.20, 0.38, 0.22), (0, -1.2, 0.04), liv["fuselage"], 18, 10))
        parts.append(add_uv_sphere("tw boom", (0.18, 0.85, 0.20), (0, 0.95, 0.08), liv["fuselage"], 18, 10))
        wing_half(-1, 5.8, 0.72, 0.10, -0.05, -0.02, 3.5, 6.0)
        wing_half(1, 5.8, 0.72, 0.10, -0.05, -0.02, 3.5, 6.0)
        for side in (-1, 1):
            parts.append(add_cylinder("tw nacelle", 0.16, 0.72, (side * 0.95, -0.35, -0.08), metal, rotation=(math.radians(90), 0, 0)))
            prop_at(-0.82, -0.08, x=side * 0.95, length=0.7)
        tail(1.5, 0.72, 1.7)
    else:  # Passenger Jet
        parts.append(add_uv_sphere("jet nose", (0.18, 0.5, 0.18), (0, -2.35, 0.02), liv["fuselage"], 20, 10))
        parts.append(add_uv_sphere("jet front", (0.30, 0.95, 0.32), (0, -1.4, 0.02), liv["fuselage"], 24, 12))
        parts.append(add_uv_sphere("jet mid", (0.34, 1.25, 0.36), (0, -0.15, 0.0), liv["fuselage"], 26, 14))
        parts.append(add_uv_sphere("jet rear", (0.26, 0.9, 0.28), (0, 0.95, 0.04), liv["fuselage"], 22, 12))
        parts.append(add_uv_sphere("jet cockpit", (0.22, 0.42, 0.12), (0, -1.85, 0.2), glass, 16, 10))
        wing_half(-1, 6.8, 0.95, 0.11, -0.15, -0.05, 5.0, 18.0)
        wing_half(1, 6.8, 0.95, 0.11, -0.15, -0.05, 5.0, 18.0)
        for side in (-1, 1):
            parts.append(add_cylinder("jet engine", 0.18, 0.95, (side * 1.25, -0.45, -0.32), metal, rotation=(math.radians(90), 0, 0)))
        tail(1.55, 0.78, 1.85)

    return parent_target(parts, "aircraft", location, rotation, scale, model + "-" + livery, role)


def create_bird(mats, rng, location, rotation, scale, role="hero"):
    plumage = mats["hawk"] if rng.random() < 0.35 else mats["gull"]
    tip = mats["gull_tip"]
    parts = []
    parts.append(add_uv_sphere("bird body", (0.16, 0.48, 0.15), (0, 0, 0), plumage, 20, 12))
    parts.append(add_uv_sphere("bird head", (0.12, 0.14, 0.11), (0, -0.44, 0.06), plumage, 16, 10))
    parts.append(add_cube("bird beak", (0.045, 0.20, 0.035), (0, -0.62, 0.05), mats["beak"], bevel=0.006))
    flap = rng.uniform(0.08, 0.4)
    for side in (-1.0, 1.0):
        inner = add_cube("wing inner", (0.72, 0.34, 0.07), (side * 0.48, 0.02, 0.06), plumage)
        inner.rotation_euler[1] = side * math.radians(10)
        mid = add_cube("wing mid", (0.64, 0.26, 0.05), (side * 1.08, 0.0, 0.16 + flap * 0.25), plumage)
        mid.rotation_euler[1] = side * math.radians(18)
        outer = add_cube("wing tip", (0.50, 0.14, 0.035), (side * 1.58, -0.02, 0.32 + flap), tip)
        outer.rotation_euler[1] = side * math.radians(28)
        parts.extend((inner, mid, outer))
    parts.append(add_cube("tail", (0.42, 0.30, 0.035), (0, 0.46, 0.02), tip))
    return parent_target(parts, "bird", location, rotation, scale, "gull" if plumage == mats["gull"] else "hawk", role)


def create_balloon(mats, rng, location, rotation, scale, role="hero"):
    names = [name for name in mats["balloons"] if name != "white"]
    if rng.random() < 0.1:
        names.append("white")
    color_name = rng.choice(names)
    cloth = mats["balloons"][color_name]
    parts = []
    parts.append(add_uv_sphere("envelope", (0.68, 0.68, 0.86), (0, 0, 0.62), cloth, 28, 16))
    parts.append(add_uv_sphere("knot", (0.10, 0.10, 0.11), (0, 0, -0.16), cloth, 12, 8))
    for side in (-0.14, 0.14):
        parts.append(add_rod_between("string", (side, 0.0, -0.2), (side * 0.65, 0.0, -1.15), 0.014, mats["rubber"], vertices=8))
    parts.append(add_cube("basket", (0.38, 0.30, 0.26), (0, 0, -1.32), mats["basket"], bevel=0.016))
    return parent_target(parts, "balloon", location, rotation, scale, color_name, role)


# =====================================================================
# PROCEDURAL ENVIRONMENT: MOUNTAINS, CITY, GROUND, SKY
# =====================================================================

LIGHTING_PRESETS = [
    {"name": "clear_day", "sun_elev": 48, "sun_az": 150, "sun_color": (1.0, 0.96, 0.90), "sun_energy": 3.4,
     "air": 1.0, "aerosol": 0.4, "ozone": 1.2, "exposure": -0.1, "bg": 0.35, "haze_k": 0.0008,
     "cloud_cover": 0.55, "cloud_scale": 1.3, "cloud_tint": (1.0, 1.0, 1.0)},
    {"name": "golden_hour", "sun_elev": 18, "sun_az": 215, "sun_color": (1.0, 0.66, 0.34), "sun_energy": 3.0,
     "air": 1.1, "aerosol": 1.3, "ozone": 0.8, "exposure": -0.05, "bg": 0.38, "haze_k": 0.0010,
     "cloud_cover": 0.50, "cloud_scale": 1.2, "cloud_tint": (1.0, 0.80, 0.62)},
    {"name": "overcast", "sun_elev": 40, "sun_az": 180, "sun_color": (0.80, 0.84, 0.90), "sun_energy": 1.8,
     "air": 1.15, "aerosol": 1.6, "ozone": 0.95, "exposure": 0.0, "bg": 0.42, "haze_k": 0.0012,
     "cloud_cover": 0.38, "cloud_scale": 1.0, "cloud_tint": (0.80, 0.83, 0.88)},
    {"name": "sunset", "sun_elev": 10, "sun_az": 235, "sun_color": (1.0, 0.45, 0.20), "sun_energy": 2.4,
     "air": 1.1, "aerosol": 1.5, "ozone": 0.75, "exposure": 0.0, "bg": 0.35, "haze_k": 0.0010,
     "cloud_cover": 0.48, "cloud_scale": 1.2, "cloud_tint": (1.0, 0.62, 0.45)},
]


def _haze_values(verts):
    cam = active_scene().camera
    cp = cam.location if cam else Vector((0.0, 0.0, 6.0))
    k = CURRENT_PRESET.get("haze_k", 0.0010) if AW_HAZE else 0.0
    out = []
    for (x, y, z) in verts:
        d = math.sqrt((x - cp.x) ** 2 + (y - cp.y) ** 2 + (z - cp.z) ** 2)
        d_eff = max(0.0, d - HAZE_START)
        out.append(min(0.85, 1.0 - math.exp(-k * d_eff * (1.0 + d / HAZE_FAR_BOOST))))
    return out


def _grid_from_axes(name, xs, ys, height_fn, material):
    verts = [(x, y, height_fn(x, y)) for y in ys for x in xs]
    row = len(xs)
    faces = [(j * row + i, j * row + i + 1, (j + 1) * row + i + 1, (j + 1) * row + i)
             for j in range(len(ys) - 1) for i in range(row - 1)]
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    mesh.validate()
    attr = mesh.attributes.new(name="aw_haze", type="FLOAT", domain="POINT")
    attr.data.foreach_set("value", _haze_values(verts))
    mesh.materials.append(material)
    for poly in mesh.polygons:
        poly.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    link_object(obj)
    return obj


def _linspace(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def _ridge_mf(x, y, seed, freq, octaves=5):
    total, amp, f, weight = 0.0, 0.5, freq, 1.0
    for i in range(octaves):
        n = 1.0 - abs(noise.noise(Vector((x * f, y * f, seed + i * 3.1))))
        n *= n
        total += n * amp * weight
        weight = min(1.0, max(0.0, n * 1.8))
        amp *= 0.5
        f *= 2.1
    return total


def _city_weight(x, y):
    if not CITY_ZONE:
        return 0.0
    cx, cy, rx, ry, _z = CITY_ZONE
    d = math.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2)
    t = min(1.0, max(0.0, (d - 0.85) / 0.30))
    return 1.0 - (t * t * (3.0 - 2.0 * t))


def ground_height(x, y, seed):
    t = min(1.0, max(0.0, (y - 25.0) / 160.0))
    ramp = t * t * (3.0 - 2.0 * t)
    n = 0.5 + 0.5 * noise.fractal(Vector((x * 0.006, y * 0.006, seed)), 0.6, 2.0, 4)
    fine = 0.35 * noise.noise(Vector((x * 0.04, y * 0.04, seed + 7.0)))
    h = -0.5 + ramp * 14.0 * n + fine * (0.3 + 0.7 * ramp)
    if CITY_ZONE:
        w = _city_weight(x, y)
        h = h * (1.0 - w) + CITY_ZONE[4] * w
    return h


def make_landscape_material(name, kind, amp):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    set_if_present(bsdf, "Roughness", 0.92)

    col = (0.08, 0.14, 0.06, 1.0) if kind == "ground" else (0.16, 0.18, 0.19, 1.0)
    set_if_present(bsdf, "Base Color", col)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def make_ground(rng):
    seed = rng.uniform(0, 400)
    draft = CFG["quality"] == "draft"
    mat = make_landscape_material("AW_GroundMat", "ground", 14.0)
    xs = _linspace(-700, 700, 120 if draft else 180)
    nrows = 75 if draft else 120
    ys = [-40.0 + 560.0 * ((j / (nrows - 1)) ** 1.8) for j in range(nrows)]
    _grid_from_axes("AW_Ground", xs, ys, lambda x, y: ground_height(x, y, seed), mat)
    return seed


def make_mountain_range(name, y0, depth, amp, rng):
    seed = rng.uniform(0, 500)
    draft = CFG["quality"] == "draft"
    mat = make_landscape_material(name + "Mat", "mountain", amp)
    xs = _linspace(-480, 480, 160 if draft else 260)
    ys = _linspace(y0, y0 + depth, 24 if draft else 36)

    def height(x, y):
        v = (y - y0) / depth
        profile = min(1.0, v / 0.5) ** 1.3 * (1.0 - 0.5 * v)
        env = 0.55 + 0.45 * noise.noise(Vector((x * 0.0042, seed, 1.7)))
        return -2.0 + amp * (0.12 + 1.15 * env * _ridge_mf(x, y, seed, 0.0030)) * profile

    return _grid_from_axes(name, xs, ys, height, mat)


# ---------------- Merged City Builder ----------------
class _M:
    def __init__(self):
        self.v, self.f, self.col = [], [], []

    def add(self, verts, faces, color):
        b = len(self.v)
        self.v.extend(verts)
        self.f.extend([tuple(b + i for i in fc) for fc in faces])
        self.col.extend([color] * len(verts))


_BOX_F = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]


def _box(cx, cy, z0, z1, sx, sy):
    hx, hy = sx / 2.0, sy / 2.0
    return [(cx - hx, cy - hy, z0), (cx + hx, cy - hy, z0), (cx + hx, cy + hy, z0), (cx - hx, cy + hy, z0),
            (cx - hx, cy - hy, z1), (cx + hx, cy - hy, z1), (cx + hx, cy + hy, z1), (cx - hx, cy + hy, z1)]


def make_city(rng, ground_seed):
    if not CITY_ZONE:
        return
    cx, cy, rx, ry, z0 = CITY_ZONE
    pitch = 30.0
    nx = int(2 * rx * 0.70 / pitch) + 1
    ny = int(2 * ry * 0.70 / pitch) + 1
    M = _M()
    buildings = []

    wall_colors = [
        (0.24, 0.24, 0.25), (0.32, 0.30, 0.28), (0.18, 0.20, 0.22),
        (0.40, 0.38, 0.35), (0.28, 0.28, 0.30)
    ]

    for i in range(nx):
        bx = cx + (i - (nx - 1) / 2.0) * pitch
        for j in range(ny):
            by = cy + (j - (ny - 1) / 2.0) * pitch
            d = math.sqrt(((bx - cx) / rx) ** 2 + ((by - cy) / ry) ** 2)
            if d > 0.72:
                continue
            h = 8.0 + 24.0 * max(0.0, 1.0 - d / 0.72) ** 1.5 * rng.uniform(0.6, 1.2)
            w, dd = rng.uniform(8.0, 11.0), rng.uniform(8.0, 11.0)
            col = rng.choice(wall_colors)
            M.add(_box(bx, by, z0 - 2.0, z0 + h, w, dd), _BOX_F, col)
            buildings.append({"x": bx, "y": by, "w": w, "d": dd, "h": h, "kind": "block"})

    CITY_TOPS[:] = [(b["x"], b["y"], b["w"] / 2.0, b["d"] / 2.0, b["h"], b["kind"]) for b in buildings]

    if M.v:
        mesh = bpy.data.meshes.new("AW_CityMesh")
        mesh.from_pydata(M.v, [], M.f)
        mesh.update()
        mat = make_material("AW_CityMat", (0.25, 0.25, 0.26), roughness=0.85)
        mesh.materials.append(mat)
        obj = bpy.data.objects.new("AW_City", mesh)
        link_object(obj)


def setup_world(scene, preset):
    global CURRENT_PRESET
    CURRENT_PRESET = preset
    world = bpy.data.worlds.new("AW_Sky")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    sky = nt.nodes.new("ShaderNodeTexSky")

    items = [i.identifier for i in sky.bl_rna.properties["sky_type"].enum_items]
    for name in ("MULTIPLE_SCATTERING", "SINGLE_SCATTERING", "HOSEK_WILKIE", "PREETHAM"):
        if name in items:
            sky.sky_type = name
            break

    if hasattr(sky, "sun_elevation"):
        sky.sun_elevation = math.radians(preset["sun_elev"])
    if hasattr(sky, "sun_rotation"):
        sky.sun_rotation = math.radians(preset["sun_az"])
    if hasattr(sky, "air_density"):
        sky.air_density = preset["air"]
    if hasattr(sky, "aerosol_density"):
        sky.aerosol_density = preset["aerosol"]

    bg.inputs["Strength"].default_value = preset["bg"]
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    return sky


def setup_camera(scene, rng):
    cam_data = bpy.data.cameras.new("AW_Camera")
    camera = bpy.data.objects.new("AW_Camera", cam_data)
    link_object(camera)
    camera.location = (rng.uniform(-0.5, 0.5), 0.0, rng.uniform(5.5, 8.5))
    camera.rotation_euler = (math.radians(rng.uniform(89.0, 92.0)), 0.0, math.radians(rng.uniform(-1.5, 1.5)))
    cam_data.lens = rng.uniform(50.0, 68.0)
    cam_data.sensor_width = 36.0
    cam_data.clip_start = 0.05
    cam_data.clip_end = 4000.0
    scene.camera = camera
    bpy.context.view_layer.update()
    return camera


def setup_lighting(scene, preset):
    sun_data = bpy.data.lights.new("AW_Sun", type="SUN")
    sun = bpy.data.objects.new("AW_Sun", sun_data)
    link_object(sun)
    sun.rotation_euler = (math.radians(90.0 - preset["sun_elev"]), 0.0, math.radians(preset["sun_az"] + 180.0))
    sun_data.energy = preset["sun_energy"]
    sun_data.color = preset["sun_color"]

    fill_data = bpy.data.lights.new("AW_Fill", type="SUN")
    fill = bpy.data.objects.new("AW_Fill", fill_data)
    link_object(fill)
    fill.rotation_euler = (math.radians(70), 0.0, math.radians(preset["sun_az"]))
    fill_data.energy = 0.20
    fill_data.color = (0.60, 0.72, 0.95)


def configure_render(scene):
    items = [item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items]
    for name in ("BLENDER_EEVEE", "CYCLES"):
        if name in items:
            scene.render.engine = name
            break

    eevee = getattr(scene, "eevee", None)
    samples = 16 if CFG["quality"] == "draft" else 48
    if eevee is not None:
        if hasattr(eevee, "taa_render_samples"):
            eevee.taa_render_samples = samples
        if hasattr(eevee, "use_shadows"):
            eevee.use_shadows = True

    scene.render.resolution_x = CFG["width"]
    scene.render.resolution_y = CFG["height"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False

    try:
        scene.view_settings.view_transform = "AgX"
    except Exception:
        pass
    return scene.render.engine


# =====================================================================
# RAYCAST VISIBILITY & YOLO LABELING
# =====================================================================

def point_on_screen(camera, nx, ny, dist):
    cam = camera.data
    tan_h = (cam.sensor_width / 2.0) / cam.lens
    tan_v = tan_h * CFG["height"] / CFG["width"]
    local = Vector(((nx - 0.5) * 2 * tan_h, (ny - 0.5) * 2 * tan_v, -1.0)).normalized()
    world_dir = camera.matrix_world.to_quaternion() @ local
    return camera.matrix_world.translation + world_dir * dist


def yolo_bbox_for_target(scene, camera, root):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points = []
    for obj in root.children_recursive:
        if not obj.get("aw_target_part") or obj.type != "MESH":
            continue
        evaluated = obj.evaluated_get(depsgraph)
        for corner in evaluated.bound_box:
            ndc = world_to_camera_view(scene, camera, evaluated.matrix_world @ Vector(corner))
            if ndc.z > 0:
                points.append((ndc.x, ndc.y))
    if not points:
        return None
    x1 = max(0.0, min(p[0] for p in points))
    x2 = min(1.0, max(p[0] for p in points))
    y_bottom = max(0.0, min(p[1] for p in points))
    y_top = min(1.0, max(p[1] for p in points))
    if x2 - x1 < 0.01 or y_top - y_bottom < 0.01:
        return None
    top = 1.0 - y_top
    bottom = 1.0 - y_bottom
    return ((x1 + x2) / 2.0, (top + bottom) / 2.0, x2 - x1, bottom - top)


def _visible_fraction(scene, depsgraph, camera, root):
    origin = camera.matrix_world.translation
    pts = []
    for obj in root.children_recursive:
        if obj.type != "MESH" or not obj.get("aw_target_part"):
            continue
        ev = obj.evaluated_get(depsgraph)
        c = Vector((0.0, 0.0, 0.0))
        for corner in ev.bound_box:
            c += ev.matrix_world @ Vector(corner)
        pts.append(c / 8.0)
    if not pts:
        return 0.0
    pts = pts[::max(1, len(pts) // 20)]
    seen = 0
    target_objs = set(root.children_recursive)
    target_objs.add(root)
    for p in pts:
        d = p - origin
        dist = d.length
        hit, _loc, _nrm, _idx, hobj, _mw = scene.ray_cast(depsgraph, origin, d.normalized(), distance=dist + 0.5)
        if not hit:
            seen += 1
        elif (
            hobj in target_objs
            or getattr(hobj, "original", None) in target_objs
            or (hobj.parent and (hobj.parent in target_objs or getattr(hobj.parent, "original", None) in target_objs))
        ):
            seen += 1
    return seen / float(len(pts))


def cull_and_label(scene, camera, roots):
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    lines = []
    for root in list(roots):
        box = yolo_bbox_for_target(scene, camera, root)
        if box is None or not all(math.isfinite(v) for v in box):
            delete_hierarchy(root)
            continue
        cx, cy, bw, bh = box
        if bw < MIN_BOX_W or bh < MIN_BOX_H or bw > 0.90 or bh > 0.90:
            delete_hierarchy(root)
            continue
        if _visible_fraction(scene, depsgraph, camera, root) < MIN_VISIBLE:
            delete_hierarchy(root)
            continue
        kind = root["aw_target_class"]
        class_id = CLASS_TO_ID.get(kind, 0)
        lines.append((class_id, cx, cy, bw, bh, kind, root.get("aw_variant", ""), root.get("aw_role", "hero")))
    return lines


# =====================================================================
# SCENARIO COMPOSITION
# =====================================================================

def _in_frame(scene, camera, p):
    ndc = world_to_camera_view(scene, camera, p)
    return ndc.z > 0 and 0.05 < ndc.x < 0.95 and 0.07 < ndc.y < 0.93


def _ok_point(scene, camera, p, kind, used, gseed):
    min_z = MIN_Z.get(kind, 3.0)
    if p.z < min_z or p.z < ground_height(p.x, p.y, gseed) + 3.0:
        return False
    if not _in_frame(scene, camera, p):
        return False
    for (bx, by, hw, hd, h, _k) in CITY_TOPS:
        if abs(p.x - bx) < hw + 3.0 and abs(p.y - by) < hd + 3.0 and p.z < h + 3.0:
            return False
    for (q, k2) in used:
        sep = max(SEPARATION.get(kind, 2.0), SEPARATION.get(k2, 2.0))
        if (p - q).length < sep:
            return False
    return True


def _try_point(scene, camera, rng, kind, dist_range, used, gseed, hero=False, around=None, spread=8.0):
    for _ in range(120):
        if around is None:
            nx = rng.uniform(0.22, 0.78) if hero else rng.uniform(0.08, 0.92)
            ny = rng.uniform(0.40, 0.74) if hero else rng.uniform(0.36, 0.80)
            p = point_on_screen(camera, nx, ny, rng.uniform(*dist_range))
        else:
            p = around + Vector((rng.uniform(-spread, spread), rng.uniform(-spread, spread), rng.uniform(-3.0, 3.0)))
        if _ok_point(scene, camera, p, kind, used, gseed):
            return p
    return None


def _heading_to_city(p, rng):
    best = None
    for (bx, by, hw, hd, h, _k) in CITY_TOPS:
        if by < p.y - 5.0:
            continue
        score = math.hypot(bx - p.x, by - p.y) - h * 0.8
        if best is None or score < best[0]:
            best = (score, bx, by)
    bx, by = (best[1], best[2]) if best else (CITY_ZONE[0], CITY_ZONE[1])
    yaw = math.atan2(bx - p.x, -(by - p.y))
    return yaw + math.radians(rng.gauss(0.0, 10.0))


def _spawn_target(mats, rng, kind, p, yaw, role):
    pitch = rng.uniform(-4.0, 12.0)
    roll = rng.uniform(-14.0, 14.0)
    rot = (math.radians(pitch), math.radians(roll), yaw)

    if kind in ("drone", "kamikaze_drone"):
        # Select among rich drone variants
        sub_choice = rng.choice(["kamikaze_delta", "fpv", "twin_boom", "multirotor"])
        if kind == "kamikaze_drone" or sub_choice == "kamikaze_delta":
            livery = rng.choice(["desert", "olive", "ram"])
            return build_delta_kamikaze_drone(mats, rng, p, rot, rng.uniform(0.85, 1.25), role, livery=livery)
        elif sub_choice == "fpv":
            return build_fpv_drone(mats, rng, p, rot, rng.uniform(0.75, 1.15), role)
        elif sub_choice == "twin_boom":
            return build_twin_boom_drone(mats, rng, p, rot, rng.uniform(0.70, 1.10), role)
        else:
            var = rng.choice(["survey_quad", "field_hex", "coast_quad", "rescue_quad", "heavy_hex"])
            return build_multirotor(mats, rng, p, rot, rng.uniform(0.40, 0.75), var, role)

    elif kind == "military_jet":
        # Select among diverse warplane fighter types
        fighter_type = rng.choice(["twin_tail", "delta_canard", "stealth"])
        livery = rng.randint(0, 2)
        if fighter_type == "twin_tail":
            return create_twin_tail_fighter(mats, rng, p, rot, rng.uniform(0.90, 1.15), role, livery)
        elif fighter_type == "delta_canard":
            return create_delta_canard_fighter(mats, rng, p, rot, rng.uniform(0.90, 1.15), role, livery)
        else:
            return create_stealth_fighter(mats, rng, p, rot, rng.uniform(0.90, 1.15), role, livery)

    elif kind == "aircraft":
        model = rng.choice(("highwing", "lowwing", "twin", "jet"))
        livery = rng.choice(("blue", "green", "orange"))
        scale_val = rng.uniform(2.0, 2.5) if model != "jet" else rng.uniform(4.0, 5.0)
        return create_aircraft(mats, rng, p, rot, scale_val, role, model, livery)

    elif kind == "bird":
        return create_bird(mats, rng, p, rot, rng.uniform(0.35, 0.55), role)

    else:
        return create_balloon(mats, rng, p, rot, rng.uniform(1.0, 1.6), role)


def show_colors_in_viewport():
    try:
        screen = bpy.context.screen
        if screen is None:
            return
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type != "VIEW_3D":
                    continue
                try:
                    space.shading.type = "RENDERED"
                except Exception:
                    space.shading.type = "MATERIAL"
                try:
                    space.overlay.show_overlays = False
                except Exception:
                    pass
                try:
                    space.region_3d.view_perspective = "CAMERA"
                except Exception:
                    pass
            area.tag_redraw()
    except Exception as exc:
        print("[AeroWatch] Viewport left unchanged:", exc, flush=True)


def build_scene(rng, primary_kind):
    global CITY_ZONE, WORLD_RNG
    WORLD_RNG = rng
    scene = active_scene()
    clear_scene(scene)
    mats = build_materials_library()
    preset = rng.choice(LIGHTING_PRESETS)
    setup_world(scene, preset)
    camera = setup_camera(scene, rng)
    setup_lighting(scene, preset)

    CITY_ZONE = (rng.uniform(-30.0, 30.0), rng.uniform(120.0, 140.0),
                 rng.uniform(150.0, 190.0), rng.uniform(55.0, 70.0), 0.0)

    gseed = make_ground(rng)
    make_mountain_range("AW_Near", 190, 90, rng.uniform(16, 26), rng)
    make_mountain_range("AW_Mid", 330, 120, rng.uniform(34, 48), rng)
    make_mountain_range("AW_Far", 510, 140, rng.uniform(52, 74), rng)
    make_city(rng, gseed)
    bpy.context.view_layer.update()

    # Spawn Scenario
    roots = []
    used = []

    def place_target(kind, role, hero=False, around=None, yaw=None, dist=None, spread=8.0):
        d_range = dist or DIST_RANGE.get(kind, (15.0, 45.0))
        p = _try_point(scene, camera, rng, kind, d_range, used, gseed, hero, around, spread)
        if p is None:
            return None, None
        y = yaw if yaw is not None else _heading_to_city(p, rng)
        roots.append(_spawn_target(mats, rng, kind, p, y, role))
        used.append((p, kind))
        return p, y

    hero_p, hero_yaw = place_target(primary_kind, "hero", hero=True)

    if hero_p is not None:
        if primary_kind in ("drone", "kamikaze_drone"):
            # Drone swarm / formation (1-3 wingmen)
            for _ in range(rng.choice((1, 2, 3))):
                place_target(primary_kind, "secondary", around=hero_p, yaw=hero_yaw, spread=7.0)
            if rng.random() < 0.40:
                place_target("military_jet", "secondary", dist=(70.0, 140.0))
        elif primary_kind == "military_jet":
            # Fighter pair (wingman)
            if rng.random() < 0.60:
                place_target("military_jet", "secondary", around=hero_p, yaw=hero_yaw, spread=18.0)
            if rng.random() < 0.50:
                place_target("drone", "secondary")
        elif primary_kind == "aircraft":
            if rng.random() < 0.45:
                place_target("drone", "secondary")
        elif primary_kind == "bird":
            for _ in range(rng.choice((1, 2, 3))):
                place_target("bird", "secondary", around=hero_p, spread=5.0)

    # Far distractor birds
    for _ in range(rng.choice((0, 1, 2))):
        place_target("bird", "secondary", dist=(22.0, 50.0))

    bpy.context.view_layer.update()
    show_colors_in_viewport()
    return camera, roots, preset


# =====================================================================
# DATASET GENERATION & VALIDATION
# =====================================================================

def write_dataset_yaml(output_root: Path):
    lines = [
        f'path: "{output_root.as_posix()}"',
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for index, name in enumerate(CLASS_NAMES):
        lines.append(f"  {index}: {name}")
    (output_root / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_root / "classes.json").write_text(
        json.dumps({str(i): name for i, name in enumerate(CLASS_NAMES)}, indent=2),
        encoding="utf-8",
    )


def build_contact_sheet(paths, dest: Path):
    if not paths:
        return
    tw, th, cols = 320, 180, min(4, len(paths))
    rows = math.ceil(len(paths) / cols)
    sheet = bpy.data.images.new("AW_Contact", cols * tw, rows * th, alpha=True)
    pixels = [0.08, 0.1, 0.09, 1.0] * (cols * tw * rows * th)
    loaded = []
    try:
        for index, path in enumerate(paths):
            image = bpy.data.images.load(str(path), check_existing=False)
            loaded.append(image)
            image.scale(tw, th)
            source = list(image.pixels)
            col = index % cols
            row_from_top = index // cols
            y0 = (rows - 1 - row_from_top) * th
            x0 = col * tw
            width = cols * tw
            for y in range(th):
                for x in range(tw):
                    src_i = (y * tw + x) * 4
                    dst_i = ((y0 + y) * width + (x0 + x)) * 4
                    pixels[dst_i : dst_i + 4] = source[src_i : src_i + 4]
        sheet.pixels = pixels
        sheet.filepath_raw = str(dest)
        sheet.file_format = "PNG"
        sheet.save()
    finally:
        for image in loaded:
            bpy.data.images.remove(image)
        bpy.data.images.remove(sheet)


def validate_output(output_root: Path, expected_images: int):
    errors = []
    counts = {split: 0 for split in ("train", "val", "test")}
    for split in counts:
        images = sorted((output_root / "images" / split).glob("*.png"))
        counts[split] = len(images)
        for image_path in images:
            label_path = output_root / "labels" / split / (image_path.stem + ".txt")
            if not label_path.exists():
                errors.append(f"Missing label for {image_path.name}")
                continue
            text = label_path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            for line in text.splitlines():
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"Bad field count in {label_path.name}")
                    continue
                try:
                    class_id = int(parts[0])
                    nums = [float(v) for v in parts[1:]]
                except ValueError:
                    errors.append(f"Non-numeric label in {label_path.name}")
                    continue
                if class_id < 0 or class_id >= len(CLASS_NAMES):
                    errors.append(f"Bad class id {class_id} in {label_path.name}")
                if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in nums):
                    errors.append(f"Coordinate outside [0,1] in {label_path.name}")

    if sum(counts.values()) != expected_images:
        errors.append(f"Image count {sum(counts.values())} != {expected_images}")
    return errors, counts


def main():
    global SCENE
    parse_args()
    started = time.perf_counter()
    project_root = resolve_project_root()
    output_root = prepare_output_dir(project_root)
    scene_dir = project_root / "scenes"
    scene_dir.mkdir(parents=True, exist_ok=True)
    blend_path = scene_dir / ("AeroWatch_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".blend")

    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    SCENE = bpy.context.scene
    plan = make_plan(CFG["samples"])

    print("=" * 65)
    print("AEROWATCH AI — ADVANCED MULTI-DRONE & AERIAL SCENE GENERATOR")
    print(f"Project Root     : {project_root}")
    print(f"Output Directory : {output_root}")
    print(f"Samples          : {CFG['samples']} ({CFG['quality']} mode, {CFG['width']}x{CFG['height']})")
    print(f"Classes Active   : {CLASS_NAMES}")
    print("=" * 65)

    rows = []
    class_counts = {name: 0 for name in CLASS_NAMES}
    image_paths = []
    engine_name = ""

    for index, (primary, split) in enumerate(plan, start=1):
        sample_seed = CFG["seed"] + index * 7919
        rng = random.Random(sample_seed)
        engine_name = configure_render(SCENE)
        lines, preset_name = [], ""

        for _attempt in range(14):
            camera, roots, preset = build_scene(rng, primary)
            preset_name = preset["name"]
            lines = cull_and_label(SCENE, camera, roots)
            if any(item[5] == primary and item[7] == "hero" for item in lines):
                break
            lines = []

        if not lines:
            print(f"[AeroWatch] Warning: Sample {index} ({primary}) fallback to best available attempt.")

        image_path = output_root / "images" / split / (f"aw_{index:05d}.png")
        label_path = output_root / "labels" / split / (f"aw_{index:05d}.txt")
        SCENE.render.filepath = str(image_path)
        bpy.ops.render.render(write_still=True)

        label_lines, kinds = [], []
        for class_id, cx, cy, bw, bh, kind, variant, _role in lines:
            label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            class_counts[kind] = class_counts.get(kind, 0) + 1
            kinds.append(kind)

        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        image_paths.append(image_path)
        hero_w = max((item[3] for item in lines if item[7] == "hero"), default=0.0)

        rows.append({
            "image": image_path.relative_to(output_root).as_posix(),
            "label": label_path.relative_to(output_root).as_posix(),
            "split": split,
            "primary_class": primary,
            "objects": len(lines),
            "objects_by_class": ",".join(sorted(kinds)),
            "hero_bbox_w": round(hero_w, 4),
            "lighting": preset_name,
            "seed": sample_seed,
            "resolution": f"{CFG['width']}x{CFG['height']}",
            "quality": CFG["quality"],
            "engine": engine_name,
        })
        print(f"[AeroWatch] {index:3d}/{len(plan)} {primary:14s} | split={split:5s} | light={preset_name:12s} | objects={len(lines)} | hero_w={hero_w:.3f}")

    with (output_root / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (output_root / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    write_dataset_yaml(output_root)

    contact_path = output_root / "contact_sheet.png"
    try:
        build_contact_sheet(image_paths, contact_path)
        print(f"[AeroWatch] Contact sheet written: {contact_path}")
    except Exception as exc:
        print("[AeroWatch] Contact sheet notice:", exc)

    errors, split_counts = validate_output(output_root, len(plan))
    elapsed = round(time.perf_counter() - started, 2)

    summary = {
        "blender_version": bpy.app.version_string,
        "engine": engine_name,
        "samples": len(plan),
        "resolution": f"{CFG['width']}x{CFG['height']}",
        "quality": CFG["quality"],
        "elapsed_sec": elapsed,
        "split_image_counts": split_counts,
        "labeled_object_counts": class_counts,
        "classes": CLASS_NAMES,
        "validation_errors": errors,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if CFG["save_blend"]:
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        print(f"[AeroWatch] Blend scene saved: {blend_path}")

    print("=" * 65)
    print(f"[AeroWatch] COMPLETE in {elapsed:.1f}s -> {output_root}")
    print("=" * 65)


if __name__ == "__main__":
    main()
