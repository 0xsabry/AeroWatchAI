"""Generate a small synthetic aerial-object dataset with Blender.

Run from the AeroWatchAI project root:
  blender --background --factory-startup --python blender_scripts/generate_synthetic_data.py -- --samples 80

This is an educational computer-vision dataset generator. It produces generic
visual classes (drone, bird, aircraft, balloon), not weapon/launch detections.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "synthetic_v1"
SCENE_PATH = PROJECT_ROOT / "scenes" / "aerowatch_synthetic_sky.blend"
CLASS_NAMES = ["drone", "bird", "aircraft", "balloon"]
SEED = 20261010
WIDTH, HEIGHT = 640, 360


def parse_args():
    blender_args = []
    if "--" in sys.argv:
        blender_args = sys.argv[sys.argv.index("--") + 1 :]
    parser = argparse.ArgumentParser(description="Generate synthetic Blender dataset")
    parser.add_argument("--samples", type=int, default=80,
                        help="Number of rendered images (start with 40-80 to validate pipeline)")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(blender_args)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.cameras, bpy.data.lights):
        for block in list(datablocks):
            if block.users == 0:
                datablocks.remove(block)


def make_material(name, color, metallic=0.0, roughness=0.65):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1.0)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Metallic"].default_value = metallic
        bsdf.inputs["Roughness"].default_value = roughness
    return mat


def add_cube(name, dimensions, location, material):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if material:
        obj.data.materials.append(material)
    return obj


def add_uv_sphere(name, scale, location, material, segments=16, rings=8):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=segments, ring_count=rings, radius=1.0, location=location
    )
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if material:
        obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def add_cylinder(name, radius, depth, location, material, vertices=12):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices, radius=radius, depth=depth, location=location
    )
    obj = bpy.context.object
    obj.name = name
    if material:
        obj.data.materials.append(material)
    return obj


def join_parts(name, parts):
    bpy.ops.object.select_all(action="DESELECT")
    for part in parts:
        part.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    root = bpy.context.object
    root.name = name
    # Preserve the active part's origin/location. Components are joined into
    # that local coordinate system, which matters for cloud clusters and the
    # balloon envelope offset from its basket.
    root.rotation_euler = (0.0, 0.0, 0.0)
    root.scale = (1.0, 1.0, 1.0)
    return root


def create_drone(mats):
    dark, metal, accent = mats["dark"], mats["metal"], mats["accent"]
    parts = [add_cube("drone_body", (0.34, 0.48, 0.16), (0, 0, 0), dark)]
    parts.append(add_cube("drone_arm_x", (1.45, 0.07, 0.055), (0, 0, 0.02), metal))
    parts.append(add_cube("drone_arm_y", (0.07, 1.15, 0.055), (0, 0, 0.02), metal))
    for x in (-0.62, 0.62):
        for y in (-0.46, 0.46):
            parts.append(add_cylinder("motor", 0.085, 0.09, (x, y, 0.075), dark))
            parts.append(add_cylinder("rotor", 0.20, 0.018, (x, y, 0.13), accent, vertices=20))
    return join_parts("synthetic_drone", parts)


def create_bird(mats):
    # Flat silhouette intentionally makes a clear, controllable visual class.
    verts = [
        (-0.82, 0.0, 0.00), (-0.45, 0.0, 0.12), (-0.18, 0.0, 0.08),
        (0.00, 0.0, 0.16), (0.18, 0.0, 0.08), (0.45, 0.0, 0.12),
        (0.82, 0.0, 0.00), (0.28, 0.0, -0.06), (0.00, 0.0, -0.16),
        (-0.28, 0.0, -0.06),
    ]
    faces = [(0, 1, 2, 9), (2, 3, 4, 7, 8, 9), (4, 5, 6, 7)]
    mesh = bpy.data.meshes.new("bird_silhouette_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.materials.append(mats["dark"])
    mesh.update()
    obj = bpy.data.objects.new("synthetic_bird", mesh)
    bpy.context.collection.objects.link(obj)
    obj.pass_index = 2
    return obj


def create_aircraft(mats):
    dark, metal = mats["dark"], mats["metal"]
    parts = [add_cube("aircraft_fuselage", (0.16, 1.15, 0.14), (0, 0, 0), dark)]
    parts.append(add_cube("aircraft_wing", (1.55, 0.26, 0.055), (0, 0.02, 0), metal))
    parts.append(add_cube("aircraft_tailplane", (0.62, 0.15, 0.045), (0, 0.43, 0), dark))
    parts.append(add_cube("aircraft_fin", (0.065, 0.30, 0.19), (0, 0.40, 0.10), dark))
    return join_parts("synthetic_aircraft", parts)


def create_balloon(mats):
    parts = [add_uv_sphere("balloon_envelope", (0.35, 0.31, 0.43), (0, 0, 0.25), mats["accent"])]
    parts.append(add_cube("balloon_basket", (0.14, 0.13, 0.16), (0, 0, -0.38), mats["dark"]))
    parts.append(add_cube("balloon_rope_left", (0.018, 0.018, 0.20), (-0.08, 0, -0.23), mats["metal"]))
    parts.append(add_cube("balloon_rope_right", (0.018, 0.018, 0.20), (0.08, 0, -0.23), mats["metal"]))
    return join_parts("synthetic_balloon", parts)


def create_ridge(name, y, max_height, material, rng):
    xs = list(range(-100, 101, 5))
    verts = []
    for x in xs:
        # Irregular skyline with a broad range of low/high peaks.
        h = max(0.8, rng.uniform(0.5, max_height) + 0.9 * math.sin(x * 0.17))
        verts.extend([(x, y, 0.0), (x, y + 7.0, h)])
    faces = []
    for i in range(len(xs) - 1):
        a = i * 2
        faces.append((a, a + 1, a + 3, a + 2))
    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.materials.append(material)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def create_cloud_cluster(name, location, scale, material):
    # Simple soft, low-poly cloud masses behind the labeled objects.
    parts = []
    offsets = [(-1.3, 0.0, 0.0), (-0.35, 0.2, 0.28), (0.7, 0.0, 0.05), (1.45, 0.15, -0.05)]
    sizes = [(1.0, 0.65, 0.7), (1.2, 0.8, 0.9), (0.95, 0.62, 0.65), (0.72, 0.5, 0.5)]
    for i, (off, size) in enumerate(zip(offsets, sizes)):
        loc = (location[0] + off[0] * scale, location[1] + off[1] * scale, location[2] + off[2] * scale)
        s = tuple(v * scale for v in size)
        parts.append(add_uv_sphere(f"{name}_{i}", s, loc, material, segments=12, rings=8))
    cloud = join_parts(name, parts)
    cloud.hide_render = True
    return cloud


def build_camera(scene):
    camera_data = bpy.data.cameras.new("AeroWatchCamera")
    camera = bpy.data.objects.new("AeroWatchCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (0.0, -27.0, 3.5)
    target = Vector((0.0, 18.0, 7.8))
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = 52.0
    camera.data.sensor_width = 36.0
    scene.camera = camera
    return camera


def setup_world(scene):
    world = bpy.data.worlds.new("AeroWatchSky") if not bpy.data.worlds else bpy.data.worlds[0]
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    sky = nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    sky.sun_elevation = math.radians(36)
    sky.sun_rotation = math.radians(130)
    sky.air_density = 1.1
    bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 0.7
    output = nodes.new("ShaderNodeOutputWorld")
    world.node_tree.links.new(sky.outputs["Color"], bg.inputs["Color"])
    world.node_tree.links.new(bg.outputs["Background"], output.inputs["Surface"])
    return sky


def add_lighting(mats):
    light_data = bpy.data.lights.new("SunKey", type="SUN")
    light_data.energy = 2.0
    light_obj = bpy.data.objects.new("SunKey", light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.rotation_euler = (math.radians(25), math.radians(-10), math.radians(-25))


def setup_scene(rng):
    clear_scene()
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = WIDTH
    scene.render.resolution_y = HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.filepath = "//preview.png"
    try:
        scene.view_settings.view_transform = "AgX"
    except Exception:
        pass

    mats = {
        "dark": make_material("GraphiteComposite", (0.055, 0.075, 0.095), metallic=0.35, roughness=0.42),
        "metal": make_material("AircraftAluminium", (0.34, 0.39, 0.43), metallic=0.55, roughness=0.38),
        "accent": make_material("SafetyOrange", (0.92, 0.24, 0.045), metallic=0.1, roughness=0.5),
        "ground": make_material("DistantGround", (0.11, 0.16, 0.14), roughness=1.0),
        "ridge_far": make_material("FarRidge", (0.15, 0.21, 0.24), roughness=1.0),
        "ridge_near": make_material("NearRidge", (0.075, 0.115, 0.12), roughness=1.0),
        "cloud": make_material("CloudSoft", (0.72, 0.77, 0.80), roughness=1.0),
    }

    # Distant landscape provides a horizon; the sky remains the main background.
    add_cube("ground_plane", (400.0, 400.0, 0.8), (0.0, 80.0, -1.0), mats["ground"])
    ridge_far = create_ridge("far_mountain_ridge", 63.0, 10.0, mats["ridge_far"], rng)
    ridge_near = create_ridge("near_mountain_ridge", 45.0, 5.0, mats["ridge_near"], rng)
    ridge_far.pass_index = 0
    ridge_near.pass_index = 0

    clouds = []
    for i in range(5):
        clouds.append(create_cloud_cluster(
            f"cloud_{i}", (rng.uniform(-24, 24), rng.uniform(25, 39), rng.uniform(10, 19)),
            rng.uniform(0.75, 1.8), mats["cloud"]
        ))

    prototypes = {
        "drone": create_drone(mats),
        "bird": create_bird(mats),
        "aircraft": create_aircraft(mats),
        "balloon": create_balloon(mats),
    }
    for class_id, name in enumerate(CLASS_NAMES, start=1):
        prototypes[name].pass_index = class_id
        prototypes[name].hide_render = True

    camera = build_camera(scene)
    sky = setup_world(scene)
    add_lighting(mats)
    scene.camera = camera
    # Prefer consistent, reasonably quick rendering on a 4 GB laptop GPU.
    return scene, camera, sky, prototypes, clouds, (ridge_far, ridge_near)


def projected_yolo_box(scene, camera, obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    points = []
    for corner in evaluated.bound_box:
        world_point = evaluated.matrix_world @ Vector(corner)
        co = world_to_camera_view(scene, camera, world_point)
        if co.z > 0:
            points.append((co.x * WIDTH, (1.0 - co.y) * HEIGHT))
    if len(points) < 2:
        return None
    x0 = max(0.0, min(p[0] for p in points))
    y0 = max(0.0, min(p[1] for p in points))
    x1 = min(float(WIDTH), max(p[0] for p in points))
    y1 = min(float(HEIGHT), max(p[1] for p in points))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    xc = ((x0 + x1) / 2.0) / WIDTH
    yc = ((y0 + y1) / 2.0) / HEIGHT
    bw = (x1 - x0) / WIDTH
    bh = (y1 - y0) / HEIGHT
    return xc, yc, bw, bh


def prepare_output():
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            folder = DATA_ROOT / kind / split
            folder.mkdir(parents=True, exist_ok=True)
            # Remove only files produced by this generator; leave other formats untouched.
            for pattern in ("aw_*.png", "aw_*.txt"):
                for old in folder.glob(pattern):
                    old.unlink()


def generate(args):
    rng = random.Random(args.seed)
    prepare_output()
    scene, camera, sky, prototypes, clouds, ridges = setup_scene(rng)
    scene.render.resolution_x = WIDTH
    scene.render.resolution_y = HEIGHT

    # Save a reusable Blender scene with one visible example object.
    prototypes["drone"].hide_render = False
    prototypes["drone"].location = (0.0, 5.0, 8.0)
    prototypes["drone"].scale = (0.9, 0.9, 0.9)
    SCENE_PATH.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(SCENE_PATH))

    summary = {"seed": args.seed, "samples_requested": args.samples,
               "resolution": [WIDTH, HEIGHT], "synthetic_only": True,
               "classes": CLASS_NAMES, "generated": {"train": 0, "val": 0},
               "renderer": "Blender Eevee Next"}

    for index in range(1, args.samples + 1):
        split = "val" if index % 5 == 0 else "train"
        active_count = 1 if rng.random() < 0.78 else 2
        selected_names = rng.sample(CLASS_NAMES, k=active_count)
        labels = []

        # Vary lighting, camera focal length, skyline scale, and cloud layout.
        sky.sun_elevation = math.radians(rng.uniform(12.0, 68.0))
        sky.sun_rotation = math.radians(rng.uniform(0.0, 360.0))
        camera.data.lens = rng.uniform(42.0, 60.0)
        ridges[0].scale.z = rng.uniform(0.75, 1.35)
        ridges[1].scale.z = rng.uniform(0.65, 1.4)
        for cloud in clouds:
            cloud.hide_render = rng.random() < 0.32
            cloud.location.x = rng.uniform(-26.0, 26.0)
            cloud.location.y = rng.uniform(24.0, 40.0)
            cloud.location.z = rng.uniform(10.0, 20.0)
            cloud.scale = (rng.uniform(0.7, 1.8),) * 3

        for obj in prototypes.values():
            obj.hide_render = True

        for class_name in selected_names:
            obj = prototypes[class_name]
            obj.hide_render = False
            distance = rng.uniform(25.0, 42.0)
            visible_half_width = distance * 0.33
            obj.location.x = rng.uniform(-visible_half_width * 0.75, visible_half_width * 0.75)
            obj.location.y = rng.uniform(-2.0, 15.0)
            obj.location.z = rng.uniform(5.0, 13.5)
            scale = rng.uniform(0.62, 1.35)
            obj.scale = (scale, scale, scale)
            obj.rotation_euler.z = math.radians(rng.uniform(-28.0, 28.0))

        bpy.context.view_layer.update()
        # Recompute boxes after object transforms are updated.
        for class_name in selected_names:
            obj = prototypes[class_name]
            box = projected_yolo_box(scene, camera, obj)
            if box:
                class_id = CLASS_NAMES.index(class_name)
                labels.append(str(class_id) + " " + " ".join(f"{v:.6f}" for v in box))

        image_dir = DATA_ROOT / "images" / split
        label_dir = DATA_ROOT / "labels" / split
        image_path = image_dir / f"aw_{index:05d}.png"
        label_path = label_dir / f"aw_{index:05d}.txt"
        scene.render.filepath = str(image_path)
        bpy.ops.render.render(write_still=True)
        label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        summary["generated"][split] += 1
        if index == 1 or index % 10 == 0 or index == args.samples:
            print(f"[AeroWatch] rendered {index}/{args.samples} -> {split}")

    yaml_path = DATA_ROOT / "dataset.yaml"
    yaml_text = (
        f"path: {DATA_ROOT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n" + "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)) + "\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    (DATA_ROOT / "metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[AeroWatch] dataset complete:", DATA_ROOT)
    print("[AeroWatch] dataset yaml:", yaml_path)
    print("[AeroWatch] NOTE: this small synthetic run validates the pipeline; it does not establish real-world accuracy.")


def main():
    args = parse_args()
    if args.samples < 5:
        raise SystemExit("Please use at least 5 samples so both train and val splits are created.")
    generate(args)


if __name__ == "__main__":
    main()
