"""AeroWatch AI — Procedural Scene Builder for Blender 5.2.

Constructs varied outdoor aerial environments including:
- Layered procedural mountain ridges and landscape horizon
- Multiple sky and lighting conditions (clear daylight, partly cloudy, overcast, sunset, haze)
- Procedural cloud clusters
- Camera configuration with randomized focal lengths and perspectives
- EEVEE render engine optimizations for RTX 3050 (4 GB VRAM)
"""
from __future__ import annotations

import math
import random
import bpy
from mathutils import Vector, Euler

from asset_factory import create_pbr_material, add_mesh_cube, add_mesh_sphere, join_components


LIGHTING_PRESETS = [
    "clear_day",
    "partly_cloudy",
    "overcast",
    "sunset_golden",
    "hazy_morning",
]


def clear_scene_objects():
    """Removes all objects, meshes, curves, lights, and cameras from the active blend scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.cameras, bpy.data.lights, bpy.data.materials):
        for block in list(datablocks):
            if block.users == 0:
                datablocks.remove(block)


def create_procedural_ridge(
    name: str,
    y_dist: float,
    max_height: float,
    material: bpy.types.Material,
    rng: random.Random,
    step: int = 5,
) -> bpy.types.Object:
    """Creates a continuous irregular distant mountain ridge along the horizon."""
    xs = list(range(-120, 121, step))
    verts = []
    freq1 = rng.uniform(0.08, 0.16)
    freq2 = rng.uniform(0.20, 0.35)
    phase = rng.uniform(0, math.pi * 2)

    for x in xs:
        undulation = (
            math.sin(x * freq1 + phase) * 1.8
            + math.cos(x * freq2) * 0.9
            + rng.uniform(-0.4, 0.4)
        )
        h = max(1.0, rng.uniform(0.6, max_height) + undulation)
        verts.extend([(x, y_dist, -1.0), (x, y_dist + 10.0, h)])

    faces = []
    for i in range(len(xs) - 1):
        a = i * 2
        faces.append((a, a + 1, a + 3, a + 2))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.materials.append(material)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def create_cloud_formation(
    name: str,
    location: tuple[float, float, float],
    scale: float,
    material: bpy.types.Material,
) -> bpy.types.Object:
    """Builds a soft low-poly cluster of cloud spheres."""
    parts = []
    offsets = [
        (-1.4, 0.0, 0.0),
        (-0.4, 0.2, 0.35),
        (0.65, -0.1, 0.1),
        (1.5, 0.15, -0.05),
    ]
    sizes = [
        (1.1, 0.7, 0.75),
        (1.3, 0.85, 0.95),
        (1.0, 0.65, 0.7),
        (0.75, 0.55, 0.55),
    ]
    for i, (off, sz) in enumerate(zip(offsets, sizes)):
        loc = (
            location[0] + off[0] * scale,
            location[1] + off[1] * scale,
            location[2] + off[2] * scale,
        )
        s = tuple(v * scale for v in sz)
        parts.append(add_mesh_sphere(f"{name}_part_{i}", s, loc, material, segments=12, rings=8))

    cloud = join_components(name, parts)
    return cloud


class SceneBuilder:
    """Sets up the Blender camera, terrain, procedural skies, and lighting."""

    def __init__(self, rng: random.Random, width: int = 640, height: int = 360, quality: str = "draft"):
        self.rng = rng
        self.width = width
        self.height = height
        self.quality = quality

    def initialize_scene(self) -> tuple[bpy.types.Scene, bpy.types.Object, bpy.types.Node, bpy.types.Object]:
        clear_scene_objects()
        scene = bpy.context.scene
        scene.render.engine = "BLENDER_EEVEE"
        scene.render.resolution_x = self.width
        scene.render.resolution_y = self.height
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.film_transparent = False

        # Set EEVEE render samples according to quality mode (conservative for 4GB VRAM)
        if hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = 32 if self.quality == "high" else 8

        try:
            scene.view_settings.view_transform = "AgX"
        except Exception:
            pass

        # Environment terrain materials
        mat_ground = create_pbr_material("DistantGround", (0.09, 0.14, 0.11), roughness=0.95)
        mat_ridge_far = create_pbr_material("FarRidge", (0.13, 0.18, 0.22), roughness=1.0)
        mat_ridge_near = create_pbr_material("NearRidge", (0.07, 0.11, 0.12), roughness=1.0)
        mat_cloud = create_pbr_material("CloudMat", (0.78, 0.82, 0.86), roughness=1.0)

        # Distant ground plane
        add_mesh_cube("ground_plane", (450.0, 450.0, 1.0), (0.0, 90.0, -1.0), mat_ground)

        # Mountain skyline
        ridge_far = create_procedural_ridge("ridge_far", 75.0, 12.0, mat_ridge_far, self.rng)
        ridge_near = create_procedural_ridge("ridge_near", 50.0, 6.5, mat_ridge_near, self.rng)

        # Procedural clouds in the background
        self.clouds = []
        for i in range(6):
            c_loc = (
                self.rng.uniform(-35.0, 35.0),
                self.rng.uniform(30.0, 55.0),
                self.rng.uniform(12.0, 24.0),
            )
            c_scale = self.rng.uniform(1.0, 2.2)
            cloud = create_cloud_formation(f"cloud_{i}", c_loc, c_scale, mat_cloud)
            self.clouds.append(cloud)

        # Setup Camera
        camera = self._setup_camera(scene)

        # Setup World Sky & Sunlight
        sky_node, sun_light = self._setup_world_and_sun(scene)

        self.ridge_far = ridge_far
        self.ridge_near = ridge_near
        self.sky_node = sky_node
        self.sun_light = sun_light
        self.camera = camera
        self.scene = scene

        return scene, camera, sky_node, sun_light

    def _setup_camera(self, scene: bpy.types.Scene) -> bpy.types.Object:
        cam_data = bpy.data.cameras.new("AeroWatchCamera")
        cam_obj = bpy.data.objects.new("AeroWatchCamera", cam_data)
        bpy.context.collection.objects.link(cam_obj)

        cam_obj.location = (0.0, -28.0, 3.8)
        target = Vector((0.0, 16.0, 7.5))
        cam_obj.rotation_euler = (target - cam_obj.location).to_track_quat("-Z", "Y").to_euler()
        cam_data.lens = 50.0
        cam_data.sensor_width = 36.0
        cam_data.clip_start = 0.5
        cam_data.clip_end = 500.0
        scene.camera = cam_obj
        return cam_obj

    def _setup_world_and_sun(self, scene: bpy.types.Scene) -> tuple[bpy.types.Node, bpy.types.Object]:
        world = bpy.data.worlds.new("AeroWatchSky")
        scene.world = world
        world.use_nodes = True
        nodes = world.node_tree.nodes
        nodes.clear()

        sky = nodes.new("ShaderNodeTexSky")
        sky.sky_type = "MULTIPLE_SCATTERING"
        sky.sun_elevation = math.radians(40)
        sky.sun_rotation = math.radians(120)
        sky.air_density = 1.0

        bg = nodes.new("ShaderNodeBackground")
        bg.inputs["Strength"].default_value = 0.75
        out = nodes.new("ShaderNodeOutputWorld")

        world.node_tree.links.new(sky.outputs["Color"], bg.inputs["Color"])
        world.node_tree.links.new(bg.outputs["Background"], out.inputs["Surface"])

        # Sun Key Light
        light_data = bpy.data.lights.new("SunKey", type="SUN")
        light_data.energy = 2.2
        sun_obj = bpy.data.objects.new("SunKey", light_data)
        bpy.context.collection.objects.link(sun_obj)
        sun_obj.rotation_euler = (math.radians(35), math.radians(-15), math.radians(-35))

        return sky, sun_obj

    def randomize_environment(self, preset: str | None = None) -> dict[str, any]:
        """Randomizes the lighting condition, sky atmospheric properties, clouds, and camera."""
        condition = preset if preset else self.rng.choice(LIGHTING_PRESETS)
        sun_rot = self.rng.uniform(0.0, 360.0)

        if condition == "clear_day":
            sun_elev = self.rng.uniform(38.0, 68.0)
            air_density = self.rng.uniform(0.8, 1.1)
            sun_energy = self.rng.uniform(2.2, 2.8)
            bg_strength = 0.75
            cloud_prob = 0.15
        elif condition == "partly_cloudy":
            sun_elev = self.rng.uniform(30.0, 58.0)
            air_density = self.rng.uniform(1.0, 1.3)
            sun_energy = self.rng.uniform(1.8, 2.4)
            bg_strength = 0.85
            cloud_prob = 0.65
        elif condition == "overcast":
            sun_elev = self.rng.uniform(25.0, 50.0)
            air_density = self.rng.uniform(1.4, 2.0)
            sun_energy = self.rng.uniform(0.8, 1.2)
            bg_strength = 1.20
            cloud_prob = 0.85
        elif condition == "sunset_golden":
            sun_elev = self.rng.uniform(8.0, 22.0)
            air_density = self.rng.uniform(1.5, 2.4)
            sun_energy = self.rng.uniform(1.5, 2.2)
            bg_strength = 0.70
            cloud_prob = 0.40
        else:  # "hazy_morning"
            sun_elev = self.rng.uniform(15.0, 35.0)
            air_density = self.rng.uniform(2.2, 3.2)
            sun_energy = self.rng.uniform(1.2, 1.8)
            bg_strength = 0.90
            cloud_prob = 0.35

        self.sky_node.sun_elevation = math.radians(sun_elev)
        self.sky_node.sun_rotation = math.radians(sun_rot)
        self.sky_node.air_density = air_density

        world_nodes = self.scene.world.node_tree.nodes
        bg_node = world_nodes.get("Background")
        if bg_node:
            bg_node.inputs["Strength"].default_value = bg_strength

        # Sun object alignment
        sun_light = self.sun_light.data
        sun_light.energy = sun_energy
        self.sun_light.rotation_euler = (
            math.radians(90.0 - sun_elev),
            0.0,
            math.radians(sun_rot - 90.0),
        )

        # Clouds visibility & positions
        for i, cloud in enumerate(self.clouds):
            show = self.rng.random() < cloud_prob
            cloud.hide_render = not show
            if show:
                cloud.location.x = self.rng.uniform(-36.0, 36.0)
                cloud.location.y = self.rng.uniform(28.0, 52.0)
                cloud.location.z = self.rng.uniform(11.0, 24.0)

        # Varied Camera Focal Length (38mm - 65mm) and slight perspective shift
        lens = self.rng.uniform(38.0, 68.0)
        self.camera.data.lens = lens
        cam_shift_x = self.rng.uniform(-1.5, 1.5)
        cam_shift_z = self.rng.uniform(-0.5, 0.8)
        self.camera.location = (cam_shift_x, -28.0, 3.8 + cam_shift_z)
        target = Vector((0.0, 16.0, 7.5))
        self.camera.rotation_euler = (target - self.camera.location).to_track_quat("-Z", "Y").to_euler()

        return {
            "condition": condition,
            "sun_elevation_deg": round(sun_elev, 2),
            "sun_rotation_deg": round(sun_rot, 2),
            "air_density": round(air_density, 2),
            "sun_energy": round(sun_energy, 2),
            "camera_lens_mm": round(lens, 1),
        }
