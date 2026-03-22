import os
import random
import xml.etree.ElementTree as ET
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.errors import RandomizationError
from robosuite.utils.mjcf_utils import (
    array_to_string,
    find_elements,
    string_to_array,
    xml_path_completion,
)
from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
from robosuite.utils.observables import Observable, sensor
from robosuite.environments.base import EnvMeta
from scipy.spatial.transform import Rotation

from robosuite.models.robots import PandaOmron

import robocasa
import robocasa.macros as macros
import robocasa.utils.camera_utils as CamUtils
import robocasa.utils.object_utils as OU
import robocasa.models.scenes.scene_registry as SceneRegistry
from robocasa.models.scenes import KitchenArena
from robocasa.models.fixtures import *
from robocasa.models.objects.kitchen_object_utils import sample_kitchen_object
from robocasa.models.objects.objects import MJCFObject
from robocasa.utils.placement_samplers import (
    SequentialCompositeSampler,
    UniformRandomSampler,
)
from robocasa.utils.texture_swap import (
    get_random_textures,
    replace_cab_textures,
    replace_counter_top_texture,
    replace_floor_texture,
    replace_wall_texture,
)
from robocasa.utils.config_utils import refactor_composite_controller_config


REGISTERED_KITCHEN_ENVS = {}


def register_kitchen_env(target_class):
    REGISTERED_KITCHEN_ENVS[target_class.__name__] = target_class


class KitchenEnvMeta(EnvMeta):
    """Metaclass for registering robocasa environments"""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        register_kitchen_env(cls)
        return cls


_ROBOT_POS_OFFSETS: dict[str, list[float]] = {
    "GR1FloatingBody": [0, 0, 0.97],
    "GR1": [0, 0, 0.97],
    "GR1FixedLowerBody": [0, 0, 0.97],
    "G1FloatingBody": [0, -0.33, 0],
    "G1": [0, -0.33, 0],
    "G1FixedLowerBody": [0, -0.33, 0],
    "GoogleRobot": [0, 0, 0],
}


class Kitchen(ManipulationEnv, metaclass=KitchenEnvMeta):
    """
    Initialized a Base Kitchen environment.

    Args:
        robots: Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)

        env_configuration (str): Specifies how to position the robot(s) within the environment. Default is "default",
            which should be interpreted accordingly by any subclasses.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        base_types (None or str or list of str): type of base, used to instantiate base models from base factory.
            Default is "default", which is the default base associated with the robot(s) the 'robots' specification.
            None results in no base, and any other (valid) model overrides the default base. Should either be
            single str if same base type is to be used for all robots or else it should be a list of the same
            length as "robots" param

        gripper_types (None or str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        placement_initializer (ObjectPositionSampler): if provided, will be used to place objects on every reset,
            else a UniformRandomSampler is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the abase of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        renderer (str): Specifies which renderer to use.

        renderer_config (dict): dictionary for the renderer configurations

        init_robot_base_pos (str): name of the fixture to place the near. If None, will randomly select a fixture.

        seed (int): environment seed. Default is None, where environment is unseeded, ie. random

        layout_and_style_ids (list of list of int): list of layout and style ids to use for the kitchen.

        layout_ids ((list of) LayoutType or int):  layout id(s) to use for the kitchen. -1 and None specify all layouts
            -2 specifies layouts not involving islands/wall stacks, -3 specifies layouts involving islands/wall stacks,
            -4 specifies layouts with dining areas.

        style_ids ((list of) StyleType or int): style id(s) to use for the kitchen. -1 and None specify all styles.

        generative_textures (str): if set to "100p", will use AI generated textures

        obj_registries (tuple of str): tuple containing the object registries to use for sampling objects.
            can contain "objaverse" and/or "aigen" to sample objects from objaverse, AI generated, or both.

        obj_instance_split (str): string for specifying a custom set of object instances to use. "A" specifies
            all but the last 3 object instances (or the first half - whichever is larger), "B" specifies the
            rest, and None specifies all.

        use_distractors (bool): if True, will add distractor objects to the scene

        translucent_robot (bool): if True, will make the robot appear translucent during rendering

        randomize_cameras (bool): if True, will add gaussian noise to the position and rotation of the
            wrist and agentview cameras
    """

    EXCLUDE_LAYOUTS = []

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=True,  # currently unused variable
        reward_scale=1.0,  # currently unused variable
        reward_shaping=False,  # currently unused variables
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="robot0_agentview_center",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=True,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        renderer="mjviewer",
        renderer_config=None,
        init_robot_base_pos=None,
        seed=None,
        layout_and_style_ids=None,
        layout_ids=None,
        style_ids=None,
        scene_split=None,  # unsued, for backwards compatibility
        generative_textures=None,
        obj_registries=("objaverse",),
        obj_instance_split=None,
        use_distractors=False,
        translucent_robot=False,
        randomize_cameras=False,
    ):
        self.init_robot_base_pos = init_robot_base_pos

        # object placement initializer
        self.placement_initializer = placement_initializer
        self.obj_registries = obj_registries
        self.obj_instance_split = obj_instance_split

        if layout_and_style_ids is not None:
            assert (
                layout_ids is None and style_ids is None
            ), "layout_ids and style_ids must both be set to None if layout_and_style_ids is set"
            self.layout_and_style_ids = layout_and_style_ids
        else:
            layout_ids = SceneRegistry.unpack_layout_ids(layout_ids)
            style_ids = SceneRegistry.unpack_style_ids(style_ids)
            self.layout_and_style_ids = [(l, s) for l in layout_ids for s in style_ids]

        # remove excluded layouts
        self.layout_and_style_ids = [
            (int(l), int(s))
            for (l, s) in self.layout_and_style_ids
            if l not in self.EXCLUDE_LAYOUTS
        ]

        assert generative_textures in [None, False, "100p"]
        self.generative_textures = generative_textures

        self.use_distractors = use_distractors
        self.translucent_robot = translucent_robot
        self.randomize_cameras = randomize_cameras

        if isinstance(robots, str):
            robots = [robots]

        # backward compatibility: rename all robots that were previously called PandaMobile -> PandaOmron
        for i in range(len(robots)):
            if robots[i] == "PandaMobile":
                robots[i] = "PandaOmron"
        assert len(robots) == 1

        # intialize cameras
        self._cam_configs = CamUtils.get_robot_cam_configs(robots[0])

        # set up currently unused variables (used in robosuite)
        self.use_object_obs = use_object_obs
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        if controller_configs is not None:
            # detect if using stale controller configs (before robosuite v1.5.1) and update to new convention
            arms = REGISTERED_ROBOTS[robots[0]].arms
            controller_configs = refactor_composite_controller_config(
                controller_configs, robots[0], arms
            )
            if robots[0] == "PandaOmron":
                if "composite_controller_specific_configs" not in controller_configs:
                    controller_configs["composite_controller_specific_configs"] = {}
                controller_configs["composite_controller_specific_configs"][
                    "body_part_ordering"
                ] = ["right", "right_gripper", "base", "torso"]

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=True,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        for robot in self.robots:
            if isinstance(robot.robot_model, PandaOmron):
                robot.init_qpos = (
                    -0.01612974,
                    -1.03446714,
                    -0.02397936,
                    -2.27550888,
                    0.03932365,
                    1.51639493,
                    0.69615947,
                )
                robot.init_torso_qpos = np.array([0.0])

        # determine sample layout and style
        if "layout_id" in self._ep_meta and "style_id" in self._ep_meta:
            self.layout_id = self._ep_meta["layout_id"]
            self.style_id = self._ep_meta["style_id"]
        else:
            layout_id, style_id = self.rng.choice(self.layout_and_style_ids)
            self.layout_id = int(layout_id)
            self.style_id = int(style_id)

        if macros.VERBOSE:
            print("layout: {}, style: {}".format(self.layout_id, self.style_id))

        # to be set later inside edit_model_xml function
        self._curr_gen_fixtures = self._ep_meta.get("gen_textures")

        # setup scene
        self.mujoco_arena = KitchenArena(
            layout_id=self.layout_id,
            style_id=self.style_id,
            rng=self.rng,
        )
        # Arena always gets set to zero origin
        self.mujoco_arena.set_origin([0, 0, 0])
        self.set_cameras()  # setup cameras

        # setup rendering for this layout
        if self.renderer == "mjviewer":
            camera_config = CamUtils.LAYOUT_CAMS.get(
                self.layout_id, CamUtils.DEFAULT_LAYOUT_CAM
            )
            self.renderer_config = {"cam_config": camera_config}

        # setup fixtures
        self.fixture_cfgs = self.mujoco_arena.get_fixture_cfgs()
        self.fixtures = {cfg["name"]: cfg["model"] for cfg in self.fixture_cfgs}

        # setup scene, robots, objects
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=list(self.fixtures.values()),
        )

        # setup fixture locations
        fxtr_placement_initializer = self._get_placement_initializer(
            self.fixture_cfgs, z_offset=0.0
        )
        fxtr_placements = None
        for i in range(10):
            try:
                fxtr_placements = fxtr_placement_initializer.sample()
            except RandomizationError as e:
                if macros.VERBOSE:
                    print("Ranomization error in initial placement. Try #{}".format(i))
                continue
            break
        if fxtr_placements is None:
            if macros.VERBOSE:
                print("Could not place fixtures. Trying again with self._load_model()")
            self._load_model()
            return
        self.fxtr_placements = fxtr_placements
        # Loop through all objects and reset their positions
        for obj_pos, obj_quat, obj in fxtr_placements.values():
            assert isinstance(obj, Fixture)
            obj.set_pos(obj_pos)

            # hacky code to set orientation
            obj.set_euler(T.mat2euler(T.quat2mat(T.convert_quat(obj_quat, "xyzw"))))

        # setup internal references related to fixtures
        self._setup_kitchen_references()

        # set robot position
        if self.init_robot_base_pos is not None:
            ref_fixture = self.get_fixture(self.init_robot_base_pos)
        else:
            fixtures = list(self.fixtures.values())
            valid_src_fixture_classes = [
                "CoffeeMachine",
                "Toaster",
                "Stove",
                "Stovetop",
                "SingleCabinet",
                "HingeCabinet",
                "OpenCabinet",
                "Drawer",
                "Microwave",
                "Sink",
                "Hood",
                "Oven",
                "Fridge",
                "Dishwasher",
            ]
            while True:
                ref_fixture = self.rng.choice(fixtures)
                fxtr_class = type(ref_fixture).__name__
                if fxtr_class not in valid_src_fixture_classes:
                    continue
                break

        robot_base_pos, robot_base_ori = self.compute_robot_base_placement_pose(
            ref_fixture=ref_fixture
        )
        robot_model = self.robots[0].robot_model
        robot_model.set_base_xpos(robot_base_pos)
        robot_model.set_base_ori(robot_base_ori)

        # create and place objects
        self._create_objects()

        # setup object locations
        self.placement_initializer = self._get_placement_initializer(self.object_cfgs)
        object_placements = None
        for i in range(1):
            try:
                object_placements = self.placement_initializer.sample(
                    placed_objects=self.fxtr_placements
                )
            except RandomizationError as e:
                if macros.VERBOSE:
                    print("Randomization error in initial placement. Try #{}".format(i))
                continue
            break
        if object_placements is None:
            if macros.VERBOSE:
                print("Could not place objects. Trying again with self._load_model()")
            self._load_model()
            return
        self.object_placements = object_placements

    def _create_objects(self):
        """
        Creates and places objects in the kitchen environment.
        Helper function called by _create_objects()
        """
        # add objects
        self.objects = {}
        if "object_cfgs" in self._ep_meta:
            self.object_cfgs = self._ep_meta["object_cfgs"]
            for obj_num, cfg in enumerate(self.object_cfgs):
                if "name" not in cfg:
                    cfg["name"] = "obj_{}".format(obj_num + 1)
                model, info = self._create_obj(cfg)
                cfg["info"] = info
                self.objects[model.name] = model
                self.model.merge_objects([model])
        else:
            self.object_cfgs = self._get_obj_cfgs()
            addl_obj_cfgs = []
            for obj_num, cfg in enumerate(self.object_cfgs):
                cfg["type"] = "object"
                if "name" not in cfg:
                    cfg["name"] = "obj_{}".format(obj_num + 1)
                model, info = self._create_obj(cfg)
                cfg["info"] = info
                self.objects[model.name] = model
                self.model.merge_objects([model])

                try_to_place_in = cfg["placement"].get("try_to_place_in", None)

                # place object in a container and add container as an object to the scene
                if try_to_place_in and (
                    "in_container" in cfg["info"]["groups_containing_sampled_obj"]
                ):
                    container_cfg = {
                        "name": cfg["name"] + "_container",
                        "obj_groups": cfg["placement"].get("try_to_place_in"),
                        "placement": deepcopy(cfg["placement"]),
                        "type": "object",
                    }

                    container_kwargs = cfg["placement"].get("container_kwargs", None)
                    if container_kwargs is not None:
                        for k, v in container_kwargs.items():
                            container_cfg[k] = v

                    # add in the new object to the model
                    addl_obj_cfgs.append(container_cfg)
                    model, info = self._create_obj(container_cfg)
                    container_cfg["info"] = info
                    self.objects[model.name] = model
                    self.model.merge_objects([model])

                    # modify object config to lie inside of container
                    cfg["placement"] = dict(
                        size=(0.01, 0.01),
                        ensure_object_boundary_in_range=False,
                        sample_args=dict(
                            reference=container_cfg["name"],
                        ),
                    )

            # prepend the new object configs in
            self.object_cfgs = addl_obj_cfgs + self.object_cfgs

            # # remove objects that didn't get created
            # self.object_cfgs = [cfg for cfg in self.object_cfgs if "model" in cfg]

    def _create_obj(self, cfg):
        """
        Helper function for creating objects.
        Called by _create_objects()
        """
        if "info" in cfg:
            """
            if cfg has "info" key in it, that means it is storing meta data already
            that indicates which object we should be using.
            set the obj_groups to this path to do deterministic playback
            """
            mjcf_path = cfg["info"]["mjcf_path"]
            # replace with correct base path
            new_base_path = os.path.join(robocasa.models.assets_root, "objects")
            new_path = os.path.join(new_base_path, mjcf_path.split("/objects/")[-1])
            obj_groups = new_path
            exclude_obj_groups = None
        else:
            obj_groups = cfg.get("obj_groups", "all")
            exclude_obj_groups = cfg.get("exclude_obj_groups", None)
        object_kwargs, object_info = self.sample_object(
            obj_groups,
            exclude_groups=exclude_obj_groups,
            graspable=cfg.get("graspable", None),
            washable=cfg.get("washable", None),
            microwavable=cfg.get("microwavable", None),
            cookable=cfg.get("cookable", None),
            freezable=cfg.get("freezable", None),
            max_size=cfg.get("max_size", (None, None, None)),
            object_scale=cfg.get("object_scale", None),
        )
        info = object_info

        object = MJCFObject(name=cfg["name"], **object_kwargs)

        return object, info

    def _setup_kitchen_references(self):
        """
        setup fixtures (and their references). this function is called within load_model function for kitchens
        """
        serialized_refs = self._ep_meta.get("fixture_refs", {})
        # unserialize refs
        self.fixture_refs = {
            k: self.get_fixture(v) for (k, v) in serialized_refs.items()
        }

    def _reset_observables(self):
        if self.hard_reset:
            self._observables = self._setup_observables()

    def compute_robot_base_placement_pose(self, ref_fixture, offset=None):
        """
        steps:
        1. find the nearest counter to this fixture
        2. compute offset relative to this counter
        3. transform offset to global coordinates

        Args:
            ref_fixture (Fixture): reference fixture to place th robot near

            offset (list): offset to add to the base position

        """
        # step 1: find vase fixture closest to robot
        base_fixture = None

        # get all base fixtures in the environment
        base_fixtures = [
            fxtr
            for fxtr in self.fixtures.values()
            if isinstance(fxtr, Counter)
            or isinstance(fxtr, Stove)
            or isinstance(fxtr, Stovetop)
            or isinstance(fxtr, HousingCabinet)
            or isinstance(fxtr, Fridge)
        ]

        for fxtr in base_fixtures:
            # get bounds of fixture
            point = ref_fixture.pos
            if not OU.point_in_fixture(point=point, fixture=fxtr, only_2d=True):
                continue
            base_fixture = fxtr
            break

        # set the base fixture as the ref fixture itself if cannot find fixture containing ref
        if base_fixture is None:
            base_fixture = ref_fixture
        # assert base_fixture is not None

        # step 2: compute offset relative to this counter
        base_to_ref, _ = OU.get_rel_transform(base_fixture, ref_fixture)
        cntr_y = base_fixture.get_ext_sites(relative=True)[0][1]
        base_to_edge = [
            base_to_ref[0],
            cntr_y - 0.20,
            0,
        ]
        if offset is not None:
            base_to_edge[0] += offset[0]
            base_to_edge[1] += offset[1]

        if (
            isinstance(base_fixture, HousingCabinet)
            or isinstance(base_fixture, Fridge)
            or "stack" in base_fixture.name
        ):
            base_to_edge[1] -= 0.10

        # apply robot-specific offset relative to the base fixture for x,y dims
        robot_model = self.robots[0].robot_model
        robot_class_name = robot_model.__class__.__name__
        if robot_class_name in _ROBOT_POS_OFFSETS:
            for dimension in range(0, 2):
                base_to_edge[dimension] += _ROBOT_POS_OFFSETS[robot_class_name][
                    dimension
                ]

        # step 3: transform offset to global coordinates
        robot_base_pos = np.zeros(3)
        robot_base_pos[0:2] = OU.get_pos_after_rel_offset(base_fixture, base_to_edge)[
            0:2
        ]
        # apply robot-specific absolutely for z dim
        if robot_class_name in _ROBOT_POS_OFFSETS:
            robot_base_pos[2] = _ROBOT_POS_OFFSETS[robot_class_name][2]
        robot_base_ori = np.array([0, 0, base_fixture.rot + np.pi / 2])

        return robot_base_pos, robot_base_ori

    def _get_placement_initializer(self, cfg_list, z_offset=0.01):

        """
        Creates a placement initializer for the objects/fixtures based on the specifications in the configurations list

        Args:
            cfg_list (list): list of object configurations

            z_offset (float): offset in z direction

        Returns:
            SequentialCompositeSampler: placement initializer

        """

        placement_initializer = SequentialCompositeSampler(
            name="SceneSampler", rng=self.rng
        )

        for (obj_i, cfg) in enumerate(cfg_list):
            # determine which object is being placed
            if cfg["type"] == "fixture":
                mj_obj = self.fixtures[cfg["name"]]
            elif cfg["type"] == "object":
                mj_obj = self.objects[cfg["name"]]
            else:
                raise ValueError

            placement = cfg.get("placement", None)
            if placement is None:
                continue
            fixture_id = placement.get("fixture", None)
            if fixture_id is not None:
                # get fixture to place object on
                fixture = self.get_fixture(
                    id=fixture_id,
                    ref=placement.get("ref", None),
                )

                # calculate the total available space where object could be placed
                sample_region_kwargs = placement.get("sample_region_kwargs", {})
                reset_region = fixture.sample_reset_region(
                    env=self, **sample_region_kwargs
                )
                outer_size = reset_region["size"]
                margin = placement.get("margin", 0.04)
                outer_size = (outer_size[0] - margin, outer_size[1] - margin)

                # calculate the size of the inner region where object will actually be placed
                target_size = placement.get("size", None)
                if target_size is not None:
                    target_size = deepcopy(list(target_size))
                    for size_dim in [0, 1]:
                        if target_size[size_dim] == "obj":
                            target_size[size_dim] = mj_obj.size[size_dim] + 0.005
                        if target_size[size_dim] == "obj.x":
                            target_size[size_dim] = mj_obj.size[0] + 0.005
                        if target_size[size_dim] == "obj.y":
                            target_size[size_dim] = mj_obj.size[1] + 0.005
                    inner_size = np.min((outer_size, target_size), axis=0)
                else:
                    inner_size = outer_size

                inner_xpos, inner_ypos = placement.get("pos", (None, None))
                offset = placement.get("offset", (0.0, 0.0))

                # center inner region within outer region
                if inner_xpos == "ref":
                    # compute optimal placement of inner region to match up with the reference fixture
                    x_halfsize = outer_size[0] / 2 - inner_size[0] / 2
                    if x_halfsize == 0.0:
                        inner_xpos = 0.0
                    else:
                        ref_fixture = self.get_fixture(
                            placement["sample_region_kwargs"]["ref"]
                        )
                        ref_pos = ref_fixture.pos
                        fixture_to_ref = OU.get_rel_transform(fixture, ref_fixture)[0]
                        outer_to_ref = fixture_to_ref - reset_region["offset"]
                        inner_xpos = outer_to_ref[0] / x_halfsize
                        inner_xpos = np.clip(inner_xpos, a_min=-1.0, a_max=1.0)
                elif inner_xpos is None:
                    inner_xpos = 0.0

                if inner_ypos is None:
                    inner_ypos = 0.0
                # offset for inner region
                intra_offset = (
                    (outer_size[0] / 2 - inner_size[0] / 2) * inner_xpos + offset[0],
                    (outer_size[1] / 2 - inner_size[1] / 2) * inner_ypos + offset[1],
                )

                # center surface point of entire region
                ref_pos = fixture.pos + [0, 0, reset_region["offset"][2]]
                ref_rot = fixture.rot

                # x, y, and rotational ranges for randomization
                x_range = (
                    np.array([-inner_size[0] / 2, inner_size[0] / 2])
                    + reset_region["offset"][0]
                    + intra_offset[0]
                )
                y_range = (
                    np.array([-inner_size[1] / 2, inner_size[1] / 2])
                    + reset_region["offset"][1]
                    + intra_offset[1]
                )
                rotation = placement.get("rotation", np.array([-np.pi / 4, np.pi / 4]))
            else:
                target_size = placement.get("size", None)
                x_range = np.array([-target_size[0] / 2, target_size[0] / 2])
                y_range = np.array([-target_size[1] / 2, target_size[1] / 2])
                rotation = placement.get("rotation", np.array([-np.pi / 4, np.pi / 4]))
                ref_pos = [0, 0, 0]
                ref_rot = 0.0

            if macros.SHOW_SITES is True:
                """
                show outer reset region
                """
                pos_to_vis = deepcopy(ref_pos)
                pos_to_vis[:2] += T.rotate_2d_point(
                    [reset_region["offset"][0], reset_region["offset"][1]], rot=ref_rot
                )
                size_to_vis = np.concatenate(
                    [
                        np.abs(
                            T.rotate_2d_point(
                                [outer_size[0] / 2, outer_size[1] / 2], rot=ref_rot
                            )
                        ),
                        [0.001],
                    ]
                )
                site_str = """<site type="box" rgba="0 0 1 0.4" size="{size}" pos="{pos}" name="reset_region_outer_{postfix}"/>""".format(
                    pos=array_to_string(pos_to_vis),
                    size=array_to_string(size_to_vis),
                    postfix=str(obj_i),
                )
                site_tree = ET.fromstring(site_str)
                self.model.worldbody.append(site_tree)

                """
                show inner reset region
                """
                pos_to_vis = deepcopy(ref_pos)
                pos_to_vis[:2] += T.rotate_2d_point(
                    [np.mean(x_range), np.mean(y_range)], rot=ref_rot
                )
                size_to_vis = np.concatenate(
                    [
                        np.abs(
                            T.rotate_2d_point(
                                [
                                    (x_range[1] - x_range[0]) / 2,
                                    (y_range[1] - y_range[0]) / 2,
                                ],
                                rot=ref_rot,
                            )
                        ),
                        [0.002],
                    ]
                )
                site_str = """<site type="box" rgba="1 0 0 0.4" size="{size}" pos="{pos}" name="reset_region_inner_{postfix}"/>""".format(
                    pos=array_to_string(pos_to_vis),
                    size=array_to_string(size_to_vis),
                    postfix=str(obj_i),
                )
                site_tree = ET.fromstring(site_str)
                self.model.worldbody.append(site_tree)

            placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name="{}_Sampler".format(cfg["name"]),
                    mujoco_objects=mj_obj,
                    x_range=x_range,
                    y_range=y_range,
                    rotation=rotation,
                    ensure_object_boundary_in_range=placement.get(
                        "ensure_object_boundary_in_range", True
                    ),
                    ensure_valid_placement=placement.get(
                        "ensure_valid_placement", True
                    ),
                    reference_pos=ref_pos,
                    reference_rot=ref_rot,
                    z_offset=z_offset,
                    rng=self.rng,
                    rotation_axis=placement.get("rotation_axis", "z"),
                ),
                sample_args=placement.get("sample_args", None),
            )

        return placement_initializer

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset and self.placement_initializer is not None:
            # use pre-computed object placements
            object_placements = self.object_placements

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

        # step through a few timesteps to settle objects
        action = np.zeros(self.action_spec[0].shape)  # apply empty action

        # Since the env.step frequency is slower than the mjsim timestep frequency, the internal controller will output
        # multiple torque commands in between new high level action commands. Therefore, we need to denote via
        # 'policy_step' whether the current step we're taking is simply an internal update of the controller,
        # or an actual policy update
        policy_step = True

        # Loop through the simulation at the model timestep rate until we're ready to take the next policy step
        # (as defined by the control frequency specified at the environment level)
        for i in range(10 * int(self.control_timestep / self.model_timestep)):
            self.sim.step1()
            self._pre_action(action, policy_step)
            self.sim.step2()
            policy_step = False

    def _get_obj_cfgs(self):
        """
        Returns a list of object configurations to use in the environment.
        The object configurations are usually environment-specific and should
        be implemented in the subclass.

        Returns:
            list: list of object configurations
        """

        return []

    def get_ep_meta(self):
        """
        Returns a dictionary containing episode meta data
        """

        def copy_dict_for_json(orig_dict):
            new_dict = {}
            for (k, v) in orig_dict.items():
                if isinstance(v, dict):
                    new_dict[k] = copy_dict_for_json(v)
                elif isinstance(v, Fixture):
                    new_dict[k] = v.name
                else:
                    new_dict[k] = v
            return new_dict

        ep_meta = super().get_ep_meta()
        ep_meta["layout_id"] = self.layout_id
        ep_meta["style_id"] = self.style_id
        ep_meta["object_cfgs"] = [copy_dict_for_json(cfg) for cfg in self.object_cfgs]
        ep_meta["fixtures"] = {
            k: {"cls": v.__class__.__name__} for (k, v) in self.fixtures.items()
        }
        ep_meta["gen_textures"] = self._curr_gen_fixtures or {}
        ep_meta["lang"] = ""
        ep_meta["fixture_refs"] = dict(
            {k: v.name for (k, v) in self.fixture_refs.items()}
        )
        ep_meta["cam_configs"] = deepcopy(self._cam_configs)

        return ep_meta

    def get_privileged_information(self, trajectory_horizon: int = 64) -> Dict[str, Any]:
        """
        Return privileged scene metadata and current simulator state.

        The structure mirrors the mshab helper at a higher level:
        - ``static`` contains task / layout / object / fixture metadata that is
          mostly fixed for the episode.
        - ``dynamic`` contains the latest robot / object / fixture state.
        """

        def _to_serializable(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (list, tuple)):
                return [_to_serializable(v) for v in value]
            if isinstance(value, dict):
                return {str(k): _to_serializable(v) for k, v in value.items()}
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return repr(value)

        def _safe_call(fn, *args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception:
                return None

        def _safe_attr(obj, attr, default=None):
            try:
                return getattr(obj, attr, default)
            except Exception:
                return default

        def _site_local_pos(site: Any) -> Any:
            if site is None:
                return None
            try:
                raw = site.get("pos")
            except Exception:
                raw = None
            if raw is None:
                return None
            try:
                return string_to_array(raw)
            except Exception:
                return raw

        def _body_id(name: Optional[str]) -> Optional[int]:
            if not name:
                return None
            return _safe_call(self.sim.model.body_name2id, name)

        def _site_id(name: Optional[str]) -> Optional[int]:
            if not name:
                return None
            return _safe_call(self.sim.model.site_name2id, name)

        def _joint_qpos(name: str):
            addr = _safe_call(self.sim.model.get_joint_qpos_addr, name)
            if addr is None:
                return None
            try:
                return _to_serializable(self.sim.data.qpos[addr])
            except Exception:
                return None

        def _joint_qvel(name: str):
            addr = _safe_call(self.sim.model.get_joint_qvel_addr, name)
            if addr is None:
                return None
            try:
                return _to_serializable(self.sim.data.qvel[addr])
            except Exception:
                return None

        def _pose_from_body_id(body_id: Optional[int]) -> Optional[Dict[str, Any]]:
            if body_id is None:
                return None
            try:
                quat_xyzw = T.convert_quat(self.sim.data.body_xquat[body_id], to="xyzw")
                return dict(
                    position=_to_serializable(self.sim.data.body_xpos[body_id]),
                    orientation=_to_serializable(quat_xyzw),
                )
            except Exception:
                return None

        def _pose_from_site_id(site_id: Optional[int]) -> Optional[Dict[str, Any]]:
            if site_id is None:
                return None
            try:
                pos = self.sim.data.site_xpos[site_id]
                rot = self.sim.data.site_xmat[site_id].reshape(3, 3)
                quat_xyzw = Rotation.from_matrix(rot).as_quat()
                return dict(
                    position=_to_serializable(pos),
                    orientation=_to_serializable(quat_xyzw),
                )
            except Exception:
                return None

        def _velocity_from_site_id(site_id: Optional[int]) -> Optional[Dict[str, Any]]:
            if site_id is None:
                return None
            try:
                return dict(
                    linear=_to_serializable(self.sim.data.site_xvelp[site_id]),
                    angular=_to_serializable(self.sim.data.site_xvelr[site_id]),
                )
            except Exception:
                return None

        def _end_effector_velocity(
            site_id: Optional[int],
            body_id: Optional[int],
            pose: Optional[Dict[str, Any]] = None,
        ) -> Optional[Dict[str, Any]]:
            site_velocity = _velocity_from_site_id(site_id)
            if site_velocity is not None:
                return site_velocity

            body_velocity = _velocity_from_body_id(body_id)
            if body_velocity is not None:
                return body_velocity

            if pose is None:
                return None
            try:
                prev_pose = getattr(self, "_privileged_prev_eef_pose", None)
                prev_time = getattr(self, "_privileged_prev_time", None)
                curr_time = float(getattr(self.sim.data, "time", np.nan))
                curr_pos = np.asarray(pose["position"], dtype=float)
                if (
                    prev_pose is not None
                    and prev_time is not None
                    and np.isfinite(curr_time)
                    and curr_time > float(prev_time)
                ):
                    prev_pos = np.asarray(prev_pose["position"], dtype=float)
                    linear = (curr_pos - prev_pos) / float(curr_time - float(prev_time))
                    return dict(
                        linear=_to_serializable(linear),
                        angular=None,
                    )
            except Exception:
                pass
            return None

        def _velocity_from_body_id(body_id: Optional[int]) -> Optional[Dict[str, Any]]:
            if body_id is None:
                return None
            try:
                return dict(
                    linear=_to_serializable(self.sim.data.body_xvelp[body_id]),
                    angular=_to_serializable(self.sim.data.body_xvelr[body_id]),
                )
            except Exception:
                try:
                    body_name = self.sim.model.body_id2name(body_id)
                except Exception:
                    body_name = None
                if body_name:
                    linear = _safe_call(self.sim.data.get_body_xvelp, body_name)
                    angular = _safe_call(self.sim.data.get_body_xvelr, body_name)
                    if linear is not None or angular is not None:
                        return dict(
                            linear=_to_serializable(linear),
                            angular=_to_serializable(angular),
                        )
                return None

        def _pose_from_camera_name(camera_name: Optional[str]) -> Optional[Dict[str, Any]]:
            if not camera_name:
                return None
            try:
                cam_id = int(self.sim.model.camera_name2id(camera_name))
                pos = np.asarray(self.sim.data.cam_xpos[cam_id], dtype=float)
                rot = np.asarray(self.sim.data.cam_xmat[cam_id], dtype=float).reshape(3, 3)
                quat_xyzw = Rotation.from_matrix(rot).as_quat()
                return dict(
                    position=_to_serializable(pos),
                    orientation=_to_serializable(quat_xyzw),
                )
            except Exception:
                return None

        def _camera_dynamic() -> Dict[str, Any]:
            result = {}
            for camera_name, camera_cfg in (getattr(self, "_cam_configs", {}) or {}).items():
                if not isinstance(camera_cfg, dict):
                    continue
                result[camera_name] = dict(
                    pose=_pose_from_camera_name(camera_name),
                    fovy=_safe_call(
                        lambda: float(
                            self.sim.model.cam_fovy[int(self.sim.model.camera_name2id(camera_name))]
                        )
                    ),
                )
            return result

        def _box_corners(half_extents: np.ndarray) -> np.ndarray:
            hx, hy, hz = np.asarray(half_extents, dtype=float).reshape(3)
            return np.array(
                [
                    [-hx, -hy, -hz],
                    [hx, -hy, -hz],
                    [hx, hy, -hz],
                    [-hx, hy, -hz],
                    [-hx, -hy, hz],
                    [hx, -hy, hz],
                    [hx, hy, hz],
                    [-hx, hy, hz],
                ],
                dtype=float,
            )

        def _geom_local_corners(geom_id: int) -> Optional[np.ndarray]:
            try:
                size = np.asarray(self.sim.model.geom_size[geom_id], dtype=float).reshape(-1)
            except Exception:
                size = np.empty((0,), dtype=float)
            try:
                geom_type = int(self.sim.model.geom_type[geom_id])
            except Exception:
                geom_type = -1
            local_corners = None
            if size.size >= 3 and np.all(np.isfinite(size[:3])) and np.any(np.abs(size[:3]) > 0.0):
                local_corners = _box_corners(np.abs(size[:3]))
            elif size.size == 2 and np.all(np.isfinite(size[:2])) and np.any(np.abs(size[:2]) > 0.0):
                local_corners = _box_corners(np.abs(np.array([size[0], size[0], size[1]], dtype=float)))
            elif size.size == 1 and np.isfinite(size[0]) and abs(size[0]) > 0.0:
                local_corners = _box_corners(np.full(3, abs(size[0]), dtype=float))

            # For mesh geoms, prefer the actual mesh vertices when available so the bounds
            # stay tight and don't inflate the parent/base link.
            if geom_type == 7:
                try:
                    mesh_id = int(self.sim.model.geom_dataid[geom_id])
                    vert_adr = int(self.sim.model.mesh_vertadr[mesh_id])
                    vert_num = int(self.sim.model.mesh_vertnum[mesh_id])
                    mesh_verts = np.asarray(
                        self.sim.model.mesh_vert[vert_adr : vert_adr + vert_num],
                        dtype=float,
                    )
                    if mesh_verts.size and size.size >= 3:
                        local_corners = mesh_verts * np.abs(size[:3])[None, :]
                except Exception:
                    pass

            if local_corners is None or local_corners.size == 0:
                return None

            try:
                geom_pos = np.asarray(self.sim.model.geom_pos[geom_id], dtype=float)
            except Exception:
                geom_pos = np.zeros(3, dtype=float)
            try:
                geom_quat = np.asarray(self.sim.model.geom_quat[geom_id], dtype=float)
                geom_rot = Rotation.from_quat(T.convert_quat(geom_quat, to="xyzw")).as_matrix()
            except Exception:
                geom_rot = np.eye(3, dtype=float)
            return local_corners @ geom_rot.T + geom_pos[None, :]

        def _body_local_bounds(body_id: Optional[int]) -> Optional[Dict[str, Any]]:
            if body_id is None:
                return None
            try:
                geom_start = int(self.sim.model.body_geomadr[body_id])
                geom_count = int(self.sim.model.body_geomnum[body_id])
            except Exception:
                return None
            if geom_count <= 0:
                return None
            mins = []
            maxs = []
            for geom_id in range(geom_start, geom_start + geom_count):
                geom_corners = _geom_local_corners(geom_id)
                if geom_corners is None:
                    continue
                mins.append(np.min(geom_corners, axis=0))
                maxs.append(np.max(geom_corners, axis=0))
            if not mins:
                return None
            bmin = np.min(np.stack(mins, axis=0), axis=0)
            bmax = np.max(np.stack(maxs, axis=0), axis=0)
            return dict(min=_to_serializable(bmin), max=_to_serializable(bmax))

        def _body_immediate_child_connector_bounds(
            body_id: Optional[int],
            *,
            min_half_extents: Iterable[float] = (0.02, 0.02, 0.02),
            pad: float = 0.01,
        ) -> Optional[Dict[str, Any]]:
            if body_id is None:
                return None
            try:
                parents = np.asarray(self.sim.model.body_parentid, dtype=int).reshape(-1)
                body_pos = np.asarray(self.sim.data.body_xpos[body_id], dtype=float)
                body_rot = np.asarray(self.sim.data.body_xmat[body_id], dtype=float).reshape(3, 3)
            except Exception:
                return None

            child_ids = [int(idx) for idx, parent in enumerate(parents) if int(parent) == int(body_id) and int(idx) != int(body_id)]
            if not child_ids:
                return None

            try:
                min_half = np.asarray(list(min_half_extents), dtype=float).reshape(3)
            except Exception:
                min_half = np.full(3, 0.02, dtype=float)

            local_points = [np.zeros(3, dtype=float)]
            body_rot_T = body_rot.T
            for child_id in child_ids:
                try:
                    child_pos = np.asarray(self.sim.data.body_xpos[child_id], dtype=float)
                except Exception:
                    continue
                child_local = (child_pos - body_pos) @ body_rot_T
                local_points.append(child_local)

            if len(local_points) <= 1:
                return None

            stacked = np.stack(local_points, axis=0)
            bmin = np.min(stacked, axis=0) - float(pad)
            bmax = np.max(stacked, axis=0) + float(pad)
            center = 0.5 * (bmin + bmax)
            half = np.maximum(0.5 * (bmax - bmin), min_half)
            return dict(
                min=_to_serializable(center - half),
                max=_to_serializable(center + half),
            )

        def _body_vertical_connector_bounds(
            body_id: Optional[int],
            *,
            target_child_names: Iterable[str],
            min_half_extents: Iterable[float] = (0.03, 0.03, 0.03),
            lateral_half_extents: Iterable[float] = (0.08, 0.08),
            pad_z: float = 0.02,
        ) -> Optional[Dict[str, Any]]:
            if body_id is None:
                return None
            try:
                parents = np.asarray(self.sim.model.body_parentid, dtype=int).reshape(-1)
                body_name = self.sim.model.body_id2name(body_id)
                body_pos = np.asarray(self.sim.data.body_xpos[body_id], dtype=float)
                body_rot = np.asarray(self.sim.data.body_xmat[body_id], dtype=float).reshape(3, 3)
            except Exception:
                return None
            target_child_names = set(target_child_names)
            child_points = []
            body_rot_T = body_rot.T
            for child_id, parent in enumerate(parents):
                if int(parent) != int(body_id) or int(child_id) == int(body_id):
                    continue
                try:
                    child_name = self.sim.model.body_id2name(int(child_id))
                except Exception:
                    child_name = None
                if child_name not in target_child_names:
                    continue
                try:
                    child_pos = np.asarray(self.sim.data.body_xpos[int(child_id)], dtype=float)
                except Exception:
                    continue
                child_points.append((child_pos - body_pos) @ body_rot_T)
            if not child_points:
                return None
            try:
                min_half = np.asarray(list(min_half_extents), dtype=float).reshape(3)
                lateral_half = np.asarray(list(lateral_half_extents), dtype=float).reshape(2)
            except Exception:
                min_half = np.full(3, 0.03, dtype=float)
                lateral_half = np.full(2, 0.08, dtype=float)
            stacked = np.stack([np.zeros(3, dtype=float)] + child_points, axis=0)
            bmin = np.min(stacked, axis=0)
            bmax = np.max(stacked, axis=0)
            center_xy = 0.5 * (bmin[:2] + bmax[:2])
            half_xy = np.maximum(0.5 * (bmax[:2] - bmin[:2]), lateral_half)
            center_z = 0.5 * (bmin[2] + bmax[2])
            half_z = max(0.5 * (bmax[2] - bmin[2]) + float(pad_z), float(min_half[2]))
            center = np.array([center_xy[0], center_xy[1], center_z], dtype=float)
            half = np.array([half_xy[0], half_xy[1], half_z], dtype=float)
            return dict(
                min=_to_serializable(center - half),
                max=_to_serializable(center + half),
            )

        def _robot_subtree_body_ids(root_body_id: Optional[int]) -> List[int]:
            if root_body_id is None:
                return []
            try:
                parents = np.asarray(self.sim.model.body_parentid, dtype=int).reshape(-1)
            except Exception:
                return [int(root_body_id)]
            body_ids: List[int] = []
            for body_id in range(len(parents)):
                cur = body_id
                while cur >= 0:
                    if cur == root_body_id:
                        body_ids.append(int(body_id))
                        break
                    parent = int(parents[cur])
                    if parent == cur:
                        break
                    cur = parent
            return body_ids

        def _body_subtree_local_bounds(body_id: Optional[int]) -> Optional[Dict[str, Any]]:
            if body_id is None:
                return None
            try:
                root_pos = np.asarray(self.sim.data.body_xpos[body_id], dtype=float)
                root_rot = np.asarray(self.sim.data.body_xmat[body_id], dtype=float).reshape(3, 3)
            except Exception:
                return None
            mins = []
            maxs = []
            root_rot_T = root_rot.T
            for sub_body_id in _robot_subtree_body_ids(body_id):
                try:
                    geom_start = int(self.sim.model.body_geomadr[sub_body_id])
                    geom_count = int(self.sim.model.body_geomnum[sub_body_id])
                except Exception:
                    continue
                for geom_id in range(geom_start, geom_start + geom_count):
                    half_extents = _geom_half_extents(geom_id)
                    if half_extents is None:
                        continue
                    try:
                        geom_pos = np.asarray(self.sim.data.geom_xpos[geom_id], dtype=float)
                        geom_rot = np.asarray(self.sim.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
                    except Exception:
                        continue
                    local_corners = np.array(
                        [
                            [-half_extents[0], -half_extents[1], -half_extents[2]],
                            [half_extents[0], -half_extents[1], -half_extents[2]],
                            [half_extents[0], half_extents[1], -half_extents[2]],
                            [-half_extents[0], half_extents[1], -half_extents[2]],
                            [-half_extents[0], -half_extents[1], half_extents[2]],
                            [half_extents[0], -half_extents[1], half_extents[2]],
                            [half_extents[0], half_extents[1], half_extents[2]],
                            [-half_extents[0], half_extents[1], half_extents[2]],
                        ],
                        dtype=float,
                    )
                    world_corners = local_corners @ geom_rot.T + geom_pos[None, :]
                    root_local_corners = (world_corners - root_pos[None, :]) @ root_rot_T
                    mins.append(np.min(root_local_corners, axis=0))
                    maxs.append(np.max(root_local_corners, axis=0))
            if not mins:
                return None
            bmin = np.min(np.stack(mins, axis=0), axis=0)
            bmax = np.max(np.stack(maxs, axis=0), axis=0)
            return dict(min=_to_serializable(bmin), max=_to_serializable(bmax))

        def _site_local_box_bounds(body_id: Optional[int], site_id: Optional[int], half_extents: Iterable[float]) -> Optional[Dict[str, Any]]:
            if body_id is None or site_id is None:
                return None
            try:
                body_pos = np.asarray(self.sim.data.body_xpos[body_id], dtype=float)
                body_rot = np.asarray(self.sim.data.body_xmat[body_id], dtype=float).reshape(3, 3)
                site_pos = np.asarray(self.sim.data.site_xpos[site_id], dtype=float)
                half = np.asarray(list(half_extents), dtype=float).reshape(3)
            except Exception:
                return None
            corners = np.array(
                [
                    [-half[0], -half[1], -half[2]],
                    [half[0], -half[1], -half[2]],
                    [half[0], half[1], -half[2]],
                    [-half[0], half[1], -half[2]],
                    [-half[0], -half[1], half[2]],
                    [half[0], -half[1], half[2]],
                    [half[0], half[1], half[2]],
                    [-half[0], half[1], half[2]],
                ],
                dtype=float,
            )
            world_corners = corners + site_pos[None, :]
            local_corners = (world_corners - body_pos[None, :]) @ body_rot.T
            return dict(
                min=_to_serializable(np.min(local_corners, axis=0)),
                max=_to_serializable(np.max(local_corners, axis=0)),
            )

        def _body_external_wrench(body_id: Optional[int]) -> Optional[Dict[str, Any]]:
            if body_id is None:
                return None
            try:
                wrench = np.asarray(self.sim.data.cfrc_ext[body_id], dtype=float).reshape(-1)
            except Exception:
                return None
            if wrench.size < 6 or not np.all(np.isfinite(wrench[:6])):
                return None
            force = wrench[:3]
            torque = wrench[3:6]
            return dict(
                force=_to_serializable(force),
                torque=_to_serializable(torque),
                force_norm=float(np.linalg.norm(force)),
                torque_norm=float(np.linalg.norm(torque)),
            )

        def _robot_joint_bundle() -> Dict[str, Any]:
            joint_names: List[str] = []
            seen = set()

            def _append_joint(name: Optional[str]) -> None:
                if not name or name in seen:
                    return
                seen.add(name)
                joint_names.append(name)

            for name in (_safe_attr(robot, "robot_joints", []) or []):
                _append_joint(name)

            gripper_obj = _safe_attr(robot, "gripper")
            if isinstance(gripper_obj, dict):
                grippers = list(gripper_obj.values())
            elif gripper_obj is None:
                grippers = []
            else:
                grippers = [gripper_obj]
            for gripper in grippers:
                for attr_name in ("joints", "joint_names"):
                    for name in (_safe_attr(gripper, attr_name, []) or []):
                        _append_joint(name)

            return dict(
                joint_names=_to_serializable(joint_names),
                joint_positions=_to_serializable([_joint_qpos(name) for name in joint_names]),
                joint_velocities=_to_serializable([_joint_qvel(name) for name in joint_names]),
            )

        def _physics_dynamic(robot_link_meta: Dict[str, Any], eef_body_name: Optional[str]) -> Dict[str, Any]:
            robot_body_ids = {
                int(info.get("body_id"))
                for info in robot_link_meta.values()
                if isinstance(info, dict) and info.get("body_id") is not None
            }
            geom_bodyid = None
            try:
                geom_bodyid = np.asarray(self.sim.model.geom_bodyid, dtype=int).reshape(-1)
            except Exception:
                geom_bodyid = None

            total_contact_count = int(_safe_attr(self.sim.data, "ncon", 0) or 0)
            robot_contact_count = 0
            for contact_idx in range(total_contact_count):
                try:
                    contact = self.sim.data.contact[contact_idx]
                    geom1 = int(contact.geom1)
                    geom2 = int(contact.geom2)
                except Exception:
                    continue
                body1 = int(geom_bodyid[geom1]) if geom_bodyid is not None and 0 <= geom1 < len(geom_bodyid) else None
                body2 = int(geom_bodyid[geom2]) if geom_bodyid is not None and 0 <= geom2 < len(geom_bodyid) else None
                if body1 in robot_body_ids or body2 in robot_body_ids:
                    robot_contact_count += 1

            per_link_force_norms = {}
            max_robot_link_force = np.nan
            robot_force_sum = 0.0
            end_effector_force_norm = np.nan
            for name, info in robot_link_meta.items():
                body_id = info.get("body_id") if isinstance(info, dict) else None
                wrench = _body_external_wrench(body_id)
                if wrench is None:
                    continue
                force_norm = float(wrench["force_norm"])
                per_link_force_norms[name] = force_norm
                robot_force_sum += force_norm
                if not np.isfinite(max_robot_link_force) or force_norm > max_robot_link_force:
                    max_robot_link_force = force_norm
                if eef_body_name and name == eef_body_name:
                    end_effector_force_norm = force_norm

            return dict(
                total_contact_count=total_contact_count,
                robot_contact_count=robot_contact_count,
                max_robot_link_force=max_robot_link_force,
                robot_force_sum=robot_force_sum,
                end_effector_force_norm=end_effector_force_norm,
                per_link_force_norms=_to_serializable(per_link_force_norms),
            )

        def _robot_link_static(body_id: int, *, fallback_bounds: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
            try:
                body_name = self.sim.model.body_id2name(body_id)
            except Exception:
                body_name = None
            if not body_name:
                return None
            direct_bounds = _body_local_bounds(body_id)
            structural_connector_bounds = None
            if direct_bounds is None and fallback_bounds is None:
                if body_name == "mobilebase0_base":
                    structural_connector_bounds = _body_vertical_connector_bounds(
                        body_id,
                        target_child_names=("mobilebase0_support",),
                        lateral_half_extents=(0.11, 0.11),
                        pad_z=0.03,
                    )
                elif body_name == "robot0_base":
                    structural_connector_bounds = _body_vertical_connector_bounds(
                        body_id,
                        target_child_names=("robot0_link0",),
                        lateral_half_extents=(0.06, 0.06),
                        pad_z=0.02,
                    )
            connector_bounds = (
                _body_immediate_child_connector_bounds(body_id)
                if direct_bounds is None and fallback_bounds is None and structural_connector_bounds is None
                else None
            )
            visual_bounds = direct_bounds or fallback_bounds or structural_connector_bounds or connector_bounds
            if direct_bounds is not None:
                visual_bounds_source = "direct_geom"
            elif fallback_bounds is not None:
                visual_bounds_source = "eef_site_fallback"
            elif structural_connector_bounds is not None:
                visual_bounds_source = "structural_connector"
            elif connector_bounds is not None:
                visual_bounds_source = "immediate_child_connector"
            else:
                visual_bounds_source = None
            return dict(
                name=body_name,
                body_id=int(body_id),
                visual_bounds=visual_bounds,
                visual_bounds_source=visual_bounds_source,
            )

        def _relative_pose_to_eef(
            target_pose: Optional[Dict[str, Any]], eef_pose: Optional[Dict[str, Any]]
        ) -> Optional[Dict[str, Any]]:
            if target_pose is None or eef_pose is None:
                return None
            try:
                target_mat = T.pose2mat(
                    (
                        np.asarray(target_pose["position"]),
                        np.asarray(target_pose["orientation"]),
                    )
                )
                eef_inv = T.pose_inv(
                    T.pose2mat(
                        (
                            np.asarray(eef_pose["position"]),
                            np.asarray(eef_pose["orientation"]),
                        )
                    )
                )
                rel = T.pose_in_A_to_pose_in_B(target_mat, eef_inv)
                rel_pos, rel_quat = T.mat2pose(rel)
                return dict(
                    position=_to_serializable(rel_pos),
                    orientation=_to_serializable(rel_quat),
                )
            except Exception:
                return None

        def _fixture_joint_state(fixture) -> Optional[Dict[str, Any]]:
            joints = _safe_attr(fixture, "_joints", []) or []
            if not joints:
                return None
            result = {}
            for joint_suffix in joints:
                joint_name = f"{fixture.name}_{joint_suffix}"
                result[joint_name] = dict(
                    qpos=_joint_qpos(joint_name),
                    qvel=_joint_qvel(joint_name),
                )
            return result or None

        def _fixture_static(fixture) -> Dict[str, Any]:
            local_bounds = {}
            for key, site in (_safe_attr(fixture, "_bounds_sites", {}) or {}).items():
                local_bounds[key] = _to_serializable(_site_local_pos(site))
            return dict(
                class_name=fixture.__class__.__name__,
                nat_lang=_safe_attr(fixture, "nat_lang"),
                size=_to_serializable(_safe_attr(fixture, "size")),
                position=_to_serializable(_safe_attr(fixture, "pos")),
                quaternion=_to_serializable(_safe_attr(fixture, "quat")),
                euler=_to_serializable(_safe_attr(fixture, "euler")),
                origin_offset=_to_serializable(_safe_attr(fixture, "origin_offset")),
                placement=_to_serializable(_safe_attr(fixture, "_placement")),
                local_bounds_sites=local_bounds or None,
            )

        def _fixture_dynamic(fixture) -> Dict[str, Any]:
            get_state_fn = _safe_attr(fixture, "get_state")
            get_door_state_fn = _safe_attr(fixture, "get_door_state")
            get_site_info_fn = _safe_attr(fixture, "get_site_info")

            state = _safe_call(get_state_fn, self) if callable(get_state_fn) else None
            if state is None:
                state = _safe_call(get_state_fn, self.sim) if callable(get_state_fn) else None
            if state is None:
                state = _safe_call(get_state_fn) if callable(get_state_fn) else None
            door_state = (
                _safe_call(get_door_state_fn, self)
                if callable(get_door_state_fn)
                else None
            )
            site_info = (
                _safe_call(get_site_info_fn, self.sim)
                if callable(get_site_info_fn)
                else None
            )
            return dict(
                state=_to_serializable(state),
                door_state=_to_serializable(door_state),
                joints=_fixture_joint_state(fixture),
                sites=_to_serializable(site_info),
            )

        def _object_static(name: str, model, cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            bbox_points = _safe_call(model.get_bbox_points)
            bbox_min = None
            bbox_max = None
            object_info = cfg.get("info") if isinstance(cfg, dict) else None
            if bbox_points is not None:
                try:
                    bbox_points = np.asarray(bbox_points, dtype=float)
                    bbox_min = bbox_points.min(axis=0)
                    bbox_max = bbox_points.max(axis=0)
                except Exception:
                    bbox_min = None
                    bbox_max = None
            return dict(
                root_body=_safe_attr(model, "root_body"),
                info=_to_serializable(object_info),
                category=(
                    object_info.get("cat")
                    if isinstance(object_info, dict)
                    else None
                ),
                config=_to_serializable(cfg),
                mjcf_path=_safe_attr(model, "asset_path", _safe_attr(model, "mjcf_path")),
                scale=_to_serializable(_safe_attr(model, "scale")),
                horizontal_radius=_to_serializable(_safe_attr(model, "horizontal_radius")),
                bbox=dict(
                    min=_to_serializable(bbox_min),
                    max=_to_serializable(bbox_max),
                )
                if bbox_min is not None
                else None,
            )

        robot = self.robots[0]
        eef_site_name = None
        try:
            eef_site_id = robot.eef_site_id["right"]
        except Exception:
            eef_site_id = None
        if eef_site_id is not None:
            try:
                eef_site_name = self.sim.model.site_id2name(eef_site_id)
            except Exception:
                eef_site_name = None

        robot_root_body = _safe_attr(robot.robot_model, "root_body")
        robot_root_body_id = _body_id(robot_root_body)
        robot_root_pose = _pose_from_body_id(robot_root_body_id)
        eef_body_name = None
        eef_body_id = None
        if eef_site_id is not None:
            try:
                eef_body_id = int(self.sim.model.site_bodyid[eef_site_id])
                eef_body_name = self.sim.model.body_id2name(eef_body_id)
            except Exception:
                eef_body_name = None
                eef_body_id = None
        eef_fallback_bounds = _site_local_box_bounds(
            eef_body_id,
            eef_site_id,
            half_extents=(0.035, 0.025, 0.025),
        )
        robot_link_meta = {}
        for body_id in _robot_subtree_body_ids(robot_root_body_id):
            link_info = _robot_link_static(
                body_id,
                fallback_bounds=(
                    eef_fallback_bounds
                    if eef_body_id is not None and int(body_id) == int(eef_body_id)
                    else None
                ),
            )
            if link_info is None:
                continue
            robot_link_meta[link_info["name"]] = link_info

        if not hasattr(self, "_privileged_static_cache"):
            ep_meta = self.get_ep_meta()
            object_cfg_by_name = {
                cfg["name"]: cfg for cfg in getattr(self, "object_cfgs", []) if "name" in cfg
            }
            self._privileged_static_cache = dict(
                task=dict(
                    env_name=self.__class__.__name__,
                    language=_to_serializable(ep_meta.get("lang", "")),
                    episode_meta=_to_serializable(ep_meta),
                ),
                robot=dict(
                    robot_class=self.robots[0].__class__.__name__,
                    robot_model_class=self.robots[0].robot_model.__class__.__name__,
                    naming_prefix=_safe_attr(self.robots[0].robot_model, "naming_prefix"),
                    init_base_pos=_to_serializable(
                        _safe_attr(self.robots[0].robot_model, "base_xpos")
                        or (robot_root_pose or {}).get("position")
                    ),
                    link_metadata=_to_serializable(robot_link_meta),
                    base_link=(
                        dict(
                            name=robot_root_body,
                            info=_to_serializable(robot_link_meta.get(robot_root_body)),
                        )
                        if robot_root_body
                        else None
                    ),
                    end_effector_link=(
                        dict(
                            name=eef_body_name,
                            info=_to_serializable(robot_link_meta.get(eef_body_name)),
                        )
                        if eef_body_name
                        else None
                    ),
                ),
                scene_layout=dict(
                    layout_id=int(self.layout_id),
                    style_id=int(self.style_id),
                    fixture_refs=_to_serializable(
                        {k: v.name for k, v in getattr(self, "fixture_refs", {}).items()}
                    ),
                    cameras=_to_serializable(getattr(self, "_cam_configs", {})),
                    fixtures={
                        name: _fixture_static(fixture)
                        for name, fixture in getattr(self, "fixtures", {}).items()
                    },
                    objects={
                        name: _object_static(
                            name,
                            model,
                            object_cfg_by_name.get(name),
                        )
                        for name, model in getattr(self, "objects", {}).items()
                    },
                ),
            )

        if not hasattr(self, "_privileged_history"):
            self._privileged_history = dict(eef_positions=[], robot_force_cumulative=0.0)
        eef_pose = _pose_from_site_id(eef_site_id)
        eef_velocity = _end_effector_velocity(eef_site_id, eef_body_id, eef_pose)
        if eef_pose is not None:
            self._privileged_history["eef_positions"].append(eef_pose["position"])
            if len(self._privileged_history["eef_positions"]) > trajectory_horizon:
                self._privileged_history["eef_positions"].pop(0)
        robot_joint_bundle = _robot_joint_bundle()
        physics = _physics_dynamic(robot_link_meta, eef_body_name)
        if np.isfinite(physics.get("robot_force_sum", np.nan)):
            self._privileged_history["robot_force_cumulative"] += float(physics["robot_force_sum"])

        dynamic = dict(
            task=dict(
                timestep=int(getattr(self, "timestep", -1)),
                success=bool(_safe_call(self._check_success)),
            ),
            robot=dict(
                root_body=robot_root_body,
                root_pose=_pose_from_body_id(robot_root_body_id),
                root_velocity=_velocity_from_body_id(robot_root_body_id),
                eef_site=eef_site_name,
                end_effector_pose=eef_pose,
                end_effector_velocity=eef_velocity,
                link_poses={
                    name: _pose_from_body_id(info.get("body_id"))
                    for name, info in robot_link_meta.items()
                },
                joint_names=robot_joint_bundle["joint_names"],
                joint_positions=robot_joint_bundle["joint_positions"],
                joint_velocities=robot_joint_bundle["joint_velocities"],
                gripper_action_dim=int(len(getattr(robot, "gripper", {}))),
                trajectory=_to_serializable(self._privileged_history["eef_positions"]),
            ),
            scene=dict(
                objects={},
                fixtures={},
                cameras=_camera_dynamic(),
            ),
            physics=dict(
                **physics,
                robot_force_cumulative=float(self._privileged_history["robot_force_cumulative"]),
            ),
        )

        for name, model in getattr(self, "objects", {}).items():
            body_id = _body_id(_safe_attr(model, "root_body"))
            pose = _pose_from_body_id(body_id)
            dynamic["scene"]["objects"][name] = dict(
                pose=pose,
                velocity=_velocity_from_body_id(body_id),
                relative_to_eef=_relative_pose_to_eef(pose, eef_pose),
            )

        for name, fixture in getattr(self, "fixtures", {}).items():
            dynamic["scene"]["fixtures"][name] = _fixture_dynamic(fixture)

        self._privileged_prev_eef_pose = eef_pose
        try:
            self._privileged_prev_time = float(getattr(self.sim.data, "time", np.nan))
        except Exception:
            self._privileged_prev_time = None

        return dict(
            static=self._privileged_static_cache,
            dynamic=dynamic,
        )

    def find_object_cfg_by_name(self, name):
        """
        Finds and returns the object configuration with the given name.

        Args:
            name (str): name of the object configuration to find

        Returns:
            dict: object configuration with the given name
        """
        for cfg in self.object_cfgs:
            if cfg["name"] == name:
                return cfg
        raise ValueError

    def set_cameras(self):
        """
        Adds new kitchen-relevant cameras to the environment. Will randomize cameras if specified.
        """
        self._cam_configs = CamUtils.get_robot_cam_configs(self.robots[0].name)
        if self.randomize_cameras:
            self._randomize_cameras()

        for (cam_name, cam_cfg) in self._cam_configs.items():
            if cam_cfg.get("parent_body", None) is not None:
                continue

            self.mujoco_arena.set_camera(
                camera_name=cam_name,
                pos=cam_cfg["pos"],
                quat=cam_cfg["quat"],
                camera_attribs=cam_cfg.get("camera_attribs", None),
            )

    def _randomize_cameras(self):
        """
        Randomizes the position and rotation of the wrist and agentview cameras.
        Note: This function is called only if randomize_cameras is set to True.
        """
        for camera in self._cam_configs:
            if "agentview" in camera:
                pos_noise = self.rng.normal(loc=0, scale=0.05, size=(1, 3))[0]
                euler_noise = self.rng.normal(loc=0, scale=3, size=(1, 3))[0]
            elif "eye_in_hand" in camera:
                pos_noise = np.zeros_like(pos_noise)
                euler_noise = np.zeros_like(euler_noise)
            else:
                # skip randomization for cameras not implemented
                continue

            old_pos = self._cam_configs[camera]["pos"]
            new_pos = [pos + n for pos, n in zip(old_pos, pos_noise)]
            self._cam_configs[camera]["pos"] = list(new_pos)

            old_euler = Rotation.from_quat(self._cam_configs[camera]["quat"]).as_euler(
                "xyz", degrees=True
            )
            new_euler = [eul + n for eul, n in zip(old_euler, euler_noise)]
            new_quat = Rotation.from_euler("xyz", new_euler, degrees=True).as_quat()
            self._cam_configs[camera]["quat"] = list(new_quat)

    def edit_model_xml(self, xml_str):
        """
        This function postprocesses the model.xml collected from a MuJoCo demonstration
        for retrospective model changes.

        Args:
            xml_str (str): Mujoco sim demonstration XML file as string

        Returns:
            str: Post-processed xml file as string
        """
        xml_str = super().edit_model_xml(xml_str)

        tree = ET.fromstring(xml_str)
        root = tree
        worldbody = root.find("worldbody")
        actuator = root.find("actuator")
        asset = root.find("asset")
        meshes = asset.findall("mesh")
        textures = asset.findall("texture")
        all_elements = meshes + textures

        robocasa_path_split = os.path.split(robocasa.__file__)[0].split("/")

        # replace robocasa-specific asset paths
        for elem in all_elements:
            old_path = elem.get("file")
            if old_path is None:
                continue

            old_path_split = old_path.split("/")
            # maybe replace all paths to robosuite assets
            if (
                ("models/assets/fixtures" in old_path)
                or ("models/assets/textures" in old_path)
                or ("models/assets/objects/objaverse" in old_path)
            ):
                if "/robosuite/" in old_path:
                    check_lst = [
                        loc
                        for loc, val in enumerate(old_path_split)
                        if val == "robosuite"
                    ]
                elif "/robocasa/" in old_path:
                    check_lst = [
                        loc
                        for loc, val in enumerate(old_path_split)
                        if val == "robocasa"
                    ]
                else:
                    raise ValueError

                ind = max(check_lst)  # last occurrence index
                new_path_split = robocasa_path_split + old_path_split[ind + 1 :]

                new_path = "/".join(new_path_split)
                elem.set("file", new_path)

        # set cameras
        for cam_name, cam_config in self._cam_configs.items():
            parent_body = cam_config.get("parent_body", None)

            cam_root = worldbody
            if parent_body is not None:
                cam_root = find_elements(
                    root=worldbody, tags="body", attribs={"name": parent_body}
                )
                if cam_root is None:
                    # camera config refers to body that doesnt exist on the robot
                    continue

            cam = find_elements(
                root=cam_root, tags="camera", attribs={"name": cam_name}
            )

            if cam is None:
                cam = ET.Element("camera")
                cam.set("mode", "fixed")
                cam.set("name", cam_name)
                cam_root.append(cam)

            cam.set("pos", array_to_string(cam_config["pos"]))
            cam.set("quat", array_to_string(cam_config["quat"]))
            for (k, v) in cam_config.get("camera_attribs", {}).items():
                cam.set(k, v)

        # replace base -> mobilebase (this is needed for old PandaOmron demos)
        for elem in find_elements(
            root=worldbody, tags=["geom", "site", "body", "joint"], return_first=False
        ):
            if elem.get("name") is None:
                continue
            if elem.get("name").startswith("base0_"):
                old_name = elem.get("name")
                new_name = "mobilebase0_" + old_name[6:]
                elem.set("name", new_name)
        for elem in find_elements(
            root=actuator,
            tags=["velocity", "position", "motor", "general"],
            return_first=False,
        ):
            if elem.get("name") is None:
                continue
            if elem.get("name").startswith("base0_"):
                old_name = elem.get("name")
                new_name = "mobilebase0_" + old_name[6:]
                elem.set("name", new_name)
        for elem in find_elements(
            root=actuator,
            tags=["velocity", "position", "motor", "general"],
            return_first=False,
        ):
            if elem.get("joint") is None:
                continue
            if elem.get("joint").startswith("base0_"):
                old_joint = elem.get("joint")
                new_joint = "mobilebase0_" + old_joint[6:]
                elem.set("joint", new_joint)

        # result = ET.tostring(root, encoding="utf8").decode("utf8")
        result = ET.tostring(root).decode("utf8")

        # replace with generative textures
        if (self.generative_textures is not None) and (
            self.generative_textures is not False
        ):
            # sample textures
            assert self.generative_textures == "100p"
            if self._curr_gen_fixtures is None or self._curr_gen_fixtures == {}:
                self._curr_gen_fixtures = get_random_textures(self.rng)

            cab_tex = self._curr_gen_fixtures["cab_tex"]
            counter_tex = self._curr_gen_fixtures["counter_tex"]
            wall_tex = self._curr_gen_fixtures["wall_tex"]
            floor_tex = self._curr_gen_fixtures["floor_tex"]

            result = replace_cab_textures(
                self.rng, result, new_cab_texture_file=cab_tex
            )
            result = replace_counter_top_texture(
                self.rng, result, new_counter_top_texture_file=counter_tex
            )
            result = replace_wall_texture(
                self.rng, result, new_wall_texture_file=wall_tex
            )
            result = replace_floor_texture(
                self.rng, result, new_floor_texture_file=floor_tex
            )

        return result

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        self.obj_body_id = {}
        for (name, model) in self.objects.items():
            self.obj_body_id[name] = self.sim.model.body_name2id(model.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information

        # Get robot prefix and define observables modality
        pf = self.robots[0].robot_model.naming_prefix
        modality = "object"

        # for conversion to relative gripper frame
        @sensor(modality=modality)
        def world_pose_in_gripper(obs_cache):
            return (
                T.pose_inv(
                    T.pose2mat((obs_cache[f"{pf}eef_pos"], obs_cache[f"{pf}eef_quat"]))
                )
                if f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache
                else np.eye(4)
            )

        sensors = [world_pose_in_gripper]
        names = ["world_pose_in_gripper"]
        actives = [False]

        # add ground-truth poses (absolute and relative to eef) for all objects
        for obj_name in self.obj_body_id:
            obj_sensors, obj_sensor_names = self._create_obj_sensors(
                obj_name=obj_name, modality=modality
            )
            sensors += obj_sensors
            names += obj_sensor_names
            actives += [True] * len(obj_sensors)

        # Create observables
        for name, s, active in zip(names, sensors, actives):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
                active=active,
            )

        return observables

    def _create_obj_sensors(self, obj_name, modality="object"):
        """
        Helper function to create sensors for a given object. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            obj_name (str): Name of object to create sensors for

            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """

        ### TODO: this was stolen from pick-place - do we want to move this into utils to share it? ###
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(
                self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw"
            )

        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):
            # Immediately return default value if cache is empty
            if any(
                [
                    name not in obs_cache
                    for name in [
                        f"{obj_name}_pos",
                        f"{obj_name}_quat",
                        "world_pose_in_gripper",
                    ]
                ]
            ):
                return np.zeros(3)
            obj_pose = T.pose2mat(
                (obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"])
            )
            rel_pose = T.pose_in_A_to_pose_in_B(
                obj_pose, obs_cache["world_pose_in_gripper"]
            )
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            return (
                obs_cache[f"{obj_name}_to_{pf}eef_quat"]
                if f"{obj_name}_to_{pf}eef_quat" in obs_cache
                else np.zeros(4)
            )

        sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
        names = [
            f"{obj_name}_pos",
            f"{obj_name}_quat",
            f"{obj_name}_to_{pf}eef_pos",
            f"{obj_name}_to_{pf}eef_quat",
        ]

        return sensors, names

    def _post_action(self, action):
        """
        Do any housekeeping after taking an action.

        Args:
            action (np.array): Action to execute within the environment

        Returns:
            3-tuple:
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) information about the current state of the environment
        """
        reward, done, info = super()._post_action(action)

        # Check if stove is turned on or not
        self.update_state()
        return reward, done, info

    def convert_rel_to_abs_action(self, rel_action):
        # if moving mobile base, there is no notion of absolute actions.
        # use relative actions instead.

        # if moving arm, get the absolute action
        robot = self.robots[0]
        robot.control(rel_action, policy_step=True)
        rel_pose = robot.composite_controller.part_controllers[
            "right"
        ].goal_origin_to_eef_pose()
        ac_pos, ac_ori = rel_pose[:3, 3], rel_pose[:3, :3]
        ac_ori = Rotation.from_matrix(ac_ori).as_rotvec()
        action_abs = np.hstack(
            [
                ac_pos,
                ac_ori,
                rel_action[6:],
            ]
        )
        return action_abs

    def update_state(self):
        """
        Updates the state of the environment.
        This involves updating the state of all fixtures in the environment.
        """
        super().update_state()

        for fixtr in self.fixtures.values():
            fixtr.update_state(self)

    def visualize(self, vis_settings):
        """
        In addition to super call, make the robot semi-transparent

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        visual_geom_names = []

        for robot in self.robots:
            robot_model = robot.robot_model
            visual_geom_names += robot_model.visual_geoms

        for name in visual_geom_names:
            rgba = self.sim.model.geom_rgba[self.sim.model.geom_name2id(name)]
            if self.translucent_robot:
                rgba[-1] = 0.10
            else:
                rgba[-1] = 1.0

    def reward(self, action=None):
        """
        Reward function for the task. The reward function is based on the task
        and to be implemented in the subclasses. Returns 0 by default.

        Returns:
            float: Reward for the task
        """
        reward = 0
        return reward

    def _check_success(self):
        """
        Checks if the task has been successfully completed.
        Success condition is based on the task and to be implemented in the
        subclasses. Returns False by default.

        Returns:
            bool: True if the task is successfully completed, False otherwise
        """
        return False

    def sample_object(
        self,
        groups,
        exclude_groups=None,
        graspable=None,
        microwavable=None,
        washable=None,
        cookable=None,
        freezable=None,
        split=None,
        obj_registries=None,
        max_size=(None, None, None),
        object_scale=None,
    ):
        """
        Sample a kitchen object from the specified groups and within max_size bounds.

        Args:
            groups (list or str): groups to sample from or the exact xml path of the object to spawn

            exclude_groups (str or list): groups to exclude

            graspable (bool): whether the sampled object must be graspable

            washable (bool): whether the sampled object must be washable

            microwavable (bool): whether the sampled object must be microwavable

            cookable (bool): whether whether the sampled object must be cookable

            freezable (bool): whether whether the sampled object must be freezable

            split (str): split to sample from. Split "A" specifies all but the last 3 object instances
                        (or the first half - whichever is larger), "B" specifies the  rest, and None
                        specifies all.

            obj_registries (tuple): registries to sample from

            max_size (tuple): max size of the object. If the sampled object is not within bounds of max size,
                            function will resample

            object_scale (float): scale of the object. If set will multiply the scale of the sampled object by this value


        Returns:
            dict: kwargs to apply to the MJCF model for the sampled object

            dict: info about the sampled object - the path of the mjcf, groups which the object's category belongs to,
            the category of the object the sampling split the object came from, and the groups the object was sampled from
        """
        return sample_kitchen_object(
            groups,
            exclude_groups=exclude_groups,
            graspable=graspable,
            washable=washable,
            microwavable=microwavable,
            cookable=cookable,
            freezable=freezable,
            rng=self.rng,
            obj_registries=(obj_registries or self.obj_registries),
            split=(split or self.obj_instance_split),
            max_size=max_size,
            object_scale=object_scale,
        )

    def _is_fxtr_valid(self, fxtr, size):
        """
        checks if counter is valid for object placement by making sure it is large enough

        Args:
            fxtr (Fixture): fixture to check
            size (tuple): minimum size (x,y) that the counter region must be to be valid

        Returns:
            bool: True if fixture is valid, False otherwise
        """
        for region in fxtr.get_reset_regions(self).values():
            if region["size"][0] >= size[0] and region["size"][1] >= size[1]:
                return True
        return False

    def get_fixture(self, id, ref=None, size=(0.2, 0.2)):
        """
        search fixture by id (name, object, or type)

        Args:
            id (str, Fixture, FixtureType): id of fixture to search for

            ref (str, Fixture, FixtureType): if specified, will search for fixture close to ref (within 0.10m)

            size (tuple): if sampling counter, minimum size (x,y) that the counter region must be

        Returns:
            Fixture: fixture object
        """
        # case 1: id refers to fixture object directly
        if isinstance(id, Fixture):
            return id
        # case 2: id refers to exact name of fixture
        elif id in self.fixtures.keys():
            return self.fixtures[id]

        if ref is None:
            # find all fixtures with names containing given name
            if isinstance(id, FixtureType) or isinstance(id, int):
                matches = [
                    name
                    for (name, fxtr) in self.fixtures.items()
                    if fixture_is_type(fxtr, id)
                ]
            else:
                matches = [name for name in self.fixtures.keys() if id in name]
            if id == FixtureType.COUNTER or id == FixtureType.COUNTER_NON_CORNER:
                matches = [
                    name
                    for name in matches
                    if self._is_fxtr_valid(self.fixtures[name], size)
                ]
            assert len(matches) > 0
            # sample random key
            key = self.rng.choice(matches)
            return self.fixtures[key]
        else:
            ref_fixture = self.get_fixture(ref)

            assert isinstance(id, FixtureType)
            cand_fixtures = []
            for fxtr in self.fixtures.values():
                if not fixture_is_type(fxtr, id):
                    continue
                if fxtr is ref_fixture:
                    continue
                if id == FixtureType.COUNTER:
                    fxtr_is_valid = self._is_fxtr_valid(fxtr, size)
                    if not fxtr_is_valid:
                        continue
                cand_fixtures.append(fxtr)

            # first, try to find fixture "containing" the reference fixture
            for fxtr in cand_fixtures:
                if OU.point_in_fixture(ref_fixture.pos, fxtr, only_2d=True):
                    return fxtr
            # if no fixture contains reference fixture, sample all close fixtures
            dists = [
                OU.fixture_pairwise_dist(ref_fixture, fxtr) for fxtr in cand_fixtures
            ]
            min_dist = np.min(dists)
            close_fixtures = [
                fxtr for (fxtr, d) in zip(cand_fixtures, dists) if d - min_dist < 0.10
            ]
            return self.rng.choice(close_fixtures)

    def register_fixture_ref(self, ref_name, fn_kwargs):
        """
        Registers a fixture reference for later use. Initializes the fixture
        if it has not been initialized yet.

        Args:
            ref_name (str): name of the reference

            fn_kwargs (dict): keyword arguments to pass to get_fixture

        Returns:
            Fixture: fixture object
        """
        if ref_name not in self.fixture_refs:
            self.fixture_refs[ref_name] = self.get_fixture(**fn_kwargs)
        return self.fixture_refs[ref_name]

    def get_obj_lang(self, obj_name="obj", get_preposition=False):
        """
        gets a formatted language string for the object (replaces underscores with spaces)

        Args:
            obj_name (str): name of object
            get_preposition (bool): if True, also returns preposition for object

        Returns:
            str: language string for object
        """
        obj_cfg = None
        for cfg in self.object_cfgs:
            if cfg["name"] == obj_name:
                obj_cfg = cfg
                break
        lang = obj_cfg["info"]["cat"].replace("_", " ")

        if not get_preposition:
            return lang

        if lang in ["bowl", "pot", "pan"]:
            preposition = "in"
        elif lang in ["plate"]:
            preposition = "on"
        else:
            raise ValueError

        return lang, preposition


class KitchenDemo(Kitchen):
    def __init__(
        self,
        init_robot_base_pos="cab_main_main_group",
        obj_groups="all",
        num_objs=1,
        *args,
        **kwargs,
    ):
        self.obj_groups = obj_groups
        self.num_objs = num_objs

        super().__init__(init_robot_base_pos=init_robot_base_pos, *args, **kwargs)

    def _get_obj_cfgs(self):
        cfgs = []

        for i in range(self.num_objs):
            cfgs.append(
                dict(
                    name="obj_{}".format(i),
                    obj_groups=self.obj_groups,
                    placement=dict(
                        fixture="counter_main_main_group",
                        sample_region_kwargs=dict(
                            ref="cab_main_main_group",
                        ),
                        size=(1.0, 1.0),
                        pos=(0.0, -1.0),
                    ),
                )
            )

        return cfgs
