import json
import gzip
import os
import attr
from typing import Any, Dict, List, Optional, Sequence

from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, ShortestPathPoint
from habitat.core.utils import (
    DatasetFloatJSONEncoder,
    not_none_validator
)
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
    ALL_SCENES_MASK,
    PointNavDatasetV1,
)
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal
)
from habitat.tasks.nav.object_nav_task import (
    ObjectGoal,
    ObjectGoalNavEpisode,
    ObjectViewLocation,
)


@attr.s(auto_attribs=True, kw_only=True)
class AttrObjNavEpisode(NavigationEpisode):
    r"""ObjectGoal Navigation Episode

    :param object_category: Category of the obect
    """
    object_category: Optional[str] = None
    goal_object_id: Optional[str] = None
    goal_image_id: Optional[str] = None
    goal_attributes: Optional[dict] = None

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals"""
        scene_short_id = os.path.basename(self.scene_id).split(".")[0]
        return f"{scene_short_id}_{self.goal_object_id}"


@attr.s(auto_attribs=True, kw_only=True)
class AttrObjGoal(NavigationGoal):
    r"""Object goal provides information about an object that is target for
    navigation. That can be specify object_id, position and object
    category. An important part for metrics calculation are view points that
     describe success area for the navigation.

    Args:
        object_id: id that can be used to retrieve object from the semantic
        scene annotation
        object_name: name of the object
        object_category: object category name usually similar to scene semantic
        categories
        room_id: id of a room where object is located, can be used to retrieve
        room from the semantic scene annotation
        room_name: name of the room, where object is located
        view_points: navigable positions around the object with specified
        proximity of the object surface used for navigation metrics calculation.
        The object is visible from these positions.
    """

    position: Optional[List[float]] = None
    object_id: Optional[int] = None
    object_name: Optional[str] = None
    object_category: Optional[str] = None
    object_surface_area: Optional[float] = None
    view_points: Optional[List[ObjectViewLocation]] = None
    image_goals: Optional[Any] = None


@registry.register_dataset(name="AttrObjNav-v1")
class AttrObjNavDatasetV1(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads Object Navigation dataset."""
    category_to_task_category_id: Dict[str, int]
    category_to_scene_annotation_category_id: Dict[str, int]
    episodes: List[ObjectGoalNavEpisode] = []  # type: ignore
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"
    goals_by_category: Dict[str, Sequence[ObjectGoal]]

    @staticmethod
    def dedup_goals(dataset: Dict[str, Any]) -> Dict[str, Any]:
        if len(dataset["episodes"]) == 0:
            return dataset

        # goals_by_category = {}
        # for i, ep in enumerate(dataset["episodes"]):
        #     ep_goal_id = "_".join((
        #         ep["scene_id"]['scene_id'].split("/")[-1].split(".")[0],
        #         ep["goal_object_id"]
        #     ))
        #     dataset["episodes"][i]["object_category"] = dataset["goals"][ep_goal_id][
        #         "object_category"
        #     ]
        #     ep = AttrObjNavEpisode(**ep)

        #     goals_key = ep.goals_key
        #     if goals_key not in goals_by_category:
        #         goals_by_category[goals_key] = ep.goals

        #     dataset["episodes"][i]["goals"] = []

        dataset["goals_by_category"] = dataset["goals"]

        return dataset

    def to_json(self) -> str:
        for i in range(len(self.episodes)):
            self.episodes[i].goals = []

        result = DatasetFloatJSONEncoder().encode(self)

        for i in range(len(self.episodes)):
            goals = self.goals_by_category[self.episodes[i].goals_key]
            if not isinstance(goals, list):
                goals = list(goals)
            self.episodes[i].goals = goals

        return result

    def __init__(self, config: Optional[Config] = None) -> None:
        self.goals_by_category = {}
        self.attribute_data = None
        self.episodes = []

        if config is None:
            return

        datasetfile_path = config.DATA_PATH.format(split=config.SPLIT)
        with gzip.open(datasetfile_path, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        # Read separate file for each scene
        dataset_dir = os.path.dirname(datasetfile_path)
        has_individual_scene_files = os.path.exists(
            self.content_scenes_path.split("{scene}")[0].format(
                data_path=dataset_dir
            )
        )
        if has_individual_scene_files:
            scenes = config.CONTENT_SCENES
            if ALL_SCENES_MASK in scenes:
                scenes = self._get_scenes_from_folder(
                    content_scenes_path=self.content_scenes_path,
                    dataset_dir=dataset_dir,
                )

            for scene in scenes:
                scene_filename = self.content_scenes_path.format(
                    data_path=dataset_dir, scene=scene.split("-")[-1]
                )
                with gzip.open(scene_filename, "rt") as f:
                    self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        else:
            self.episodes = list(
                filter(self.build_content_scenes_filter(config), self.episodes)
            )

        self.episodes = list(self.episodes)

    @staticmethod
    def __deserialize_goal(serialized_goal: Dict[str, Any]) -> ObjectGoal:
        g = AttrObjGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            view_location = ObjectViewLocation(**view)  # type: ignore
            view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore
            g.view_points[vidx] = view_location

        return g

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        if "category_to_task_category_id" in deserialized:
            self.category_to_task_category_id = deserialized[
                "category_to_task_category_id"
            ]

        if "category_to_scene_annotation_category_id" in deserialized:
            self.category_to_scene_annotation_category_id = deserialized[
                "category_to_scene_annotation_category_id"
            ]

        if "category_to_mp3d_category_id" in deserialized:
            self.category_to_scene_annotation_category_id = deserialized[
                "category_to_mp3d_category_id"
            ]

        if "attribute_data" in deserialized:
            self.attribute_data = deserialized["attribute_data"]

        # assert len(self.category_to_task_category_id) == len(
        #     self.category_to_scene_annotation_category_id
        # )

        # assert set(self.category_to_task_category_id.keys()) == set(
        #     self.category_to_scene_annotation_category_id.keys()
        # ), "category_to_task and category_to_mp3d must have the same keys"

        if len(deserialized["episodes"]) == 0:
            return

        if "goals_by_category" not in deserialized:
            deserialized = self.dedup_goals(deserialized)

        for k, v in deserialized["goals"].items():
            self.goals_by_category[k] = [self.__deserialize_goal(v)]

        for i, episode in enumerate(deserialized["episodes"]):
            episode = AttrObjNavEpisode(**episode)
            episode.episode_id = str(i)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.goals = self.goals_by_category[episode.goals_key]
            if self.attribute_data is not None:
                episode.goal_attributes = self.attribute_data[episode.goals_key]

            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        if point is None or isinstance(point, (int, str)):
                            point = {
                                "action": point,
                                "rotation": None,
                                "position": None,
                            }

                        path[p_index] = ShortestPathPoint(**point)

            self.episodes.append(episode)  # type: ignore [attr-defined]
