# Download Dataset from PSL-InstanceNav

This webpage provides download links and preparation guidelines for the ECCV 2024 paper ["Prioritized Semantic Learning for Zero-shot Instance Navigation"](https://arxiv.org/abs/2403.11650)

## The Proposed InstanceNav Dataset (Text-Goal)
Previous works on the instance-level object navigation task focus on discrete graph-based environments and are limited by complicated human annotation processes. In this work, instead, we utilize the recent success of the Generative Vision-and-Language Models (GVLMs) to automatically build the open-vocabulary **text-goal** setting for the `InstanceNav` task on the popular HM3D environment. Specifically, we randomly select a goal view for each validation episode to render an image of the goal object in the Instance Image Navigation (IIN) dataset. Each episode in the IIN dataset corresponds to a unique goal object instance. To specify each object instance, we follow a previous attempt to separate text descriptions into two aspects: Intrinsic Attributes and Extrinsic Attributes. Intrinsic attributes cover inherent characteristics of the object, such as shape, color, and material. Extrinsic attributes describe the environment surrounding the object, which is used to determine instances with similar intrinsic attributes. We instruct a GVLM model (i.e., [CogVLM](https://github.com/THUDM/CogVLM)) to generate both types of attributes according to the instance image with a hand-crafted prompt. The original ground truth trajectories are preserved to construct the additional `InstanceNav` test set. In total, the test set of the **text-goal** setting comprises 1,000 episodes featuring 795 unique objects across 36 scenes.

### Dataset Specification
|File Name|Link|
|:-:|:-:|
|val_text.json.gz|[Google Drive](https://drive.google.com/uc?export=download&id=1KNdv6isX1FDZi4KCVPiECYDxijg9cZ3L)|


## Setup All Datasets Utilized in Our Paper
1. Follow the instructions [here](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md) to set up the `data/scene_datasets/` directory. `gibson` scenes can be found [here](http://gibsonenv.stanford.edu/database/).

2. Download the HM3D objectnav dataset from ZSON.
   ```bash
   wget https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip

   unzip objectnav_hm3d_v1.zip -d data/datasets/objectnav/

   # clean-up
   rm objectnav_hm3d_v1.zip
   ```

3. Download the HM3D instance navigation dataset.
   ```bash
   # download the original Instance Image Navigation dataset
   wget https://dl.fbaipublicfiles.com/habitat/data/datasets/imagenav/hm3d/v3/instance_imagenav_hm3d_v3.zip

   unzip instance_imagenav_hm3d_v3.zip -d data/datasets/

   # clean-up
   rm instance_imagenav_hm3d_v3.zip

   mkdir -p data/datasets/instancenav/val

   # download the attribute descriptions
   wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1KNdv6isX1FDZi4KCVPiECYDxijg9cZ3L" -O data/datasets/instancenav/val/val_text.json.gz

   export PROJECT_ROOT=`pwd`
   cd data/datasets/instancenav/val/
   ln -s $PROJECT_ROOT/data/datasets/instance_imagenav_hm3d_v3/val/content .
   cd $PROJECT_ROOT
   ```
