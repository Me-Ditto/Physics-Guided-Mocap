# Physics-Guided-Mocap


## Installation
Create conda environment and install dependencies.
```
conda create -n Physics-Guided-Mocap python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.1 -c pytorch -c nvidia # install pytorch
pip install -r requirements.txt
```

1. Due to the licenses, please download SMPL model file [here](http://smplify.is.tuebingen.mpg.de/).
2. The [Mujoco](https://github.com/openai/mujoco-py/releases) environment is built into the project path(./mujoco_py) to avoid cumbersome environment configuration. We recommend to download [mujoco-py-1.50.1.0](https://github.com/openai/mujoco-py/releases/tag/1.50.1.0) and [mjpro150](https://www.roboti.us/download.html) and [activation key](https://www.roboti.us/license.html) for win10/win11.




Finally put these data following the directory structure as below:
```
${ROOT}
|-- assets
    |-- mujoco_models
    |-- bigfoot_template.pkl
    |-- bigfoot_template_v1.pkl
|-- data
    |-- mujoco
        |-- mujoco-py-1.50.1.0
        |-- mjpro150
        |-- mjkey.txt
    |-- sample_data
        |-- amass_copycat_occlusion.pkl
        |-- amass_copycat_take5_test_small.pkl
        |-- standing_neutral.pkl
    |-- smpl
        |-- SMPL_NEUTRAL.pkl
        |-- SMPL_FEMALE.pkl
        |-- SMPL_MALE.pkl
    |-- init_data.pkl
    |-- iter_19000.p
    |-- J_regressor_h36m.npy
    |-- J_regressor_halpe.npy
    |-- J_regressor_lsp.npy
```





## Usage

### Demo
- Reconstruct physically-plausible human motions with physics-guided diffusion framework.

```
python demo.py
```



