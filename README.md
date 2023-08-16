# \[IJCAI2023\] Physics-Guided Human Motion Capture with Pose Probability Modeling (Physics-Guided-Mocap)

The code for IJCAI 2023 paper "Physics-Guided Human Motion Capture with Pose Probability Modeling"<br>
Jingyi Ju, [Buzhen Huang](http://www.buzhenhuang.com/), Chen Zhu, Zhihao Li, [Yangang Wang](https://www.yangangwang.com/#me)<br>
\[[Paper](https://www.yangangwang.com/papers/IJCAI2023/jyj23_pgh.pdf)\]<br>

![figure](/images/pipeline.jpg)




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



### To Do:
* Training code
* Pretrained model




## Citation
If you find this code useful for your research, please consider citing the paper.
```
@inproceedings{Ju2023physics,
      title={Physics-Guided Human Motion Capture with Pose Probability Modeling}, 
      author={Jingyi Ju, Buzhen Huang, Chen Zhu, Zhihao Li and Yangang Wang},
      year={2023},
      booktitle={IJCAI},
}
```
## Reference
Some of the code is based on the following works. We gratefully appreciate the impact they have on our work.<br>

[UniversalHumanoidControl](https://github.com/ZhengyiLuo/UniversalHumanoidControl)<br>
[SMPL-X](https://github.com/vchoutas/smplify-x)<br>
[CLIFF](https://github.com/huawei-noah/noah-research/tree/master/CLIFF)<br>
