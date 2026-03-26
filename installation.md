### Setup for PHC
1. Setup the virtual environment

Create the virtual env
```
virtualenv phc -p python3.8
```
Activate the virtualenv
```
source phc/bin/activate
```
Install torch and cuda dependencies
```
pip install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

```
Install all the requirements
```
pip install -r PHC/requirement.txt
```

2. Download and setup [Isaac Gym](https://developer.nvidia.com/isaac-gym). 

3. Download SMPL paramters from [SMPL](https://smpl.is.tue.mpg.de/) and [SMPLX](https://smpl-x.is.tue.mpg.de/download.php). Put them in the `data/smpl` folder, unzip them into 'data/smpl' folder. For SMPL, please download the v1.1.0 version, which contains the neutral humanoid. Rename the files `basicmodel_neutral_lbs_10_207_0_v1.1.0`, `basicmodel_m_lbs_10_207_0_v1.1.0.pkl`, `basicmodel_f_lbs_10_207_0_v1.1.0.pkl` to `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl` and `SMPL_FEMALE.pkl`. For SMPLX, please download the v1.1 version. Rename The file structure should look like this: