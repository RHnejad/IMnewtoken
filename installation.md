# Setup for PHC with virtualenv
Clone PHC repo
```
git clone https://github.com/ZhengyiLuo/PHC.git
```

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
```
|-- data
    |-- smpl
        |-- SMPL_FEMALE.pkl
        |-- SMPL_NEUTRAL.pkl
        |-- SMPL_MALE.pkl
        |-- SMPLX_FEMALE.pkl
        |-- SMPLX_NEUTRAL.pkl
        |-- SMPLX_MALE.pkl
```

# Setup for PHC with Docker (alternative)
### Clone PHC repo
```
git clone https://github.com/ZhengyiLuo/PHC.git
```
### Donwload IsaacGym
Download [Isaac Gym](https://developer.nvidia.com/isaac-gym)
```
mkdir -p third_party
cd third_party
tar -xzf ~/Downloads/IsaacGym_Preview_4_Package.tar.gz
```
### Download SMPL and SMPL-X parameters
Download SMPL paramters from [SMPL](https://smpl.is.tue.mpg.de/) and [SMPLX](https://smpl-x.is.tue.mpg.de/download.php). Put them in the `data/smpl` folder, unzip them into 'data/smpl' folder. For SMPL, please download the v1.1.0 version, which contains the neutral humanoid. Rename the files `basicmodel_neutral_lbs_10_207_0_v1.1.0`, `basicmodel_m_lbs_10_207_0_v1.1.0.pkl`, `basicmodel_f_lbs_10_207_0_v1.1.0.pkl` to `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl` and `SMPL_FEMALE.pkl`. For SMPLX, please download the v1.1 version. Rename The file structure should look like this:
```
|-- data
    |-- smpl
        |-- SMPL_FEMALE.pkl
        |-- SMPL_NEUTRAL.pkl
        |-- SMPL_MALE.pkl
        |-- SMPLX_FEMALE.pkl
        |-- SMPLX_NEUTRAL.pkl
        |-- SMPLX_MALE.pkl
```
### Build
```
cd docker
./run.sh build
```
### Interactive run
```
cd docker
./run.sh # With GPU support
./run.sh --no-gpu # No GPU support (Won't work as IsaacGym requires it)
```