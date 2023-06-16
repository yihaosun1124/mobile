# Model-Bellman Inconsistency Penalized Offline Policy Optimization (MOBILE)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/yihaosun1124/mobile/blob/main/LICENSE)

Code for MOBILE: [Model-Bellman Inconsistency Penalized Offline Policy Optimization](https://openreview.net/forum?id=rwLwGPdzDD).

## Requirements

To install all the required dependencies:

1. Install MuJoCo engine, which can be downloaded from [here](https://mujoco.org/download).
2. Install Python packages listed in `requirements.txt` using `pip install -r requirements.txt`. You should specify the version of `mujoco-py` in `requirements.txt` depending on the version of MuJoCo engine you have installed.
3. Manually download and install `d4rl` package from [here](https://github.com/rail-berkeley/d4rl).
4. Manually download and install `neorl` package from [here](https://github.com/polixir/NeoRL).

## Usage

Just run `train.py` with specifying the task name. Other hyperparameters are automatically loaded from `config`.

```bash
python train.py --task [TASKNAME]
```

## Citation

If you find this repository useful for your research, please cite:

```bash
@inproceedings{
    mobile,
    title={Model-Bellman Inconsistency Penalized Offline Policy Optimization},
    author={Yihao Sun and Jiaji Zhang and Chengxing Jia and Haoxin Lin and Junyin Ye and Yang Yu},
    booktitle={International Conference on Machine Learning},
    year={2023}
}
```
