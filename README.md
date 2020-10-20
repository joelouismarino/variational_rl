# Iterative Amortized Policy Optimization

This code accompanies the paper [Iterative Amortized Policy Optimization](https://openreview.net/pdf?id=49mMdsxkPlD) by Marino et al., 2020.

## Installation & Set-Up

After cloning or downloading the repo, you can install dependencies by running
```
pip install -r requirements.txt
```
from within the main directory. We use [comet](https://www.comet.ml/site/) for plotting and logging. We recommend you set up a (free) account to utilize this functionality. Once you have set up your account, you can enter your comet credentials in `local_vars.py`, replacing `None` with the corresponding entry in each case.

Currently, the code is set up to handle MuJoCo environments from OpenAI gym, which require a [MuJoCo license](https://www.roboti.us/license.html). However, the code can be readily applied to other gym environments with vector observation spaces.

## Training

Specifying a training setup is handled through a combination of command-line arguments to `main.py` and a configuration file located in `config/mujoco_config.py`. Experiment-level settings are specified in `main.py`, and model-level settings are specified in `mujoco_config.py`.

### mujoco_config.py

The `mujoco_config.py` contains configuration settings for distributions, network architectures, policy optimizers, and value estimators. The paper focuses on evaluating different policy optimizers, which are specified on line 78 in `mujoco_config.py`. To run the baseline soft actor-critic setup (direct amortization), set `optimizer_type = 'direct'`. For iterative amortization, set `optimizer_type = 'iterative'`.

We found that iterative amortization, due to its more flexible form, tended to exploit the value estimator. For this reason, we found it necessary to penalize high-variance value estimates using a "pessimism" parameter described in the paper. This parameter is set on line 17. For iterative amortization, we recommend `'pessimism': 2.5`.

### main.py


To run an experiment, call `main.py` with the appropriate command-line arguments. For instance, to run an experiment on the `HalfCheetah-v2` environment using GPU 0, run the following:
```
python main.py --env HalfCheetah-v2 --device_id 0
```
*Note*: not specifying `device_id` will run the experiment entirely on the CPU by default.

To disable plotting (not recommended), you can use the command-line argument `--plotting False`.

## Contact

If you have any questions, send me an email at `jmarino` [at] `caltech` [dot] `edu`, or post an issue on Github.
