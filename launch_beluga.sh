#!/usr/bin/env bash
# Walker2d-v2 HalfCheetah-v2 Hopper-v2 Humanoid-v2
# ControllerMF ControllerCEM ControllerSIR ControllerSIS ControllerPLANR
# Walker2d-v2 HalfCheetah-v2
# Hopper-v2 Humanoid-v2

for seed in `seq 0 9`; do
    for env in Walker2d-v2 HalfCheetah-v2 Hopper-v2 Humanoid-v2; do
        for controller in ControllerPLANR ; do
            sbatch launch.sh --env $env
        done
    done
done