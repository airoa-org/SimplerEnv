# **Installation Guide: SimplerEnv + ManiSkill2_real2sim + geniac25_team5_codebase**

This document provides step-by-step instructions to set up the environment and install all required dependencies.

## 1. Create and Activate Conda Environment
```bash
conda create -n simpler_env python=3.10 && conda activate simpler_env
```

## 2. Clone the Repository

```bash
git clone https://github.com/simpler-env/SimplerEnv --recurse-submodules --depth 1
```

## 3. Install Core Dependencies

```bash
pip install numpy==1.24.4
```

## 4. Install ManiSkill2\_real2sim

```bash
cd {this_repo}/ManiSkill2_real2sim
pip install -e .
```

## 5. Install SimplerEnv

```bash
cd {this_repo}
pip install -e .
```

## 6. Install geniac25\_team5\_codebase

```bash
cd {this_repo}/geniac25_team5_codebase
pip install -e .
```

## 7. Install Additional Packages

```bash
pip install rerun-sdk==0.23.1 numpy==1.24.4
pip install bitsandbytes transformers==4.48.1 pytest
```

## 8. Reinstall draccus with Specific Version

```bash
pip uninstall draccus
pip install draccus==0.10.0
```

## 9.

```bash
conda install -c conda-forge ffmpeg
```