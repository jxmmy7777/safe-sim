
# SAFE-SIM

This repository is the official implementation of [SAFE-SIM: Safety-Critical Closed-Loop Traffic Simulation with Diffusion-Controllable Adversaries](https://arxiv.org/abs/2401.00391). 



> **SAFE-SIM: Safety-Critical Closed-Loop Traffic Simulation with Diffusion-Controllable Adversaries**  
> [Wei-Jer Chang](https://scholar.google.com/citations?user=tF-OmYgAAAAJ&hl=en)<sup>1</sup>, [Francesco Pittaluga](https://www.francescopittaluga.com/)<sup>2</sup>, [Masayoshi Tomizuka](https://me.berkeley.edu/people/masayoshi-tomizuka/)<sup>1</sup>, [Wei Zhan](https://zhanwei.site/)<sup>1</sup>, [Manmohan Chandraker](https://cseweb.ucsd.edu/~mkchan)<sup>2,3</sup>  
> <sup>1</sup>University of California Berkeley, <sup>2</sup>NEC Labs America, <sup>3</sup>University of California San Diego

<p>
Evaluating the performance of autonomous vehicle planning algorithms necessitates simulating long-tail safety-critical traffic scenarios. However, traditional methods for generating such scenarios often fall short in terms of controllability and realism; they also neglect the dynamics of agent interactions. To address these limitations, we introduce SAFE-SIM, a novel diffusion-based controllable closed-loop safety-critical simulation framework. Our approach yields two distinct advantages: 1) generating realistic long-tail safety-critical scenarios that closely reflect real-world conditions, and 2) providing controllable adversarial behavior for more comprehensive and interactive evaluations. We develop a novel approach to simulate safety-critical scenarios through an adversarial term in the denoising process of diffusion models, which allows an adversarial agent to challenge a planner with plausible maneuvers while all agents in the scene exhibit reactive and realistic behaviors. Furthermore, we propose novel guidance objectives and a partial diffusion process that enables users to control key aspects of the scenarios, such as the collision type and aggressiveness of the adversarial agent, while maintaining the realism of the behavior. We validate our framework empirically using the nuScenes and nuPlan datasets across multiple planners, demonstrating improvements in both realism and controllability. These findings affirm that diffusion models provide a robust and versatile foundation for safety-critical, interactive traffic simulation, extending their utility across the broader autonomous driving landscape. Project website: https://safe-sim.github.io/.
</p>

## News
- [2024.7.1] Accepted by [ECCV 2024](https://eccv2024.ecva.net/)!

## TODO
- [ ] Release selected scenarios
- [x] Code release 
- [x] Initial repository & preprint release

This repo is mostly built on top of [traffic-behavior-simulation (tbsim)](https://github.com/NVlabs/traffic-behavior-simulation), which handle datasets based on [trajdata](https://github.com/NVlabs/trajdata). The diffusion model are build on top of [MID](https://github.com/Gutianpei/MID), and [diffuser](https://github.com/jannerm/diffuser).

## 1. Environment Setup

Create conda environment 

```python
conda create -n safesim python=3.8
conda activate safesim
```

Install `safesim` (this repo)

```
git clone https://github.com/jxmmy7777/safe-sim.git
cd safesim
pip install -e .
```

Install modified version of `trajdata`

```
cd ..
git clone https://github.com/jxmmy7777/trajdata.git
cd trajdata
pip install -e .

```

 One might Install PyTorch manually from https://pytorch.org/get-started/ to match your system (CUDA/CPU)

## **2. Data Preparation:**

We support the **nuScenes dataset** and utilize [**trajdata**](https://github.com/NVlabs/trajdata) for data processing.

**1️⃣ Download the Dataset**

Follow the [**nuScenes devkit setup guide**](https://github.com/nutonomy/nuscenes-devkit#nuscenes-setup) to download the dataset.

**2️⃣ Organize the Dataset**

Ensure that the dataset is structured as follows:

```
/path/to/nuScenes/
            ├── maps/
            ├── samples/
            ├── sweeps/
            ├── v1.0-mini/
            ├── v1.0-test/
            └── v1.0-trainval/

```

**3️⃣ Preprocess the Data**

Run the following script to **cache dataset and map information** using trajdata:

```python
cd trajdata
python examples/preprocess_data.py 
```

## 3. Quick Start

- Pretrained checkpoints can be downloaded at
    
    `git clone https://huggingface.co/wjchang/safesim_checkpoints`
    
- Update the checkpoint path in `evaluation/Diffusion.yaml`

### Running_adv_simulation

Run the simulation with:

```python
scripts/run_adv_simulation.py \
  --results_root_dir=/path/to/results \
  --num_scenes_per_batch=1 --dataset_path=/path/to/nuscenes \
  --env=nusc --eval_class=StrivePolicy_trajdata --agent_eval_class=Diffusion \
  --ckpt_yaml=evaluation/Diffusion.yaml --guidance \
  --guidance_fn=route_collision_ttc_causecollision \
  --guidance_params=combine_loss,ctrl_weights,\[0.5,0.5,0.5,0.5\]\
  --render --prefix=exp1  --scene_select_mode=default --sim-steps=100 --num_scenes_to_evaluate=100 --skip_first_n=0 
```
### Key Arguments:
| Argument | Description |
| --- | --- |
| `--results_root_dir` | Directory to store simulation results |
| `--num_scenes_per_batch` | Number of scenes processed per batch |
| `--env` | Specifies the dataset environment (e.g., `nusc`) |
| `--guidance_fn` | Specifies the adversarial guidance function |
| `--guidance_params` | Defines hyperparameters for guidance (e.g., control weights) |
| `--eval_class` | The planner to evaluate (`StrivePolicy_trajdata`, `HierAgentAware`,`IDMPolicy` etc.) |
| `--agent_eval_class` | The reactive agent  model (e.g., `Diffusion`) |

Detailed arguments can be found in [simulation_doc](docs/simulation_doc.md).
Overview of the code structure can be found in [code_structure](docs/code_structure.md).

### Visualization

```python
python scripts/visualize.py --output_dir=$OUTPUTPATH --dataset_path=$DATAPATH \
  --env=nusc --hdf5_path=$PATHDOHDF5
```
## Contact
If you have any questions or suggestions, please feel free to open an issue or scontact us (weijer_chang@berkeley.edu).

## Citation
If you find SAFE-SIM useful, please consider giving us a star; and citing our paper with the following BibTeX entry.

## License

This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/.

**Note**: This repository includes components derived from [NVIDIA’s Traffic Behavior Simulation repository](https://github.com/NVlabs/traffic-behavior-simulation), which is licensed under the [NVIDIA Source Code License - NC](https://github.com/NVlabs/traffic-behavior-simulation/blob/main/LICENSE).

Accordingly, any use of this repository must also comply with the terms of the NVIDIA Source Code License - NC, including the non-commercial use restriction.

By using this repository, you agree to follow **both** the NVIDIA Source Code License - NC and the CC BY-NC 4.0 License.
```BibTeX
@inproceedings{chang2024safesimsafetycriticalclosedlooptraffic,
  author    = {Wei-Jer Chang and Francesco Pittaluga and Masayoshi Tomizuka and Wei Zhan and Manmohan Chandraker},
  title     = {SAFE-SIM: Safety-Critical Closed-Loop Traffic Simulation with Diffusion-Controllable Adversaries},
  booktitle = {Computer Vision – ECCV 2024: 18th European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings, Part XXI},
  year      = {2024},
  publisher = {Springer-Verlag},
  address   = {Berlin, Heidelberg},
  pages     = {242--258},
  isbn      = {978-3-031-72663-7},
  doi       = {10.1007/978-3-031-72664-4_14},
  url       = {https://doi.org/10.1007/978-3-031-72664-4_14},
  location  = {Milan, Italy}
}

```
