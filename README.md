# Cyber Attack Detection in Connected Autonomous Vehicles using [PINN](https://maziarraissi.github.io/PINNs/)

> **Notice:** This is a fork of "https://maziarraissi.github.io/PINNs/". You are on the detection branch.

We introduce cyber attack detection algorithm in Connected Autonomous Vehicles (CAVs) based on physics informed neural networks. We are referencing the code provided in "/appendix/continuous_time_identification (Burgers)/Burgers.py" as a baseline. The folder "data" contains the data under: no attack scenario, communication attack, and sensor attack. The file "detect.py" contains the code for detecting cyber attacks in the communication channels and local vehicular sensors of a platoon of vehicles operating in Cooperative Adaptive Cruise Control (CACC).

> **Notice:** Information about the original repository is below.

# [Physics Informed Neural Networks](https://maziarraissi.github.io/PINNs/)

> **Notice:** This repository is no longer under active maintenance. It is highly recommended to utilize implementations of Physics-Informed Neural Networks (PINNs) available in [PyTorch](https://github.com/rezaakb/pinns-torch), [JAX](https://github.com/rezaakb/pinns-jax), and [TensorFlow v2](https://github.com/rezaakb/pinns-tf2).

We introduce physics informed neural networks – neural networks that are trained to solve supervised learning tasks while respecting any given law of physics described by general nonlinear partial differential equations. We present our developments in the context of solving two main classes of problems: data-driven solution and data-driven discovery of partial differential equations. Depending on the nature and arrangement of the available data, we devise two distinct classes of algorithms, namely continuous time and discrete time models. The resulting neural networks form a new class of data-efficient universal function approximators that naturally encode any underlying physical laws as prior information. In the first part, we demonstrate how these networks can be used to infer solutions to partial differential equations, and obtain physics-informed surrogate models that are fully differentiable with respect to all input coordinates and free parameters. In the second part, we focus on the problem of data-driven discovery of partial differential equations.

For more information, please refer to the following: (https://maziarraissi.github.io/PINNs/)

  - Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis. "[Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations](https://www.sciencedirect.com/science/article/pii/S0021999118307125)." Journal of Computational Physics 378 (2019): 686-707.

  - Raissi, Maziar, Paris Perdikaris, and George Em Karniadakis. "[Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations](https://arxiv.org/abs/1711.10561)." arXiv preprint arXiv:1711.10561 (2017).

  - Raissi, Maziar, Paris Perdikaris, and George Em Karniadakis. "[Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations](https://arxiv.org/abs/1711.10566)." arXiv preprint arXiv:1711.10566 (2017).

## Citation

    @article{raissi2019physics,
      title={Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
      author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George E},
      journal={Journal of Computational Physics},
      volume={378},
      pages={686--707},
      year={2019},
      publisher={Elsevier}
    }

    @article{raissi2017physicsI,
      title={Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations},
      author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
      journal={arXiv preprint arXiv:1711.10561},
      year={2017}
    }

    @article{raissi2017physicsII,
      title={Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations},
      author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
      journal={arXiv preprint arXiv:1711.10566},
      year={2017}
    }
