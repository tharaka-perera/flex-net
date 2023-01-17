# Flex-Net: A Graph Neural Network Approach to Resource Management in Flexible Duplex Networks

This repository is the code release corresponding to our paper which introduces a novel graph neural network based approach to maximize the sum rate of flexible duplex networks.
Main contributions of our paper are as follows.

1. We formulate a novel graph structure that can represent the flexible duplex network. This graph can represent desired links and potential interference links including the direction to efficiently learn the geometric and numerical features of the flexible duplex network.
2. We propose a novel GNN model called **Flex-Net** with an unsupervised-learning strategy to jointly optimize communication direction and transmit power to maximize the sum-rate of the flexible duplex network.
3. We compare numerical results obtained by extensive simulations using the proposed GNN with baselines listed in Table below. We show that the proposed method outperforms baselines in terms of performance and time complexity. Furthermore, we analyze the sample complexity, scalability, and generalization capability of the proposed approach.

<center>

| Approach                    | Time Complexity         | Performance (avg.) |
|-----------------------------|-------------------------|--------------------|
| Exhaustive Search           | $\mathcal{O}(2^n)$      | 100\%              |
| **Flex-Net**                | $\bm{\mathcal{O}(n^2)}$ | **95.8\%**         |
| Heuristic Search            | $\mathcal{O}(n^4)$      | 95.5\%             |
| Max Power                   | $\mathcal{O}(n)$        | 49.9\%             |
| Max Power with Silent Nodes | $\mathcal{O}(n^2)$      | 67.5\%             |

</center>

## Setup

- Make sure Python 3.7 or greater version is installed.
- Install Pytorch following the instructions from [pytorch.org](https://pytorch.org/)
- Install Pytorch Geometric following the [PyG Documentation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
- Install requirements.txt (```pip install -r requirements.txt```)

## Run Code

- A model can be trained using the `create_and_train_model`  function in `main.py`.
- Trained models can be evaluated using the `eval_model` function in `main.py`.
- More experiments and plots are available in the Jupyter notebooks (Trained models are available in `experiemnts` folder)

## Cite

Please cite [our paper]() if you use this code in your own work:

```
@inproceedings{
Pere2303:Flex,
title="{Flex-Net:} A Graph Neural Network Approach to Resource Management in Flexible Duplex Networks",
author="Tharaka Perera and Saman Atapattu and Yuting Fang and Prathapasinghe Dharmawansa and Jamie S Evans",
booktitle="2023 IEEE Wireless Communications and Networking Conference (WCNC) (IEEE WCNC 2023)",
year={2023}
}
```