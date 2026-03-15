# 🚀 Advanced GenAI Infrastructure & HPC
**Enterprise-grade GPU Optimization, Distributed Deep Learning, and Inference Orchestration**

[![Author](https://img.shields.io/badge/Senior_Manager-NVIDIA-76B900?style=for-the-badge&logo=nvidia)](https://www.linkedin.com/in/harry-bai-1b907b292/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch_2.2-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Optimization](https://img.shields.io/badge/Optimization-TensorRT--LLM-76B900?style=for-the-badge&logo=nvidia)](https://github.com/NVIDIA/TensorRT-LLM)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

## 🌟 Executive Summary
The **Advanced GenAI Infrastructure & HPC** framework is a flagship repository designed for architecting and optimizing Large Language Model (LLM) lifecycles at enterprise scale. Focused on the **NVIDIA H200/H100 ecosystem**, this project provides production-ready implementations for distributed training (DeepSpeed), inference acceleration (TensorRT-LLM), and scalable GPU orchestration.

## 🏗️ Core Pillars
### 1. High-Performance Inference (TensorRT-LLM)
Modular pipeline for converting HuggingFace checkpoints into highly optimized TensorRT engines. Features:
- FP8/INT8 Quantization (AWQ/SmoothQuant).
- KV Cache optimization for high-throughput serving.
- Integration with Triton Inference Server.

### 2. Distributed Training & Fine-Tuning
Robust scripts for multi-node training utilizing **DeepSpeed ZeRO-3** and **Flash Attention 2**. 
- Automated checkpointing and mixed-precision (BF16) stability.
- Slurm-ready orchestration templates.

### 3. GPU Orchestration & MLOps
Infrastructure-as-Code (IaC) for GPU clusters.
- Kubernetes NVIDIA Device Plugin configurations.
- Prometheus/Grafana monitoring dashboards for DCGM (Data Center GPU Manager).

## 📂 Repository Topology
```text
├── src/
│   ├── optimization/         # TensorRT-LLM conversion and quantization
│   ├── training/             # Distributed PyTorch + DeepSpeed scripts
│   └── serving/              # Triton Inference Server configuration
├── infrastructure/           # Docker, K8s, and GPU driver setup
├── scripts/                  # HPC benchmarking and MLPerf tools
├── tests/                    # Unit & integration tests for ML logic
├── Makefile                  # Standardized MLOps commands
└── Dockerfile                # High-performance CUDA development environment
```

## 🚀 Quick Start
```bash
# 1. Setup Environment
make install

# 2. Build NVIDIA-Optimized Container
make docker-build

# 3. Compile Model to TensorRT-LLM
python src/optimization/build_engine.py --model llama-2-70b --precision fp8
```

---
**Architected by [Harry Bai](https://github.com/HarryBai4)**  
*Senior Manager @ NVIDIA | 24+ YOE in GPU & AI Infrastructure*
