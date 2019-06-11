---
layout: default
title:  Final Report
---

### Video

### Project Summary

### Approaches
#### 1. Introduction
- **Image-to-Image Translation**  
  Image-to-image translation is the task of taking images from one domain and transforming them so they have the style (or characteristics) of images from another domain. 
- **Generative Adversarial Networks (GANs)**  
  Generative Adversarial Networks (GANs) achieved huge successes in image editing, image generation, and many other areas. GANs, introduced by Ian Goodfellow and other researchers at the Université de Montréal in 2014, are a type of neural networks based on the idea of adversarial training, where two neural networks, generator and discriminator, contest with each other. 
- **Cycle-Consistent Adversarial Networks (CycleGANs)**

#### 2. Formulation
- **Adversarial Loss**  
  For the mapping function $$G: X \to Y$$ and its corresponding discriminator $$D_Y$$, the adversarial loss is defined as  
  
  $$\mathcal{L}_{\text{GAN}}(G, D_Y, X, Y) = \mathbb{E}_{y\sim p_{\text{data}}(y)}[\log D_Y(y)] + \mathbb{E}_{x\sim p_{\text{data}}(x)}[\log(1-D_Y(G(x)))].$$  
  
  Similarly, for the mapping function $$F: Y \to X$$ and its corresponding discriminator $$D_X$$, the adversarial loss is defined as  
  
  $$\mathcal{L}_{\text{GAN}}(F, D_X, Y, X) = \mathbb{E}_{x\sim p_{\text{data}}(x)}[\log D_X(x)] + \mathbb{E}_{y\sim p_{\text{data}}(y)}[\log(1-D_X(F(y)))].$$  
  
  The generative network $$G$$ aims to minimize the adversarial loss, while the discriminative network $$D$$ strives to maximize this objective, i.e., $$\text{min}_G\text{max}_{D_Y} \,\mathcal{L}_{\text{GAN}}(G, D_Y, X, Y)$$ and $$\text{min}_F\text{max}_{D_X} \,\mathcal{L}_{\text{GAN}}(F, D_X, Y, X)$$.  
  
  Following the orginal paper, the negative log likelihood objective is replaced by a least-squares loss. For the modified adversarial loss $$\mathcal{L}_{\text{GAN}}(G, D, X, Y)$$, the mapping function $$G$$ tries to minimize $$\mathbb{E}_{x\sim p_{\text{data}}(x)}[(D(G(x))-1)^2]$$ during training, while the corresponding discriminator $$D$$ minimizes $$\mathbb{E}_{y\sim p_{\text{data}}(y)}[(D(y)-1)^2] + \mathbb{E}_{x\sim p_{\text{data}}(x)}[(D(G(x)))^2]$$.
  
- **Cycle Consistency Loss**
- **Identity Mapping Loss**
- **Full Objective**

#### 3. Network Architectures
- **Generator Architectures**
- **Discriminator Architectures**

#### 4. Training Details
- **Data Collection**
- **Hyperparameter Tuning**


### Evaluation
#### 1. Overview

#### 2. Quantitative Evaluation

#### 3. Qualitative Evaluation

### References
1. I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf). In NIPS, 2014.

2. Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf), in IEEE International Conference on Computer Vision (ICCV), 2017. 
