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
  We expect that if we transform a image from one domain to another one and then convert it back, the reconstructed image should be similar with the original one. Therefore, the two mapping functions $$G$$ and $$F$$ should be cycle-consistent. For every image $$x$$ from the domain $$X$$, the cycle consistency is to enforce that $$x \to G(x) \to F(G(x)) \approx x$$. Similarly, for each image $$y$$ from the domain $$Y$$, the image translation cycle should be able to reconstruct the orginal image, i.e., $$y \to F(y) \to G(F(y)) \approx y$$. The cycle consistency loss is defined using L1 loss in the orginal paper, which is
  
  $$\mathcal{L}{\text{cyc}}(G, F) = \mathbb{E}_{x\sim p_{\text{data}}(x)}[\parallel F(G(x))-x \parallel_1] + \mathbb{E}_{y\sim p{\text{data}}(y)}[\parallel G(F(y))-y \parallel_1].$$
  
  In preliminary experiments, we also tried to define the cycle consistency loss using L2 loss. However, we did not see any improvement.
  
- **Identity Mapping Loss**  

- **Full Objective**  
  Combining the loss functons defined above, we obtain the full objective for our model:
  
  $$\mathcal{L}(G, F, D_X, D_Y) = \mathcal{L}_{\text{GAN}}(G, D_Y, X, Y) + \mathcal{L}_{\text{GAN}}(G, D, X, Y) + \lambda\mathcal{L}{\text{cyc}}(G, F),$$
  
  where $$\lambda$$ is the relative weight of the cycle consistency loss. Hence, we are aiming to solve  
  
  $$G^*, F^* = \text{arg}\, \underset{G, F}{\text{min}}\, \underset{D_X, D_Y}{\text{max}}\, \mathcal{L}(G, F, D_X, D_Y).$$

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

2. J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros. [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf). In ICCV, 2017. 
