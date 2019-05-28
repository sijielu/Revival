---
layout: default
title:  Status
---

### Project Summary
Our project’s goal is to implement an image-to-image translation, in which a screenshot of the blocks-based world in Minecraft could be converted into an actual photo composed by items and scenes similar to those in the real world. We use CycleGAN, an approach for learning to map an image from the source domain of Minecraft to one from the target domain of actual environments. We would consider the translation to be “successful” if generated pictures are composed of recognizable items and scenes as same as those in input image but with smooth edges and authentic textures. By implementing the translation, Minecraft users are able to enjoy the creating or survival experience in a more realistic scene. It could be applied to other industries as well, for example, blueprint design.

### Approach
In this project, we use CycleGAN, which was introduced in the paper from UC Berekey, "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks". 

#### Model Description
CycleGAN is a Generative Adversarial Network (GAN) that uses two generators and two discriminators. Let's denote that $$X$$ is the domain of Minecraft images and $$Y$$ is the domain of real-world images. The generator $$G$$ converts images from domain $$$$ to domain $$Y$$, and the generator $$F$$ converts images from domain $$Y$$ to domain $$X$$: 

$$G: X \to Y$$

$$F: Y \to X$$

And each generator has a corresponding discriminator which distinguishes real images from fake ones: 

$$D_X: \text{Distinguish real image } x \text{ from fake image } F(y)$$

$$D_Y: \text{Distinguish real image } y \text{ from fake image } G(x)$$


#### Objective Function
* Adversarial Loss  
  Same as the paper, we use a least-squares loss rather than the negative log likelihood objective. For a generator $$G$$ and its corresponding discriminator $$D$$,  
  
  $$\mathcal{L}_{\text{adv}}(G) = \mathbb{E}_{x\sim p_{\text{data}}(x)}[(D(G(x))-1)^2]$$  
  
  $$\mathcal{L}_{\text{adv}}(D) = \mathbb{E}_{y\sim p_{\text{data}}(y)}[(D(y)-1)^2] + \mathbb{E}_{x\sim p_{\text{data}}(x)}[(D(G(x)))^2]$$  
 
* Cycle Consistency Loss  
  We expect that if we convert a image from one domain to another one then convert it back, the reconstructed image should be similar with the original one. The cycle consistency loss is to enforce that $$F(G(x)) \approx x$$ and $$G(F(y)) \approx y$$. The cycle consistency loss is defined using L1 loss:  
  
  $$\mathcal{L}_{\text{cyc}}(G, F) = \mathbb{E}_{x\sim p_{\text{data}}(x)}[\parallel F(G(x))-x \parallel_1] + \mathbb{E}_{y\sim p_{\text{data}}(y)}[\parallel G(F(y))-y \parallel_1]$$


#### Implementation
We use the network architectures described in the original paper. We also follow some parts of the authors' training procedures to train our model.
  
### Evaluation
* Quantitative Evaluation  
  Quantitative evaluation for a Generative Adversarial Network (GAN) is hard to achieve. Here we show the plot of each loss function.

* Qualitive Evaluation
  The aim of qualitive evaluation is to see if the created pictures is akin to that of the real world objects. First we take the screen shot of grassland in Minecraft as input and see the generated image after several epoches:
  
  
  
  
  Here are other examples generated after 39 epoches:
  
  ![test1](https://github.com/sijielu/Revival/blob/master/img/test1.png)
  
  ![test2](https://github.com/sijielu/Revival/blob/master/img/test2.png)

### Remaining Goals and Challenges
1. We are not very satistied with the quality of the generated image. And we are looking for further improvement. Since currently we build the model from scratch, we plan to use the implementation provided by the authors to compare with the result. Another reason of the shortage might be the limitation of the model mentioned in the original paper. We will look for another state-of-the-art model to meet our goals.

2. We are still trying to generate the real-time transformation if time permits.

### References
1. Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros.  [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)

2. PyTorch 1.1: <https://pytorch.org>

3. Urban and Natural Scene Datasets: <http://cvcl.mit.edu/database.htm>
