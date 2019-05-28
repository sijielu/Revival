---
layout: default
title:  Status
---

### Project Summary

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
  Same as the paper, we use a least-squares loss rather than the negative log likelihood objective.

### Evaluation

### Remaining Goals and Challenges

### References
