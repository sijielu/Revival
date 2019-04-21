---
layout: default
title:  Proposal
---

### **Summary of the Project**
Our project aims at converting screenshots of the view of character in Minecraft into the pictures in real world. The screenshot is based on the random game scene that has been settled. The input is the screenshot of the scene in Minecraft, a world which everything is made of cubes. And what we need to do is to use deep learning techniques to transform the screenshot to a picture composed by items and scenes similar to those in real world. In the generated picture, we smooth the vertically unreal frames of items and match the items in Minecraft with its real world symbolization. By doing this we can make game player enjoy playing game in a more realistic scene that can bring a whole new game experience.

### **AI/ML Algorithms**
For our project, we plan to use generative adversarial network (GAN) to build our machine learning system.

### **Evaluation Plan**
To determine the success of our project, we will evaluate our machine learning system both quantitatively and qualitatively:

* **Quantitative Evaluation**  
The major metric of our system is the percentage of generated pictures being considered as "real-world pictures". To acquire the percentage, we will conduct a perceptual study that participants help us label generated pictures as "real" or "fake" based on their judgment. In addition, we are also seeking an automatically quantitative measurement that does not require human participation. We found that there are many amazing landmarks of the world recreated in Minecraft. Thus, one possible idea could be that we take these replicas as input, generate the "fake" pictures, and compare them aginst real-world photos using per-pixel accuracy or other measurements. 

* **Qualitative Evaluation**  
Our qualitative evaluation for our system is based on the quality and consistency of the generated photos. Basic sanity cases would be identifying simple environments with a small number of objects in Minecraft and generating a real-world picture for the environment. In addition, the photo should be recognizable and more than 50% of it should be considered as real. For a moonshot case, we would like to be able to identify 70%-80% objects in the Minecraft world and restore them in the photo with a consistent relative position. If we cannot achieve the goals, we would figure out which step of the process needs to be improved. For example, distinguish objects from the background, identify what the objects are, make the real picture accurate. These process can also be visualized.

### **Appointment with the Instructor**
9:30am on Monday, April 29 (DBH 4204)
