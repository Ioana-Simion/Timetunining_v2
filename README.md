# Timetuning_v2

# 20_07_2023

According to what we have discussed so far : 

1 - Implementing hummingbird

2 - Getting the accuracy of motion trajectory prediction on a synthetic dataset

3 - What would be the performance of DINO when it is faced with overlapping objects

4 - Implementing the method with soft attention approach.

![Logo](Sessions/20_07_2023.jpg)

# 26_07_2023

According to what we have discussed so far : 

1 - Testing slot attention instead of prototypes

2 - Initializing the instace segmentor by training on the synthetic datasets

3 - Testing DINO + Positional + SA + GT to if it can learn instances

4 - Sliding object experiment

![Logo](Sessions/26_07_2023.jpg)

# 07_-8_2023

According to what we have discussed so far, we can start working on further improving the performance of TimeT by the following revisements:
1 - Involving coarse grained/fine grained branch to predicit the location of a tube sampled of each clip at each frame.
2 - Using CoTracker to track the location of each point in the video and clustering the object flows to detects displacements. 
3 - Training TimeT from scratch

Experiments to do : 
1 - Are the patches aware of their origins in DINO? 
2 - How can we use cotracker instead of flow and train from scratch in TimeT?

![Logo](Sessions/07_08_2023.jpg)

# Questions

1 - It is not obvious if the background is being excluded from the computation or not. I should check it out. 