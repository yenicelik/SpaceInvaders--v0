# 

# SpaceInvaders--v0
Playing around with OpenAI gym using the [SpaceInvader Environment](https://gym.openai.com/envs/SpaceInvaders-v0). Just experimenting, no serious competition intended.

## Usage
Run
```
python Spaceinvaders.py
```
for simple use. You can change the parameters within the `setup.py`-parameters dictionary.

## Code Files
#### SpaceInvaders.py
Includes the 'main thread' of the program. Very simple abstraction level.

#### build_graph.py
Builds the tensorflow graph, as suggested by [this famous paper by Mnih, Silver, Kavukcuooglu, Graves, Antanoglou, Wierstra and Riedmiller (DeepMind)](https://arxiv.org/abs/1312.5602)
Not all details are implemented yet, such as using the last 4 images as input.

#### helper.py
Includes helper functions for saving the model features, saving figures in matlab plots, and preprocess images (to apply efficient squared convolution)

#### setup.py
Creates environment and sets up parameters to be used during training

#### train.py
Implementation of a basic DeepQ-Learning algorithm. This is the heart of the program.


-------------

Feel free to contact me for suggestions on the style, implementation etc.
