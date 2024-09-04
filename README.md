## Project Overview

The core idea behind this project is to represent an image as a set of coordinate points. AKA Implicit Neural Representation.
Each point (x,y) corresponds to a pixel in the image. The model is trained to predict the RGB values of these pixels based on their coordinates. 
This is a fundamental task that can be extended to various computer vision and graphics applications. Known widely as Implicit Neural representations, or INRs, and that is use in multiple areas for ex. NERFs.
Inspired by the Kolmogorov Arnold networks that yield interpretable functions of mappings between inputs and outputs,
I decided to make my own implementation of function approximations for inputs and outputs but where one optimizes the functions directly.

The neural network tunes a set of parameters for each function, a,b,c,d where we have lambda x: c*f(a*x +b)+d
You can choose multiple f() functions, such as sin or cos, it's plug in play, try to stick to pytorch compatible functions though. 

The Neural Net is searching the parameters, but also for the correct function at the same time.
I modulo a grid of values, to keep them within a range of len(activations)-1, and after rounding a number in the grid,
apply the function that corresponds to the index.

The implementation is quite fast and runs on GPUs. 

### How train.py works:

- Loads any RGB image and resize it to a 256x256 grid.
- Convert the image into a set of coordinates and corresponding RGB values.
- Train a neural network model to learn the mapping between coordinates and RGB values.

python train.py /path/to/your/image.png --epochs 50000 --learning_rate 0.01 --hidden_size 128

#TODO:
-Yield the full formula for input to outputs
-Get fit of images closer to higher PSNR or lower MSE 
