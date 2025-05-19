# Genetic Programming for Convolutional Neural Network Architecture

This project uses genetic programming to evolve simple convolutional neural network architectures for image classification tasks. It utilizes the DEAP library for the genetic programming framework and TensorFlow/Keras for building and evaluating the neural networks.

## Datasets

The code loads and preprocesses three image datasets:

- **CIFAR-10:** A dataset of 32x32 color images across 10 classes.
- **EMNIST (Letters):** A dataset of 28x28 grayscale images of letters (excluding I and J) across 26 classes.
- **Fashion MNIST:** A dataset of 28x28 grayscale images of clothing items across 10 classes.

Smaller subsets of these datasets are created for faster experimentation during the genetic programming process.

## Genetic Programming Setup

- **Primitives:** The genetic programming uses a set of primitives including:
    - `sigmoid`: Applies a sigmoid function based on the `x` attribute of an `ObjectForTree` instance.
    - `tanh`: Applies a tanh function based on the `y` attribute of an `ObjectForTree` instance.
    - `add_pool_layer`: Adds a `MaxPool2D` layer to the Keras model within an `ObjectForTree` instance. The pooling size depends on the `y` attribute.
    - `add_conv_layer`: Adds a `Conv2D` layer to the Keras model within an `ObjectForTree` instance. The kernel size and activation function depend on the `x` attribute.
- **Individual Representation:** Individuals are represented as genetic programming trees composed of the defined primitives.
- **Fitness Function:** The fitness of an individual is evaluated by building a Keras model based on the tree structure, training it on a subset of a dataset, and measuring its accuracy on a corresponding test subset. The goal is to maximize accuracy.
- **Genetic Operators:**
    - **Selection:** Tournament selection.
    - **Crossover:** One-point crossover.
    - **Mutation:** Uniform mutation.

## How to Run

1.  **Install Dependencies:**
2.  **Run the Notebook:** Execute the cells in the Colab environment.
3.  **Select Dataset:** Run the "Registering toolbox for <Dataset_Name> dataset" block for the dataset you want to use before running the `main()` function. Only one dataset should be active at a time.
4.  **Execute `main()`:** Run the `main()` function to start the genetic programming process.

## Output

- The script will print the progress of the genetic programming algorithm.
- A file named `tree.pdf` will be generated showing an example initial tree structure.
- A file named `best_tree.pdf` will be generated showing the structure of the best evolved model.
- The final output will include the best evolved function (tree structure).

## Customization

- You can adjust the genetic programming parameters (population size, number of generations, crossover probability, mutation probability) in the `main()` function.
- You can modify the `evaluate_model` function to change the training parameters (batch size, epochs) or the architecture of the fully connected layers.
- You can add new primitives to the `pset` to explore different types of layers or operations.
