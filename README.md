# Neural-network---make-moons-with-1000-samples
This script implements a small fully-connected neural network from scratch using NumPy to classify the make_moons toy dataset. It trains the network with mini-batch SGD and demonstrates/compares two hidden-layer activation choices (sigmoid vs ReLU). The script also visualizes the decision boundary and plots loss/accuracy per epoch.

What the code does (quick tour)

Data generation

Creates a 2D binary classification dataset with sklearn.datasets.make_moons (1000 samples, noise=0.2).

Labels y are reshaped to column vectors for easier matrix math.

(A commented-out normalization line is present if you want to scale inputs.)

Activation functions

Implements sigmoid and relu activations plus their derivatives:

sigmoid(z) and sigmoid_derivative(z)

relu(z) and relu_derivative(z)

Loss

Binary cross-entropy implemented as binary_cross_entropy(y_true, y_pred) with a small epsilon (1e-8) for numerical stability.

Network architecture & initialization

Input dimension: 2

Hidden layer 1: 10 neurons

Hidden layer 2: 5 neurons

Output: 1 neuron with sigmoid (probability of class 1)

Weights W1, W2, W3 initialized from a normal distribution; biases initialized to zeros.

Global hyperparameters: learning_rate = 0.005, epochs = 1001, batch_size = 32, np.random.seed(42).

Forward pass

forward(...) computes activations through the two hidden layers (using either sigmoid or relu depending on activation) and the final sigmoid output.

Returns the predicted probabilities and a cache of intermediate values needed for backpropagation.

Backward pass (backpropagation)

backward(...) computes gradients of the loss w.r.t. weights and biases using cached forward values.

Supports both sigmoid and relu hidden activations when computing gradient flow.

Parameter updates

update_parameters(...) applies basic gradient descent updates: param -= learning_rate * grad.

Training loop (mini-batch SGD)

train_sgd(...) shuffles data every epoch, iterates mini-batches, runs forward/backward, updates parameters.

After each epoch it computes and records full-dataset loss and accuracy.

Prints progress every 100 epochs.

Evaluation & visualization

accuracy(...) computes classification accuracy using a 0.5 threshold on output probabilities.

plot_decision_boundary(...) builds a grid, predicts over the grid and shows the decision boundary with the dataset points.

Several plotting helpers show loss per epoch and accuracy per epoch for each activation type. The script trains and plots first for sigmoid, then for ReLU.
