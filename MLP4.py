import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# Δημιουργία του dataset
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
y = y.reshape(-1, 1)
#X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))  #Κανονικοποίηση 


# Συναρτήσεις ενεργοποίησης
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z): #Παράγωγος
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z): #Παράγωγος
    return (z > 0).astype(float)

# Loss Function
def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))

# Αρχικοποίηση παραμέτρων
np.random.seed(42)
input_dim = 2 #Είσοδοι
hidden_dim1 = 10 #1ή κρυφή στρώση με νευρώνες
hidden_dim2 = 5 #2ή κρυφή στρώση με νευρώνες
output_dim = 1   #Έξοδος
learning_rate = 0.005
epochs = 1001
batch_size = 32 # Μέγεθος batch για SGD

W1 = np.random.randn(input_dim, hidden_dim1)
b1 = np.zeros((1, hidden_dim1))
W2 = np.random.randn(hidden_dim1, hidden_dim2)
b2 = np.zeros((1, hidden_dim2))
W3 = np.random.randn(hidden_dim2, output_dim)
b3 = np.zeros((1, output_dim))

# Εμπρός Περασμα (Forward Pass)
def forward(X, W1, b1, W2, b2, W3, b3, activation='sigmoid'):
    if activation == 'sigmoid':
        Z1 = X.dot(W1) + b1
        A1 = sigmoid(Z1)
        Z2 = A1.dot(W2) + b2
        A2 = sigmoid(Z2)
    elif activation == 'relu':
        Z1 = X.dot(W1) + b1
        A1 = relu(Z1)
        Z2 = A1.dot(W2) + b2
        A2 = relu(Z2)

    Z3 = A2.dot(W3) + b3
    A3 = sigmoid(Z3)

    cache = {'A1': A1, 'A2': A2, 'A3': A3, 'Z1': Z1, 'Z2': Z2, 'Z3': Z3}
    return A3, cache

# Οπισθόδρομη Περασμα (Backpropagation)
def backward(X_batch, batch, cache, W1, W2, W3, activation='sigmoid'):
    A1, A2, A3 = cache['A1'], cache['A2'], cache['A3']
    Z1, Z2, Z3 = cache['Z1'], cache['Z2'], cache['Z3']
    m = batch.shape[0] # Μέγεθος του batch

    dZ3 = A3 - batch #Υπολογισμός σφάλματος εξόδου
    dW3 = A2.T.dot(dZ3) / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m
# Μεταφορά σφάλματος προς τα πίσω
    if activation == 'sigmoid':
        dA2 = dZ3.dot(W3.T) 
        dZ2 = dA2 * sigmoid_derivative(Z2)
    elif activation == 'relu':
        dA2 = dZ3.dot(W3.T)
        dZ2 = dA2 * relu_derivative(Z2)

    dW2 = A1.T.dot(dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    if activation == 'sigmoid':
        dA1 = dZ2.dot(W2.T)
        dZ1 = dA1 * sigmoid_derivative(Z1)
    elif activation == 'relu':
        dA1 = dZ2.dot(W2.T)
        dZ1 = dA1 * relu_derivative(Z1)

    dW1 = X_batch.T.dot(dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    gradients = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3}
    return gradients

# Ενημέρωση Βαρών
def update_parameters(W1, b1, W2, b2, W3, b3, gradients, learning_rate=0.005):
    W1 -= learning_rate * gradients['dW1']
    b1 -= learning_rate * gradients['db1']
    W2 -= learning_rate * gradients['dW2']
    b2 -= learning_rate * gradients['db2']
    W3 -= learning_rate * gradients['dW3']
    b3 -= learning_rate * gradients['db3']
    return W1, b1, W2, b2, W3, b3

#Ακρίβεια
def accuracy(y_true, y_pred):
    predictions = (y_pred > 0.5).astype(int) #Θεωρεί πρόβλεψη το 1 αν η πιθανότητα είναι>0.5 , αλλιώς 0
    return np.mean(predictions == y_true) #Average


# Εκπαίδευση με SGD
def train_sgd(X, y, epochs=1001, batch_size=32, learning_rate=0.005, activation='sigmoid'):
    global W1, b1, W2, b2, W3, b3
    losses = []
    accuracies = []
    num_samples = X.shape[0]
    for epoch in range(epochs):
        # Ανακάτεμα των δεδομένων σε κάθε epoch
        permutation = np.random.permutation(num_samples)
        X_shuffled = X[permutation] 
        y_shuffled = y[permutation]

        for i in range(0, num_samples, batch_size):
            # Επιλογή του mini-batch
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            # Εμπρός πέρασμα
            A3, cache = forward(X_batch, W1, b1, W2, b2, W3, b3, activation)

            # Υπολογισμός του loss
            loss = binary_cross_entropy(y_batch, A3)

            # Οπισθόδρομο πέρασμα
            gradients = backward(X_batch, y_batch, cache, W1, W2, W3, activation)

            # Ενημέρωση των παραμέτρων
            W1, b1, W2, b2, W3, b3 = update_parameters(W1, b1, W2, b2, W3, b3, gradients, learning_rate)

        # Υπολογισμός του loss και της ακρίβειας για όλο το dataset στο τέλος κάθε epoch
        A3_full, _ = forward(X, W1, b1, W2, b2, W3, b3, activation)
        loss_full = binary_cross_entropy(y, A3_full)
        acc_full = accuracy(y, A3_full)
        losses.append(loss_full)
        accuracies.append(acc_full)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss_full}, Accuracy: {acc_full * 100:.2f}%")
    return losses,accuracies

# Συνάρτηση για οπτικοποίηση των συνόρων απόφασης
def plot_decision_boundary(X, y, W1, b1, W2, b2, W3, b3, activation='sigmoid'):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds, _ = forward(grid, W1, b1, W2, b2, W3, b3, activation)
    preds = preds.reshape(xx.shape)

    
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolors='k', cmap=plt.cm.coolwarm_r) #Χρώμα
    plt.title(f"Decision Boundary with {activation} activation (SGD)")
    plt.show()

def plot_epoch_sigmoid(X, y, W1, b1, W2, b2, W3, b3, activation='sigmoid'):
    plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), losses_sgd_sigmoid, label='Sigmoid Activation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch (Sigmoid Activation)')
    plt.legend()
    plt.grid(True)
    plt.show()
     
   
def plot_epoch_relu(X, y, W1, b1, W2, b2, W3, b3, activation='Relu'):
    plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), losses_sgd_relu, label='Relu Activation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch (Relu Activation)')
    plt.legend()
    plt.grid(True)
    plt.show()  
     
     
def plot_accuracy_per_epoch(X, y, epochs, activation='sigmoid'):
    plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), acc_sgd_sigmoid , label=f'{activation} activation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy in %')
    plt.title(f'Accuracy per Epoch ({activation} activation)')
    plt.legend()
    plt.grid(True)
    plt.show()
     
     
def plot_accuracy_per_epoch_relu(X, y, epochs, activation='relu'):
    plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), acc_sgd_relu, label=f'{activation} activation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy per Epoch ({activation} activation)')
    plt.legend()
    plt.grid(True)
    plt.show()     
  
     
#Sigmoid
losses_sgd_sigmoid, acc_sgd_sigmoid = train_sgd(X, y, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, activation='sigmoid')
plot_decision_boundary(X, y, W1, b1, W2, b2, W3, b3, activation='sigmoid')
plot_epoch_sigmoid(X, y, W1, b1, W2, b2, W3, b3, activation='sigmoid')
plot_accuracy_per_epoch(X, y, epochs=epochs, activation='sigmoid')


#Relu
losses_sgd_relu, acc_sgd_relu = train_sgd(X, y, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, activation='relu')
plot_decision_boundary(X, y, W1, b1, W2, b2, W3, b3, activation='relu')
plot_epoch_relu(X, y, W1, b1, W2, b2, W3, b3, activation='relu')
plot_accuracy_per_epoch_relu(X, y, epochs=epochs, activation='relu')
