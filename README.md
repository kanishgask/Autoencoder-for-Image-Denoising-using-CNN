# Autoencoder-for-Image-Denoising-using-CNN
**🧼 Denoising Autoencoder with CNN on MNIST**
This project implements a convolutional autoencoder to remove noise from handwritten digit images from the MNIST dataset. It uses a simple CNN-based encoder-decoder architecture trained to reconstruct clean images from noisy inputs.

**📌 Overview**
Dataset: MNIST handwritten digits

Task: Image denoising

Architecture: Convolutional Autoencoder

Loss: Mean Squared Error (MSE)

Visualizes: Noisy input, denoised output, and original image

**🧰 Requirements**
Python 3.x

TensorFlow 2.x

NumPy

Matplotlib

Install dependencies:

bash
Copy
Edit
pip install tensorflow numpy matplotlib
**🧠 How It Works**
1. Load and Preprocess Data
Loads MNIST grayscale images (28x28)

Normalizes pixel values to [0, 1]

Adds Gaussian noise to create noisy images

2. Autoencoder Architecture
🔒 Encoder:
Conv2D → ReLU → MaxPooling

Repeats to reduce spatial dimensions

🔓 Decoder:
Conv2D → ReLU → UpSampling

Reconstructs the denoised image using a sigmoid output

plaintext
Copy
Edit
Input (28x28x1)
   ↓
Conv2D + MaxPool
   ↓
Conv2D + MaxPool
   ↓
Conv2D + UpSample
   ↓
Conv2D + UpSample
   ↓
Output (28x28x1)
3. Training
Trained to minimize MSE between the original and reconstructed images

Uses noisy inputs as input, and clean images as targets

Trained for 5 epochs with a batch size of 128

4. Evaluation
Predicts denoised images from noisy test images

Displays:

Top Row: Noisy input

Middle Row: Denoised output

Bottom Row: Original image

**🖼️ Example Visualization**
A sample output shows how well the autoencoder removes noise:

mathematica
Copy
Edit
+----------------+----------------+----------------+
|     Noisy      |    Denoised    |    Original    |
+----------------+----------------+----------------+
|    Image 1     |    Image 1     |    Image 1     |
|    Image 2     |    Image 2     |    Image 2     |
|    Image 3     |    Image 3     |    Image 3     |
🧪 Try It Out
python
Copy
Edit
decoded_imgs = autoencoder.predict(x_test_noisy)

# Visualize first 5 examples
for i in range(5):
    ...
💾 Saving the Model
To save the model after training:

python
Copy
Edit
autoencoder.save("mnist_denoising_autoencoder.h5")
📚 References
Keras Autoencoder Docs

MNIST Dataset

🧼 Denoising Autoencoder with CNN on MNIST
This project implements a convolutional autoencoder to remove noise from handwritten digit images from the MNIST dataset. It uses a simple CNN-based encoder-decoder architecture trained to reconstruct clean images from noisy inputs.

📌 Overview
Dataset: MNIST handwritten digits

Task: Image denoising

Architecture: Convolutional Autoencoder

Loss: Mean Squared Error (MSE)

Visualizes: Noisy input, denoised output, and original image

🧰 Requirements
Python 3.x

TensorFlow 2.x

NumPy

Matplotlib

Install dependencies:

bash
Copy
Edit
pip install tensorflow numpy matplotlib
🧠 How It Works
1. Load and Preprocess Data
Loads MNIST grayscale images (28x28)

Normalizes pixel values to [0, 1]

Adds Gaussian noise to create noisy images

2. Autoencoder Architecture
🔒 Encoder:
Conv2D → ReLU → MaxPooling

Repeats to reduce spatial dimensions

🔓 Decoder:
Conv2D → ReLU → UpSampling

Reconstructs the denoised image using a sigmoid output

plaintext
Copy
Edit
Input (28x28x1)
   ↓
Conv2D + MaxPool
   ↓
Conv2D + MaxPool
   ↓
Conv2D + UpSample
   ↓
Conv2D + UpSample
   ↓
Output (28x28x1)
3. Training
Trained to minimize MSE between the original and reconstructed images

Uses noisy inputs as input, and clean images as targets

Trained for 5 epochs with a batch size of 128

4. Evaluation
Predicts denoised images from noisy test images

Displays:

Top Row: Noisy input

Middle Row: Denoised output

Bottom Row: Original image

🖼️ Example Visualization
A sample output shows how well the autoencoder removes noise:

mathematica
Copy
Edit
+----------------+----------------+----------------+
|     Noisy      |    Denoised    |    Original    |
+----------------+----------------+----------------+
|    Image 1     |    Image 1     |    Image 1     |
|    Image 2     |    Image 2     |    Image 2     |
|    Image 3     |    Image 3     |    Image 3     |
🧪 Try It Out
python
Copy
Edit
decoded_imgs = autoencoder.predict(x_test_noisy)

# Visualize first 5 examples
for i in range(5):
    ...
💾 Saving the Model
To save the model after training:

python
Copy
Edit
autoencoder.save("mnist_denoising_autoencoder.h5")
📚 References
Keras Autoencoder Docs

MNIST Dataset

**📄 License**
This project is licensed under the MIT License
This project is licensed under the MIT License
