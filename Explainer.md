# Image Colorization using Deep Learning (Naive CNN & Pix2Pix CGANs)

This project focuses on automatically colorizing grayscale images of natural scenes (birds in this case) using deep learning. Two primary approaches are investigated: a **Naive Model** and a **GAN-based Model**. The goal is to produce visually appealing colorized images from low-information grayscale inputs.

---

## Dataset Details

- **Dataset:** UK Garden Birds(sourced from [Kaggle: davemahony/20-uk-garden-birds](https://www.kaggle.com/davemahony/20-uk-garden-birds))
- **Content:** Bird images in natural settings, often with background clutter
- **Preprocessing:** All images are resized to 128×128 pixels
- **Input/Output:**
  - Input: Grayscale image
  - Output: Full-color (RGB) image

---

## Project Structure

The project is organized into three main sections:

### 1. Notebook 1 – Naive_CNN_Colorization.ipynb


This category includes three experiments using an autoencoder-based architecture that converts a grayscale image (1 channel) into an RGB image (3 channels). Each experiment explores different hyperparameter settings and architectural variations:

- **Experiment 1:** Basic architecture with minimal modifications.
- **Experiment 2:** Architecture enhanced with dropout to mitigate overfitting.
- **Experiment 3:** A deeper architecture with batch normalization – this experiment yielded the best results among the Naive Models.

### 2. Notebook 2 – Pix2Pix_CGAN_Colorization.ipynb


This category investigates GAN-based approaches for image colorization. Two distinct architectures are explored:

- **Architecture 1 – Using GAN:**A conditional GAN model that employs a U-Net–style generator and a corresponding discriminator. Multiple experiments were run with various hyperparameters and training strategies.
- **Architecture 2 – (Not included in Test Environment):**
  (Other GAN-based experiments were conducted; however, for the final test environment only the above two models are used.)

### 3. Notebook 3 – Test_Environment.ipynb


The Test Environment notebook is used to evaluate the best performing models from the two categories. It:

- Downloads the pre-trained weights using `gdown` (with provided Google Drive IDs).
- Preprocesses user-uploaded images according to each model’s expected input:
  - **Naive CNN Model:** Expects a grayscale input (128×128×1).
  - **Pix2Pix CGAN Model:** Expects an RGB input (128×128×3).
- Runs inference on the models and displays the results side by side for comparison.
- **Note:** The earlier Pix2Pix model was removed from the test environment so that only the two models above are evaluated.

---

## Colab Notebook Structure

The notebooks are divided into the following sections:

1. **Utils & Installations**

   - Installation of required libraries (e.g., TensorFlow, NumPy, Matplotlib, gdown).
   - Downloading pre-trained models via `gdown`.
2. **Data Preprocessing**

   - Downloading and organizing the UK Garden Birds dataset.
   - Resizing images to 128×128 pixels and splitting data into training, validation, and test sets.
   - Data augmentation techniques (e.g., random jitter, horizontal flip) are applied during training.
3. **Model Architecture & Training**

   - **Naive Model:**Contains three experiments with different autoencoder architectures.
   - **With GAN:**Contains experiments using a conditional GAN setup with a U-Net generator and a discriminator.
   - Training routines include separate loss definitions for the generator and discriminator (for the GAN model).
4. **Test Environment**

   - Downloads the best performing models:
     - **Naive CNN Best Model:** The best Naive Model (Experiment 3).
     - **Pix2Pix CGAN Model:** The best GAN-based model.
   - Preprocesses user images:
     - Converts images to grayscale for Naive CNN Model.
     - Converts images to RGB for the Pix2Pix CGAN Model.
   - Runs inference and displays the original input alongside the outputs from each model.
5. **Results & Evaluation**

   - Presents training loss convergence graphs.
   - Provides side-by-side visual comparisons of colorized outputs.
   - Qualitative evaluation of model performance based on visual inspection.

---

## Installation & Setup

1. **Run the Installations and Imports section:**

   - Install dependencies using `pip` (e.g., `gdown`, `tensorflow`, `numpy`, `matplotlib`).
   - Download the pre-trained model files using the provided `gdown` commands with their Google Drive IDs.
2. **Prepare the Dataset:**

   - Authenticate with Kaggle and download the **UK Garden Birds** dataset.
   - Organize the dataset into training, validation, and test folders.
3. **Configure the Environment:**

   - Ensure a GPU-enabled runtime for faster training, especially when training GANs.
   - Run notebook sections sequentially.

---

## Training the Models

1. **Naive Model Training:**

   - Run the Naive Model notebook to perform three experiments with varying architectures and hyperparameters.
   - Monitor loss convergence and compare generated colorized images.
   - Save the best performing Naive Model (Experiment 3).
2. **GAN-Based Model Training:**

   - Run the GAN notebook to train the conditional GAN models.
   - Alternate updates between the generator and discriminator.
   - Experiment with reconstruction losses (e.g., L1 loss with lambda = 100) and TTA (during training, if applicable) to enhance visual quality.
   - Save the best performing GAN model.

---

## Testing the Models

1. **Load Pre-trained Models:**

   - Use the Test Environment notebook to download and load the two pre-trained models:
     - **ARCH1 Best Model** (Naive approach)
     - **ARC2 Model** (Using GAN)
2. **Run Inference:**

   - Upload an image (128x128, gray-level image).
   - The notebook preprocesses the image appropriately for each model:
     - **ARCH1:** Converts the image to grayscale.
     - **ARC2:** Converts the image to RGB.
   - Display the original grayscale input and the colorized outputs side by side for visual comparison.
3. **Analyze the Results:**

   - Evaluate the colorization quality based on visual output.
   - Compare the performance between the Naive model and the GAN-based model.

---

## Conclusion

This project demonstrates two distinct strategies for image colorization:

- A **Naive Model** based on an autoencoder architecture, which was experimented with in three different variations.
- A **GAN-based Model** that leverages a conditional GAN setup with a U-Net generator and a discriminator to achieve more realistic colorizations.

The Test Environment notebook integrates the best performing models from these approaches (ARCH1 and ARC2) to allow for direct comparison of their outputs on unseen data. The removal of the Pix2Pix model from the test phase streamlines the evaluation to focus on the most promising methods.
