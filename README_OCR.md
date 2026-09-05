# Distorted Visual Sequence Recognition --- Residual CRNN

An end-to-end deep learning OCR pipeline for recognizing text sequences
from **noisy and distorted grayscale images**.

The project focuses on robustness to variations such as blur, small
geometric distortions, missing/occluded regions and alignment changes,
while keeping the recognition pipeline fully trainable end-to-end.

## Overview

**Image → Preprocessing & Augmentation → Residual CNN → BiLSTM → CTC
Decoding → Text**

The model is designed for sequence recognition without requiring
explicit character-level bounding boxes or segmentation.

## Model Architecture

### Residual CNN

The visual encoder uses ResNet-style residual blocks:

-   Convolution + Batch Normalization + ReLU
-   Residual/skip connections
-   Progressive feature extraction from the input image
-   Adaptive average pooling to collapse the spatial height

Residual connections help preserve information and make deeper
convolutional feature extraction easier to optimize.

### Bidirectional LSTM

The CNN feature map is converted into a sequence along the image width
and passed through a **2-layer BiLSTM** with 256 hidden units.

The bidirectional setup allows the model to use both left and right
context while recognizing each part of the sequence.

### CTC Decoding

**Connectionist Temporal Classification (CTC)** is used for training
because the images do not provide explicit alignment between character
labels and image positions.

At inference time, greedy decoding:

1.  Selects the highest-probability class at each timestep.
2.  Removes repeated consecutive predictions.
3.  Removes the CTC blank token.
4.  Produces the final text sequence.

## Preprocessing & Robustness

Input images are converted to grayscale and resized to **64 × 256**.

During training, augmentation is applied using:

-   `RandomAffine` --- handles small rotations, shear and translation.
-   `GaussianBlur` --- improves robustness to blurry/low-quality inputs.
-   `RandomErasing` --- simulates missing or occluded character regions.

Validation uses clean transforms so that model performance is measured
independently of training-time augmentation.

## Training

Key configuration:

  Parameter                               Value
  ------------------------- -------------------
  Input size                           64 × 256
  Batch size                                 64
  Maximum epochs                             60
  Early stopping patience                    10
  Optimizer                               AdamW
  Learning rate                            5e-4
  Weight decay                             1e-4
  LR scheduler                CosineAnnealingLR
  LSTM hidden size                          256
  LSTM layers                                 2
  Dropout                                   0.3
  Loss                                 CTC Loss
  Mixed precision                           AMP

Training uses **Automatic Mixed Precision (AMP)** to reduce GPU memory
usage and accelerate training. Early stopping monitors validation CER
and saves the best checkpoint.

Additional data-loading and training optimizations include pinned
memory, persistent DataLoader workers, non-blocking CPU-to-GPU transfers
and gradient clipping.

## Evaluation

The primary evaluation metric is **Character Error Rate (CER)**:

`CER = Levenshtein distance(prediction, ground truth) / ground-truth length`

A CER of **0.0 is perfect**.

The project also performs qualitative evaluation by comparing predicted
sequences with ground-truth labels on validation images.

## Inference

The trained model:

1.  Loads a grayscale input image.
2.  Applies the validation preprocessing transform.
3.  Runs the Residual CNN + BiLSTM model.
4.  Decodes the output using greedy CTC decoding.
5.  Returns the recognized text sequence.

The final inference pipeline was also used to generate predictions for
**5,000 test images** in the submitted prediction file.

### Example output

  Input            Predicted text
  ---------------- ----------------
  `test-0.png`     `QVTQ8A`
  `test-1.png`     `7PSW9D`
  `test-10.png`    `7DUP98`
  `test-100.png`   `75Z4WT`

> Add a representative distorted image to `sample_input/` and place its
> predicted text/output in `sample_output/` when publishing the
> repository.

## Repository Structure

``` text
distorted-visual-sequence-recognition/
│
├── README.md
├── notebook/
│   └── notebook_DISHA_AGRAWAL_24113039.ipynb
├── sample_input/
│   └── sample.png
├── sample_output/
│   └── result.txt
└── submission/
    └── submission.csv
```

## Tech Stack

-   Python
-   PyTorch
-   Torchvision
-   NumPy
-   Pandas
-   OpenCV
-   Pillow
-   Matplotlib
-   CUDA / Automatic Mixed Precision

## Key Takeaways

This project was built around a practical OCR problem where clean-image
performance alone is not sufficient. The main focus was on learning
robust visual representations, modelling sequence context and evaluating
recognition errors using CER.

The combination of **Residual CNN + BiLSTM + CTC**, distortion-aware
augmentation and training/inference optimizations provides an end-to-end
approach to distorted visual sequence recognition.
