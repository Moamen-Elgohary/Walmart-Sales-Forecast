# Music Genre Classification

> **Can a machine tell jazz from metal, or blues from reggae — just by listening?**

## Problem

Manually tagging music by genre is impractical at scale, and genre boundaries are inherently fuzzy. The challenge is finding the right way to represent audio so a model can learn meaningful patterns across 10 distinct genres.

## Solution

This project compares two representation strategies on the GTZAN dataset: hand-crafted tabular audio features (MFCCs, spectral centroids, tempo) fed into Random Forest, XGBoost, and MLP models, versus mel spectrogram images processed by a Custom CNN and VGG16 transfer learning. The goal is to determine which approach best solves the classification problem.

---

## Dataset

- **Source:** [GTZAN Dataset via KaggleHub](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- **Size:** 1,000 audio samples across 10 genres (100 per genre)
- **Genres:** blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
- **Files Used:**
  - `features_30_sec.csv` — Pre-extracted audio features (MFCCs, spectral features, etc.)
  - `images_original/` — Mel spectrogram PNG images

---

## Project Structure

```
Music_Genre_Classification.ipynb
│
├── Libraries
├── Dataset
├── Exploratory Data Analysis
├── Data Splitting
├── Tabular Modeling
│   ├── Preprocessing
│   ├── Random Forest
│   ├── XGBoost
│   └── MLP (Multi-Layer Perceptron)
├── Image-Based Modeling
│   ├── Preprocessing
│   ├── Custom CNN
│   └── Transfer Learning (VGG16)
└── Comparison and Analysis
```

---

## Exploratory Data Analysis

Before modeling, the dataset was explored from two angles: structure and sound.

On the structural side, the feature CSV was inspected for shape, data types, missing values, and statistical distributions across all extracted features. A count plot confirmed balanced class distribution with exactly 100 samples per genre, ruling out any class imbalance concerns.

On the audio side, librosa was used to load one audio file per genre and generate two visualizations for each: a waveform plot showing amplitude over time, and a mel spectrogram displaying how frequency energy is distributed across time on a perceptual scale. These plots made genre differences tangible — classical tracks showed clean, dynamic waveforms while metal and rock exhibited dense, high-energy patterns throughout. Genres like jazz and blues revealed more irregular, improvisational structure compared to the rhythmic consistency seen in disco and hiphop.

Together, these steps confirmed the data was clean, balanced, and rich enough in acoustic variation to support classification.

---

## Data Splitting

The dataset was split 70% train / 15% validation / 15% test with stratification per genre, ensuring balanced representation in each split. `LabelEncoder` was used to convert genre names to integer class labels, followed by one-hot encoding for image-based models.

---

## Models & Results

### Tabular Models

---

#### Random Forest

Random Forest is an ensemble learning method that builds a large number of decision trees during training and outputs the class that is the mode of the individual trees' predictions. Each tree is trained on a random subset of the data and a random subset of features, which reduces overfitting and improves generalization. For this task, the model learns to associate combinations of audio features — such as spectral centroid, MFCCs, and zero-crossing rate — with specific genres. Its strength lies in interpretability and robustness to noisy features.

| Metric | Score |
|---|---|
| Validation Accuracy | 68.0% |
| Test Accuracy | 69.0% |

Random Forest performed best on genres with strongly distinct acoustic profiles such as metal, classical, and jazz. It struggled more with genres that share overlapping rhythmic or timbral characteristics, such as disco and hiphop.

---

#### XGBoost

XGBoost (Extreme Gradient Boosting) is a gradient boosting framework that builds an ensemble of decision trees sequentially, where each new tree is trained to correct the errors of the previous one. Unlike Random Forest which builds trees in parallel, XGBoost focuses iteratively on misclassified samples, making it more sensitive to subtle distinctions between similar genres. It also includes built-in regularization to prevent overfitting.

| Metric | Score |
|---|---|
| Validation Accuracy | 72.7% |
| Test Accuracy | 70.7% |

XGBoost improved over Random Forest, particularly on genres with overlapping characteristics, by learning to progressively refine its decision boundaries. The slight drop between validation and test accuracy suggests minor overfitting to the validation distribution, but overall generalization remained solid.

---

#### MLP (Multi-Layer Perceptron)

The MLP is a fully connected feedforward neural network consisting of an input layer, one or more hidden layers, and an output layer. Each neuron applies a weighted sum of its inputs followed by a non-linear activation function (ReLU), allowing the network to learn complex, non-linear relationships between features. The MLP was trained with standard backpropagation and dropout regularization to reduce overfitting. Unlike tree-based models, the MLP can capture interactions across all features simultaneously rather than splitting on one feature at a time.

| Metric | Score |
|---|---|
| Validation Accuracy | 77.3% |
| Test Accuracy | 74.7% |

The MLP achieved the highest accuracy among all five models. Its ability to model non-linear feature interactions gave it an edge over the tree-based approaches, and the small gap between validation and test accuracy indicates good generalization.

---

### Image-Based Models
---

#### Custom CNN

The Custom CNN is a convolutional neural network designed and trained from scratch specifically for this task. It consists of three convolutional layers with 32, 64, and 128 filters respectively, each followed by ReLU activation, max pooling to reduce spatial dimensions, and dropout for regularization. The output of the convolutional layers is flattened and passed through a dense classifier. CNNs are well-suited to image data because they exploit local spatial structure — in this case, patterns in the time-frequency representation of audio — rather than treating each pixel independently. The mel spectrogram images were resized to 128x128 before being fed into the network.

| Metric | Score |
|---|---|
| Validation Accuracy | 60.7% |
| Test Accuracy | 60.4% |

The Custom CNN was the weakest model overall. With only 1,000 samples across 10 classes, the network lacked sufficient data to learn robust visual representations from scratch.

---

#### VGG16 (Transfer Learning)

VGG16 is a deep convolutional neural network originally trained on ImageNet, a dataset of over 1 million images across 1,000 categories. Transfer learning leverages the feature representations already learned by VGG16 — edges, textures, shapes — and applies them to a new domain. In this project, the pre-trained VGG16 base was retained with its weights frozen, and a new classification head consisting of global average pooling and a dense softmax output layer was added on top. Input images were resized to 224x224 to match VGG16's expected input size. The model was then fine-tuned on the mel spectrogram dataset.

| Metric | Score |
|---|---|
| Validation Accuracy | 64.7% |
| Test Accuracy | 68.4% |

VGG16 significantly outperformed the Custom CNN, demonstrating the value of transfer learning when labeled training data is limited. However, it still fell short of the tabular MLP, suggesting that the pre-trained visual features from ImageNet do not transfer as effectively to audio spectrograms as they do to natural images.

---

### Overall Comparison

| Approach | Model | Val Accuracy | Test Accuracy |
|---|---|---|---|
| Tabular | Random Forest | 68.0% | 69.0% |
| Tabular | XGBoost | 72.7% | 70.7% |
| Tabular | MLP | 77.3% | 74.7% |
| Image-Based | Custom CNN | 60.7% | 60.4% |
| Image-Based | VGG16 Fine-tuned | 64.7% | 68.4% |

---

## Conclusion

Tabular models outperformed image-based models across the board, with the MLP achieving the best test accuracy at 74.7%. Hand-crafted audio features proved to be a more effective representation than mel spectrograms at this dataset size — the CNN struggled without enough data to learn from scratch, and even VGG16 transfer learning couldn't close the gap. When strong domain-specific features are available, classical approaches remain highly competitive.

---

## Libraries Used

```python
numpy, pandas, matplotlib, seaborn, librosa, soundfile
scikit-learn (RandomForestClassifier, StandardScaler, LabelEncoder)
xgboost (XGBClassifier)
tensorflow / keras (CNN, VGG16, MLP)
kagglehub
```

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Moamen-Elgohary/Music-Genre-Classification
   ```

2. Install dependencies:
   ```bash
   pip install kagglehub librosa soundfile xgboost scikit-learn tensorflow pandas numpy matplotlib seaborn
   ```

3. Open and run the notebook:
   ```bash
   jupyter notebook Music_Genre_Classification.ipynb
   ```

---

## Dataset License

The GTZAN dataset was collected by George Tzanetakis and is widely used for research purposes. It is made available for non-commercial academic use only. When using this dataset, please cite the original paper:

> Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. *IEEE Transactions on Speech and Audio Processing*, 10(5), 293-302.

The Kaggle version of the dataset is hosted by Andrada Olteanu and is subject to Kaggle's terms of service. Users are responsible for ensuring their use complies with those terms. This project does not redistribute the dataset.

This project was completed as part of the Elevvo Pathways internship program.