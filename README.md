# 🎶 Music Genre Classification with Deep Learning

This project explores various deep learning architectures for classifying music genres using the GTZAN dataset. It involves both spectrogram images and audio waveform analysis with classical and modern neural network models, including CNNs, RNNs (LSTM), and data augmentation via GANs.

> 🔬 Developed by Abdul Wahid Rukua ([awr1u24@soton.ac.uk](mailto:awr1u24@soton.ac.uk))
> 🎓 University of Southampton – Coursework Project

---

## 📂 Dataset

We use the [GTZAN dataset](https://www.kaggle.com/code/ramoliyafenil/proper-eda-lstm-genre-classifier-for-audio-data) consisting of:

* 999 valid audio samples (1 corrupted audio was excluded)
* Precomputed spectrograms resized to **180×180 pixels**
* Audio split into **10 segments**, normalized to **660,000 samples**
* Extracted features include: **MFCC**, **Spectral Contrast**, **Spectral Centroid**, and **Chroma**

Train/test/validation split: **70% / 20% / 10%**

---

## 🏗️ Model Architectures

### 🔹 Net1: Fully Connected Network (Baseline)

* Flattened spectrogram input
* 2 hidden layers (512 ReLU units)
* 10-class output layer

### 🔹 Net2: CNN

Follows standard 3×3 kernel design with max-pooling:

* Conv2D → Conv2D → MaxPool → Conv2D → Conv2D → MaxPool
* Flatten → FC → Output

### 🔹 Net3: CNN + BatchNorm

Same as Net2 but with **Batch Normalisation** after each convolution.

### 🔹 Net4: CNN + BatchNorm + RMSprop

Same as Net3 but trained using **RMSprop optimizer**

### 🔹 Net5: LSTM (RNN)

* Two-layer stacked LSTM (128 hidden units)
* Dropout 0.3
* Early stopping with `patience = 7`

### 🔹 Net6: LSTM + Conditional GAN (Data Augmentation)

* CGAN generates audio from noise + genre labels
* Generator: Fully connected layers (110 → 66,000 waveform samples)
* Discriminator: FC network on real/fake + genre
* Audio segments undergo same preprocessing as real data

---

## ⚙️ Training & Optimization

* Optimizers: **Adam** and **RMSprop**
* Learning Rates tested: `0.01`, `0.001`, `0.0005`, `0.0001`
* Epochs tested: `50`, `100`, `early stopping` (for RNNs)

---

## 📊 Evaluation Metrics

Each model is evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**

---

## 🧪 Key Results

| Model         | Optimizer | Accuracy | Precision | Recall  | F1-score |
| ------------- | --------- | -------- | --------- | ------- | -------- |
| Net5 (LSTM)   | Adam      | **85%**  | **86%**   | **85%** | **85%**  |
| Net6 (CGAN)   | Adam      | 80%      | 80%       | 80%     | 80%      |
| Net3 (CNN+BN) | Adam      | 71%      | 74%       | 71%     | 71%      |
| Net1 (FCN)    | RMSprop   | 39–60%   | 52–67%    | 39–60%  | 35–60%   |

---

## 🧠 Conclusion & Insights

This project demonstrates that **model choice, optimization strategy, and data representation** play critical roles in music genre classification.

Key conclusions:

* **CNNs** significantly outperform fully connected networks when working with image-based spectrograms, confirming their superior spatial inductive bias.
* **LSTMs (Net5)** were most effective overall, reaching **85% F1-score**, showing that **temporal patterns in audio** are crucial and better captured by sequential models.
* **Data augmentation using CGANs (Net6)** showed **mixed results**:

  * When raw audio was generated and re-processed, performance slightly dropped (to **80% F1-score**), indicating that GAN-generated waveforms may introduce **noise rather than meaningful variation**.
  * However, applying GANs to **preprocessed feature representations (e.g., MFCC, chroma)** led to improved generalization, achieving up to **92% accuracy** in external tests.
  * This suggests that **feature-domain augmentation** is a more promising and stable direction than raw waveform synthesis for this task.

Moreover, **optimizer choice matters**:

* **Adam** consistently outperformed **RMSprop**, offering more stability across different architectures.

Finally, longer training **does not guarantee better generalization**. Several models showed early signs of overfitting beyond 50 epochs, reinforcing the importance of **early stopping** and **model regularization** in deep learning workflows.
---

## 📁 Project Structure

```
final_deep_learning/
├── models/
├── data/
│   ├── audio/
│   └── spectrograms/
├── utils/
│   └── feature_extraction.py
├── train.py
├── evaluate.py
└── README.md
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 📚 References

* [Goodfellow et al., Deep Learning (MIT Press, 2016)](http://www.deeplearningbook.org)
* Wafaa Shihab Ahmed et al., *Impact of filter size on CNN*, IEEE (2020)
* Prechelt, *Early Stopping – but when?*, Springer (2002)
