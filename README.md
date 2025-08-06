🗑️ Garbage Classification AI

A 4-week undergraduate research project using CNN-based image classification to automatically classify waste into recyclable categories and improve sustainability awareness.

## 🔍 Motivation

As the world faces growing environmental concerns, efficient waste separation is becoming increasingly important. Manual recycling systems often suffer from human error and inconsistency. This project explores how deep learning can assist in automated garbage classification using image-based inputs, thereby supporting smarter waste management.

## 🎯 Objectives

* Classify garbage images into categories: cardboard, glass, metal, paper, plastic, trash...
* Train and evaluate CNN-based models on real-world image data
* Improve model performance through data augmentation and transfer learning
* Visualize model predictions and attention areas using techniques like Grad-CAM
* (Optional) Explore lightweight models for potential mobile or web-based UI integration


## 🗓️ Timeline (4 weeks)
| Week | Tasks |
|------|-------|
| 1 | Topic confirmation, dataset selection, related papers review |
| 2 | Model setup (CNN - MobilenetV2), data preprocessing |
| 3 | Training, evaluation, visualization |
| 4 | Paper writing (Overleaf), GitHub final polish |

## 🗄️ Structure
```
Garbage-Classifier/
├── data/
├── src/
      ├── dataset.py
      ├── model.py
      ├── train.py
      └── utils.py
├── notebooks/
      ├── baseline_experiment.ipynb
      └── customised_experiment.ipynb
├── paper/
├── requirements.txt
└── README.md
```
## 📁 Dataset
- https://www.kaggle.com/datasets/mostafaabla/garbage-classification/data
- 12 Classes: battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass
- ~9,933 labeled images

## 🧠 Models
- MobileNetV2
- DenseNet121
- Squeezenet
- ShufflenetV2

## 📊 Metrics
- Accuracy, Precision, Recall, F1-Score
- Class-wise evaluation
- Grad-CAM for visual interpretation of predictions

## 📄 Tools
- Google Colab as Editor
- Python, PyTorch, scikit-learn for research and experiment
- Jupyter Notebook, Matplotlib for visualisation
- Overleaf (LaTeX) for research paper

## Author
- Doyeop Kim
- BSc Computer Science, University of Exeter, United Kingdom
- ISS Research Internship, Sungkyunkwan University, Republic of Korea
- August 2025

## Acknowledgements
This project is supported by the ISS Research Internship Programme at Sungkyunkwan University under the guidance of Prof. Khan Muhammad and TA Shehzad Ali.
