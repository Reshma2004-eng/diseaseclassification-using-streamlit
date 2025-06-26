# diseaseclassification-using-streamlit
# 🔬 Benign vs Malignant Image Classifier

This project is a machine learning-based image classification dashboard that distinguishes between **benign** and **malignant** medical images (e.g., tumor scans).  
It uses multiple ML algorithms to train on labeled image data and allows real-time predictions via a user-friendly **Streamlit** interface.
"""for dataset contact me in linkedin  https://www.linkedin.com/in/sakalabattula-reshma-1b50a2342"""
---

## 📁 Project Structure


dataset/
├── data/
│ ├── train/
│ │ ├── benign/
│ │ └── malignant/
│ └── test/
│ ├── benign/
│ └── malignant/
├── app.py # Streamlit dashboard
├── train_model.py # Model training script (saves best model)
├── bestmodel.pkl # Saved best-performing model
├── model_metrics.csv # Accuracy, F1 Score, Precision of all 5 models
└── README.md # Project documentation


---

## 🚀 Features

- Trains 5 models (Logistic Regression, SVM, KNN, Random Forest, Naive Bayes)
- Automatically saves the best model (`bestmodel.pkl`)
- Displays accuracy, F1 Score, and precision of each model
- Highlights best model in dashboard
- Supports:
  - Uploading a new image for prediction
  - Predicting 10 random test images with labels

---

## ⚠️ Dataset Note

This project uses a large image dataset consisting of:

- `train/benign/` and `train/malignant/` images
- `test/benign/` and `test/malignant/` images

Due to GitHub's file size restrictions, the dataset is **not included** here.  
If you’d like access to the dataset, feel free to reach out via my [LinkedIn](https://www.linkedin.com/in/your-profile/) 💬

---

## 🧠 Model Training

Run the training script:

```bash
python train_model.py
This will train 5 algorithms, evaluate them, and save the best model as bestmodel.pkl.
🖥️ Run the Dashboard
Launch the Streamlit dashboard using:

streamlit run app.py
📌 Author
Reshma Sakalabattula
📫 LinkedIn Profile ( https://www.linkedin.com/in/sakalabattul
 a-reshma-1b50a2342)

