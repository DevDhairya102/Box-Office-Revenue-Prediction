# Box Office Revenue Prediction: An Ensemble Learning Approach with Feature Engineering and Comprehensive Evaluation

**Author:** Aaryan Lunis 

**Institution:** Department of Computer Engineering, Mukesh Patel School of Technology Management & Engineering, SVKM’s NMIMS, Mumbai, India

**Repository:** [Box-Office-Revenue-Prediction](https://github.com/Aaryan-Lunis/Box-Office-Revenue-Prediction)

**Python Version:** 3.10+

---

## 📘 Abstract

Accurately forecasting box office revenue is a challenging task due to the complex interplay of artistic, financial, and audience-driven factors.
This project introduces an **ensemble-based learning framework** that combines **feature engineering** and **rigorous evaluation** to improve predictive accuracy.
Using a dataset of **4,880 films**, multiple machine learning algorithms—Linear, Ridge, Lasso, Random Forest, and Gradient Boosting—were tested.
The **Gradient Boosting model** achieved the best results, explaining **76.4% of the variance (R² = 0.7639)** in box office revenue.
This repository provides all code, data pipelines, and analysis supporting the study.

---

## 🧠 Research Objective

To design and evaluate a **data-driven framework** that can predict movie box office revenue using engineered features and ensemble machine learning models.
The study aims to assist producers, distributors, and marketing professionals in making informed investment and scheduling decisions.

---

## 📂 Repository Structure

```
Box-Office-Revenue-Prediction/
│
├── models/                        # Saved model files (trained and tuned versions)
├── results/                       # Evaluation outputs and prediction logs
├── data_preprocessing.py          # Data cleaning and transformation pipeline
├── exploratory_analysis.py        # EDA: feature distribution, correlations, visualizations
├── generate_historical_averages.py# Audience engagement parameter estimation
├── main_script.py                 # End-to-end execution script
├── model_training.py              # Training of regression and ensemble models
├── model_evaluation.py            # Evaluation metrics computation (R², RMSE, MAE)
├── app.py                         # Application interface for revenue prediction
├── requirements.txt               # Project dependencies
├── movies_metadata.csv            # Source dataset (if available)
├── historical_averages.csv        # Generated data from engagement lookup system
├── predictions_log.csv            # Model output logs
└── BoxOfficeRevenuePrediction.pdf # Full research paper (reference document)
```

---

## ⚙️ Methodology

### 1. Dataset Description

* Total movies analyzed: **4,880**
* Features include: `budget`, `revenue`, `vote_count`, `runtime`, `genres`, `release_year`
* Dataset derived by merging metadata and credits files, filtered for quality and completeness.

### 2. Feature Engineering

* **Budget per minute**, **log-transformed features**, **vote-weighted scores**
* **Temporal indicators** such as release year and decade classification
* **Audience engagement metrics**: vote count, popularity, and genre-based interaction
* Hierarchical averaging for missing audience metrics using similar *actor-director-genre* combinations

### 3. Model Development

* Algorithms implemented:
  `Linear Regression`, `Ridge`, `Lasso`, `Random Forest`, `Gradient Boosting`
* Evaluation metrics: **R²**, **RMSE**, **MAE**, and **Cross-Validation RMSE**
* Cross-validation used for robust hyperparameter tuning
* Implementation built on **Scikit-learn** and **NumPy**

---

## 📊 Results and Evaluation

| Model                 |   Test R²  | RMSE (in $M) | MAE (in $M) |
| :-------------------- | :--------: | :----------: | :---------: |
| Gradient Boosting     | **0.7639** |   **93.7**   |   **45.1**  |
| Random Forest (Tuned) |   0.7412   |     98.1     |     45.0    |
| Ridge Regression      |   0.7262   |     100.9    |     51.3    |
| Lasso Regression      |   0.7262   |     100.8    |     51.4    |
| Linear Regression     |   0.7262   |     100.8    |     51.4    |

* **Gradient Boosting** demonstrated superior predictive accuracy, explaining **76.4% of revenue variance.**
* Strongest predictors: **Vote Count (0.773)** and **Budget (0.734)**.
* The residual analysis confirmed robust model generalization and error control.

---

## 🔍 Key Insights

* Ensemble models capture **non-linear interactions** between financial and audience factors.
* **Feature engineering** significantly enhances model interpretability.
* Framework supports **scenario-based forecasting** using historical audience engagement profiles.

---

## 🚀 How to Run the Project

1. Clone this repository:

   ```bash
   git clone https://github.com/Aaryan-Lunis/Box-Office-Revenue-Prediction.git
   cd Box-Office-Revenue-Prediction
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Execute preprocessing and model training:

   ```bash
   python main_script.py
   ```
4. Run the prediction interface:

   ```bash
   python app.py
   ```

---

## 🔬 Future Work

* Integrate **multimodal learning** with trailers and posters using computer vision
* Explore **transformer-based hybrid models** for sequential trend learning

---

## 🧾 License

This project is released under the **MIT License** — free to use, modify, and distribute with attribution.

---

## 📚 Citation

If you use this work in your research or projects, please cite:

```
Lunis, A. (2025). Box Office Revenue Prediction: An Ensemble Learning Approach with Feature Engineering and Comprehensive Evaluation. GitHub Repository. https://github.com/Aaryan-Lunis/Box-Office-Revenue-Prediction
```

---

## 📩 Contact

**Aaryan Lunis**
Email: [lunisaaryan@gmail.com](mailto:lunisaaryan@gmail.com)
LinkedIn: [linkedin.com/in/aaryan-lunis](https://linkedin.com/in/aaryan-lunis)

---
