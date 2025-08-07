# Group 18D: Political Bias Detection in News Articles

**Contributors:** Shawn Lin, Godwin Idowu, Yuki Li

---

## Project Title

Using Machine Learning to Identify Political Bias in News Articles

_Applied machine learning algorithms (Linear SVM and Logistic Regression) to classify political bias in news articles (left, center, right), leveraging text classification skills and data analysis techniques learned through the AI4ALL Ignite program._

---

## Problem Statement

News consumers are often exposed to biased information, which can influence opinions and decision-making. Identifying and mitigating such bias is essential for promoting media literacy and transparency.

This project addresses the challenge of automatically classifying political bias in news articles using machine learning techniques, enabling users to better evaluate the credibility and neutrality of the content they consume.

---

## Key Results

1. Trained and evaluated **Linear SVM** and **Logistic Regression** models for political bias classification.
2. Compared algorithm performance using datasets from AllSides, MBIC, and BABE.
3. Analyzed sources of bias in datasets and algorithms (e.g., labeling subjectivity, imbalanced classes, source-label correlations).

---

## Methodologies

- Preprocessed datasets containing thousands of news headlines and articles using Python (tokenization, cleaning, and feature extraction with TF-IDF).
- Trained **Linear Support Vector Machine (Linear SVM)** and **Logistic Regression** models to classify articles as left, center, or right.
- Compared model performance to analyze strengths, weaknesses, and potential biases of each algorithm.

---

## Data Sources

- **AllSides Balanced News Dataset** ([GitHub](https://github.com))
- **MBIC – Media Bias Annotation Dataset** ([Kaggle](https://www.kaggle.com))
- **BABE – Media Bias Dataset: Annotations By Experts** ([Kaggle](https://www.kaggle.com))

---

## Technologies Used

- Python
- scikit-learn
- pandas
- numpy

---

## 📖 Project Summary

Our project investigates how **machine learning (ML)** can be used to automatically identify and classify political bias in news articles, providing users with transparent information about where a given article falls on the political spectrum (**left, center, or right**).

This is important because news consumers are often exposed to information filtered through various forms of bias, which can influence opinions and decision-making. By equipping users with automated, AI-driven tools that highlight possible bias, we aim to promote media literacy and empower readers to critically evaluate the information they consume.

Our approach leverages labeled datasets containing thousands of news headlines, full article texts, and source information. We will train and evaluate both **Linear SVM** and **Logistic Regression** models to classify news articles and analyze their potential for supporting users in recognizing and understanding media bias.

---

## ❓ Research Question

**How effectively can machine learning help users identify article bias?**

Specifically:

- How well can a machine learning model classify news articles as left, center, or right?
- What is the potential of such a system to promote transparency in news consumption?

---

## 🤖 Machine Learning Algorithms

We will use both **Linear Support Vector Machine (Linear SVM)** and **Logistic Regression** as our main machine learning algorithms.

- Both are well-established for text/news classification.
- They are efficient, interpretable, and perform well with high-dimensional textual features.
- Applying both methods to the same datasets allows direct comparison and helps identify algorithm-specific biases.

---

## ⚠️ Sources of Bias

1. **Labeling Subjectivity:** Expert labels may still reflect personal interpretations of bias.
2. **Imbalanced Classes:** Underrepresentation of categories (e.g., "center") can affect fairness.
3. **Source-Driven Patterns:** Models may memorize correlations between sources and labels.
4. **Duplicate/Overlapping Content:** Articles appearing in both training/testing sets can inflate performance (data leakage).
5. **Algorithm Bias:** Linear SVM and Logistic Regression each have inherent limitations and biases.

---

## 📚 Citations

- U.S. Media Polarization and the 2020 Election: A Nation Divided
- AllSides: Media Bias Overview
- Algorithmic Bias in Recommendation Systems and Its Social Impact on User Behavior
- Joachims, T. (1998). _Text categorization with Support Vector Machines: Learning with many relevant features._ ECML.
- Baly, R., Karadzhov, G., Alexandrov, D., Glass, J., & Nakov, P. (2018). _Predicting Factuality and Bias of News Media Sources._ EMNLP.
- Linear SVM Classification
- Logistic Regression Vs Support Vector Machines (SVM)
