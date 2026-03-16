# Mobile User Behavior Classification: A CRISP-DM Pipeline

##  Executive Summary
This project delivers a high-performance classification pipeline designed to categorize mobile users into five behavioral tiers. By leveraging device telemetry—including battery kinetics, app engagement, and data throughput—the model achieves a **1.00 Accuracy Score**, identifying the core drivers of user activity. This framework is engineered for integration into predictive battery management systems and targeted marketing optimization engines.

---

## 🛠 Tech Stack
* **Language:** Python 3.x
* **Libraries:** `pandas`, `numpy`, `scikit-learn`
* **Visualization:** `seaborn`, `matplotlib`
* **Methodology:** CRISP-DM (Cross-Industry Standard Process for Data Mining)

---

## Methodology: The CRISP-DM Framework

### 1. Business Understanding
Mobile hardware manufacturers and service providers require granular user segmentation to optimize OS performance (e.g., adaptive battery) and improve CX. The objective is to accurately map usage metrics to five distinct behavioral classes.

### 2. Data Understanding
The analysis utilized `user_behavior_dataset.csv` (700 samples).
* **Core Metrics:** App Usage Time (min/day), Screen On Time (hours/day), Battery Drain (mAh/day).
* **Contextual Data:** Age, Gender, Device Model, Operating System.
* **Insights:** Initial EDA revealed high linear separability between classes, particularly along the axis of Battery Drain vs. Screen Time.

### 3. Data Preparation
* **Feature Engineering:** Removed `User ID` to prevent data leakage and overfitting to unique identifiers.
* **Encoding:** Implemented Label Encoding for categorical variables (`Gender`, `Device Model`, `OS`).
* **Scaling:** Applied `StandardScaler` to ensure usage intensity features (measured in hundreds/thousands) do not disproportionately weigh against scaled metrics.
* **Splitting:** Employed an 80/20 stratified split to maintain class balance.

### 4. Modeling & Evaluation
Two models were benchmarked to ensure predictive stability:
* **Random Forest Classifier:** Utilized for non-linear boundary detection and feature importance extraction.
* **Logistic Regression:** Baseline model for testing linear separability.

#### Performance Metrics
| Metric | Result |
| :--- | :--- |
| **Accuracy** | 1.00 |
| **F1-Score (Weighted)** | 1.00 |
| **Top Predictor** | App Usage Time (min/day) |

---

##  Feature Importance Analysis
The Random Forest model identified the following features as the primary drivers of behavior classification:
1.  **App Usage Time:** ~25.6% impact.
2.  **Battery Drain:** ~22.8% impact.
3.  **App Count:** ~20.3% impact.

*Conclusion: Behavioral categories are dictated by device interaction intensity rather than demographic profiles (Age/Gender) or hardware ecosystem (iOS/Android).*

---

##  Deployment Instructions

### Prerequisites
```bash
pip install pandas scikit-learn seaborn matplotlib
