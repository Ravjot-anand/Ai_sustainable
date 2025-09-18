# 🌱 Precision Irrigation Dashboard

A machine learning–powered desktop application (Tkinter GUI) that predicts **daily water requirements for crops** based on crop type, soil type, region, temperature, and weather conditions.  
The goal is to assist farmers and researchers in optimizing irrigation schedules and improving water efficiency.

---

## ✨ Features
- ✅ Load and preprocess dataset (`.csv`) automatically.  
- ✅ Train or load a **Random Forest Regressor** model (saved with `joblib`).  
- ✅ Automatic hyperparameter tuning with **GridSearchCV**.  
- ✅ GUI built with **Tkinter** for easy user input.  
- ✅ Real-time irrigation recommendations (mm/day) with time-of-day suggestion.  
- ✅ Logs predictions with timestamps in `irrigation_log.txt`.  
- ✅ Auto-generated performance plots:
  - Scatter plot of actual vs predicted
  - Correlation heatmap of features  
- ✅ Displays **R² score** and **Mean Absolute Error (MAE)** inside the GUI.

---

## 🛠️ Tech Stack
- **Python 3.9+**
- **Pandas / NumPy**
- **scikit-learn**
- **Matplotlib / Seaborn**
- **Joblib**
- **Tkinter (GUI)**

---

## 📂 Project Structure

📊 Model Performance

Performance is evaluated using:

R² Score

Mean Absolute Error (MAE)

These metrics are shown in the console and inside the GUI window.
Plots (scatterplot & heatmap) are auto-generated when training.

⚠️ Notes

If the dataset (DATASET - Sheet1.csv) is large, consider:

Using Git LFS

Or uploading it as a GitHub Release asset instead of committing directly.

Pre-trained models (irrigation_model.joblib) can also be shared the same way.

📌 Future Improvements

Add support for more ML models (XGBoost, Gradient Boosting).

Provide a web-based dashboard (Flask/Django + React).

Include weather API integration for real-time inputs.
