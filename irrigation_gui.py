import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import tkinter as tk
from tkinter import ttk, messagebox
import warnings
import datetime
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


# Loading and preprocessing the dataset
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()
    df = df[df['WATER REQUIREMENT'].apply(lambda x: isinstance(x, (int, float)) and not np.isnan(x))]

    def temp_to_midpoint(temp_range):
        if isinstance(temp_range, str):
            low, high = map(int, temp_range.split('-'))
            return (low + high) / 2
        return temp_range

    df['TEMPERATURE'] = df['TEMPERATURE'].apply(temp_to_midpoint)

    categorical_cols = ['CROP TYPE', 'SOIL TYPE', 'REGION', 'WEATHER CONDITION']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, label_encoders


# Training or loading the model
def train_or_load_model(df, model_path='irrigation_model.joblib'):
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("✅ Loaded pre-trained model.")
        return model, None
    else:
        X = df[['CROP TYPE', 'SOIL TYPE', 'REGION', 'TEMPERATURE', 'WEATHER CONDITION']]
        y = df['WATER REQUIREMENT']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        model = grid_search.best_estimator_
        y_pred = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"✅ Model trained")
        print(f"   - Mean Absolute Error: {mae:.2f} mm/day")
        print(f"   - R² Score: {r2:.2f}")
        print(f"   - Best Hyperparameters: {grid_search.best_params_}")

        joblib.dump(model, model_path)

        # === Automatically plot ===
        # Scatterplot
        plt.figure(figsize=(6, 5))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.xlabel("Actual Water Requirement (mm/day)")
        plt.ylabel("Predicted Water Requirement (mm/day)")
        plt.title("Actual vs Predicted Water Requirement")
        plt.grid(True)
        plt.savefig("scatterplot.png")
        plt.show()

        # Heatmap
        plt.figure(figsize=(8, 6))
        corr = df.corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.savefig("heatmap.png")
        plt.show()

        return model, {"mae": mae, "r2": r2}


# Prediction function with timestamp and time-based recommendation
def predict_water_requirement(model, label_encoders, crop_type, soil_type, region, temperature_range,
                              weather_condition):
    if isinstance(temperature_range, str):
        low, high = map(int, temperature_range.split('-'))
        temp = (low + high) / 2
    else:
        temp = temperature_range

    try:
        input_data = [
            label_encoders['CROP TYPE'].transform([crop_type])[0],
            label_encoders['SOIL TYPE'].transform([soil_type])[0],
            label_encoders['REGION'].transform([region])[0],
            temp,
            label_encoders['WEATHER CONDITION'].transform([weather_condition])[0]
        ]
    except ValueError as e:
        return f"Error: Invalid input value. Ensure inputs match dataset categories. {e}"

    input_df = pd.DataFrame([input_data],
                            columns=['CROP TYPE', 'SOIL TYPE', 'REGION', 'TEMPERATURE', 'WEATHER CONDITION'])
    prediction = model.predict(input_df)[0]

    timestamp = datetime.datetime.now().strftime("%I:%M %p IST, %B %d, %Y")
    current_hour = datetime.datetime.now().hour

    if prediction < 0.5 and weather_condition.upper() == 'RAINY':
        result = f"No irrigation needed for {crop_type} (RAINY condition, predicted requirement: {prediction:.2f} mm/day)"
    else:
        time_recommendation = " (Recommended time: Morning)" if 6 <= current_hour < 12 else \
                              " (Recommended time: Evening)" if 12 <= current_hour < 18 else ""
        result = f"Irrigation recommendation for {crop_type}: {prediction:.2f} mm/day{time_recommendation}"
    return f"{result} (Predicted at {timestamp})"


# Log prediction to file
def log_prediction(prediction, log_path="irrigation_log.txt"):
    with open(log_path, "a") as log_file:
        log_file.write(f"{prediction}\n{'-' * 50}\n")


# GUI Application
class IrrigationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Precision Irrigation Dashboard")
        self.root.geometry("450x550")

        # Load and train/load model on startup
        self.file_path = "DATASET - Sheet1.csv"
        self.model_path = "irrigation_model.joblib"
        self.log_path = "irrigation_log.txt"
        try:
            self.df, self.label_encoders = load_and_preprocess_data(self.file_path)
            self.model, self.metrics = train_or_load_model(self.df, self.model_path)
        except FileNotFoundError:
            messagebox.showerror("Error", "Dataset file 'DATASET - Sheet1.csv' not found!")
            self.root.quit()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load or train model: {e}")
            self.root.quit()

        # GUI Elements
        ttk.Label(root, text="Get Irrigation Recommendation").pack(pady=10)

        # Crop Type
        ttk.Label(root, text="Crop Type:").pack()
        self.crop_var = tk.StringVar(value="BANANA")
        crops = sorted(list(self.df['CROP TYPE'].map(
            {v: k for k, v in dict(enumerate(self.label_encoders['CROP TYPE'].classes_)).items()}).unique()))
        ttk.Combobox(root, textvariable=self.crop_var, values=crops).pack()

        # Soil Type
        ttk.Label(root, text="Soil Type:").pack()
        self.soil_var = tk.StringVar(value="DRY")
        soils = sorted(list(self.df['SOIL TYPE'].map(
            {v: k for k, v in dict(enumerate(self.label_encoders['SOIL TYPE'].classes_)).items()}).unique()))
        ttk.Combobox(root, textvariable=self.soil_var, values=soils).pack()

        # Region
        ttk.Label(root, text="Region:").pack()
        self.region_var = tk.StringVar(value="DESERT")
        regions = sorted(list(self.df['REGION'].map(
            {v: k for k, v in dict(enumerate(self.label_encoders['REGION'].classes_)).items()}).unique()))
        ttk.Combobox(root, textvariable=self.region_var, values=regions).pack()

        # Temperature Range
        ttk.Label(root, text="Temperature Range:").pack()
        self.temp_var = tk.StringVar(value="20-30")
        temps = sorted(list(self.df['TEMPERATURE'].map(
            lambda x: f"{int(x - 5)}-{int(x + 5)}" if isinstance(x, (int, float)) else x).unique()))
        ttk.Combobox(root, textvariable=self.temp_var, values=temps).pack()

        # Weather Condition
        ttk.Label(root, text="Weather Condition:").pack()
        self.weather_var = tk.StringVar(value="SUNNY")
        weathers = sorted(list(self.df['WEATHER CONDITION'].map(
            {v: k for k, v in dict(enumerate(self.label_encoders['WEATHER CONDITION'].classes_)).items()}).unique()))
        ttk.Combobox(root, textvariable=self.weather_var, values=weathers).pack()

        # Predict Button
        ttk.Button(root, text="Predict", command=self.predict).pack(pady=10)

        # Result Label
        self.result_var = tk.StringVar()
        ttk.Label(root, textvariable=self.result_var, wraplength=400, justify="left").pack(pady=10)

        # Status Bar
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(root, textvariable=self.status_var, relief="sunken", anchor="w").pack(side="bottom", fill="x")

    def predict(self):
        self.status_var.set("Predicting...")
        self.root.update()

        crop = self.crop_var.get()
        soil = self.soil_var.get()
        region = self.region_var.get()
        temp = self.temp_var.get()
        weather = self.weather_var.get()

        result = predict_water_requirement(self.model, self.label_encoders, crop, soil, region, temp, weather)
        self.result_var.set(result)
        log_prediction(result, self.log_path)
        self.status_var.set("Prediction complete")


# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = IrrigationGUI(root)
    root.mainloop()
