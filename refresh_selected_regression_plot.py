"""Refresh the selected forest's plot from its saved fit, without rerunning tuning."""
import os
os.environ["MPLBACKEND"] = "Agg"
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from src.config import MODELS_DIR
from src.data import prepare_dataset, load_diabetes_dataset, get_regression_frame
from src.plots import set_publication_style, plot_regression_diagnostics, save_figure

X, y = get_regression_frame(prepare_dataset(load_diabetes_dataset()).clean, target_column="Glucose")
train, test, train_y, test_y = train_test_split(X, y, test_size=.2, random_state=42)
model = joblib.load(MODELS_DIR / "best_regressor.joblib")
assert isinstance(model.named_steps["model"], RandomForestRegressor)
prediction = model.predict(test)
assert np.isclose(np.sqrt(np.mean((prediction - test_y) ** 2)), 50.69265168190375)
set_publication_style()
figure = plot_regression_diagnostics(test_y, prediction, test_y - prediction)
figure.suptitle("Training-selected Random Forest Regressor (test n=78)", fontsize=18, y=.99)
figure.subplots_adjust(top=.80, wspace=.30)
save_figure(figure, "regression_diagnostics.png")
print("Selected Random Forest Regressor figure refreshed with title clearance.")
