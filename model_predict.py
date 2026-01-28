import sys
import numpy as np
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# -----------------------------------------------------
# model_predict.py
# Usage:  python model_predict.py <energy> <snr_db>
# Prints: 0 or 1 (PU absent / present)
# -----------------------------------------------------

# Ignore the version mismatch warning from sklearn
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore")  # (optional) ignore all warnings

if len(sys.argv) != 3:
    print(0)
    sys.exit(1)

energy_value = float(sys.argv[1])
snr_value    = float(sys.argv[2])

# Use full paths to be safe
model  = joblib.load("your_path_to_model")
scaler = joblib.load("your_path_to_model")

x = np.array([[energy_value, snr_value]], dtype=float)
x_scaled = scaler.transform(x)
pred = model.predict(x_scaled)

# Print ONLY 0 or 1
print(int(pred[0]))
