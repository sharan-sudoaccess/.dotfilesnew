# ============================================================
# Metal Removal Predictor
# Run this AFTER ANN_Metal_Removal.py has been run once
# (it loads the saved model)
# ============================================================

import numpy as np
import joblib, os, sys

# ── Check model exists ────────────────────────────────────────
if not os.path.exists('model.pkl'):
    print("ERROR: model.pkl not found.")
    print("Please run ANN_Metal_Removal.py first to train and save the model.")
    sys.exit(1)

model_data  = joblib.load('model.pkl')
net         = model_data['net']
scaler_X    = model_data['scaler_X']
scaler_Y    = model_data['scaler_Y']

print("=" * 45)
print("   Metal Removal Efficiency Predictor")
print("=" * 45)
print("Type your values below. Press Ctrl+C to quit.\n")

while True:
    try:
        print("-" * 45)
        metal  = input("Metal type  (Cu or Cr)          : ").strip().upper()
        if metal not in ['CU', 'CR']:
            print("  Please enter Cu or Cr")
            continue

        ph     = float(input("pH          (e.g. 2 to 7)       : "))
        temp   = float(input("Temperature (°C, e.g. 35)       : "))
        time   = float(input("Contact time (min, e.g. 30)     : "))
        conc   = float(input("Metal conc  (mg/L, e.g. 20)     : "))
        dosage = float(input("Dosage      (g/L, e.g. 0.5)     : "))

        code = 0.0 if metal == 'CU' else 1.0
        inp  = scaler_X.transform([[time, conc, ph, dosage, temp, code]])
        out  = net.predict(inp)
        removal = float(np.clip(
            scaler_Y.inverse_transform(out.reshape(-1,1))[0][0], 0, 100))

        print(f"\n  >>> Predicted Removal Efficiency = {removal:.2f}% <<<\n")

    except ValueError:
        print("  Please enter a valid number.")
    except KeyboardInterrupt:
        print("\n\nExiting. Goodbye!")
        break
