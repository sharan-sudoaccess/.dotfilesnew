# ============================================================
# ANN Model for Metal Removal Efficiency Prediction
# Inputs:  Time, Conc, pH, Dosage, Temp, Metal_type
# Output:  Removal Efficiency (%)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ──────────────────────────────────────────────
df_all = pd.read_excel('Metal_Removal_ANN_Dataset.xlsx', sheet_name=3)

# Encode metal type: Cu=0, Cr=1  (so model knows which metal)
df_all['Metal_code'] = (df_all.iloc[:, 6] == 'Cr(VI)').astype(float)

# Build X: Time, Conc, pH, Dosage, Temp, Metal_code
X = df_all[['Time (min)',
            'Conc (mg/L)',
            'pH',
            'Dosage (g/L)',
            'Temp (°C)',
            'Metal_code']].values

Y = df_all['Removal Efficiency (%)'].values

# Drop any bad rows
mask = ~np.isnan(X).any(axis=1) & ~np.isnan(Y)
X, Y = X[mask], Y[mask]

print(f"Dataset loaded: {X.shape[0]} rows, {X.shape[1]} input features")
print(f"Y range: {Y.min():.1f}% to {Y.max():.1f}%")

# ── 2. NORMALIZE ──────────────────────────────────────────────
scaler_X = MinMaxScaler(feature_range=(-1, 1))
scaler_Y = MinMaxScaler(feature_range=(-1, 1))

Xn = scaler_X.fit_transform(X)
Yn = scaler_Y.fit_transform(Y.reshape(-1, 1)).ravel()

# ── 3. SPLIT 70 / 15 / 15 ─────────────────────────────────────
idx = np.arange(len(Xn))
idx_tr, idx_temp = train_test_split(idx, test_size=0.30, random_state=7)
idx_val, idx_te  = train_test_split(idx_temp, test_size=0.50, random_state=7)

Xn_tr,  Yn_tr  = Xn[idx_tr],   Yn[idx_tr]
Xn_val, Yn_val = Xn[idx_val],  Yn[idx_val]
Xn_te,  Yn_te  = Xn[idx_te],   Yn[idx_te]

print(f"Train: {len(Xn_tr)} | Val: {len(Xn_val)} | Test: {len(Xn_te)}")

# ── 4. TRAIN BEST MODEL ───────────────────────────────────────
print("\nTesting neuron counts...")
best_r2, best_n = -99, 10
for n in [5, 8, 10, 12, 15, 20]:
    m = MLPRegressor(hidden_layer_sizes=(n,), activation='tanh',
                     solver='lbfgs', max_iter=3000, random_state=7, tol=1e-8)
    m.fit(Xn_tr, Yn_tr)
    r2 = r2_score(Yn_val, m.predict(Xn_val))
    print(f"  Neurons={n:2d}  ->  Val R2={r2:.4f}")
    if r2 > best_r2:
        best_r2, best_n = r2, n

print(f"\nBest neurons: {best_n}  (Val R2={best_r2:.4f})")

net = MLPRegressor(hidden_layer_sizes=(best_n,), activation='tanh',
                   solver='lbfgs', max_iter=5000, random_state=7, tol=1e-9)
net.fit(Xn_tr, Yn_tr)

# ── 5. PREDICT ────────────────────────────────────────────────
Yn_pred_all = net.predict(Xn)
Y_pred = scaler_Y.inverse_transform(Yn_pred_all.reshape(-1,1)).ravel()

Yn_pred_tr  = net.predict(Xn_tr)
Yn_pred_val = net.predict(Xn_val)
Yn_pred_te  = net.predict(Xn_te)

Y_tr  = scaler_Y.inverse_transform(Yn_tr.reshape(-1,1)).ravel()
Y_val = scaler_Y.inverse_transform(Yn_val.reshape(-1,1)).ravel()
Y_te  = scaler_Y.inverse_transform(Yn_te.reshape(-1,1)).ravel()
Yp_tr  = scaler_Y.inverse_transform(Yn_pred_tr.reshape(-1,1)).ravel()
Yp_val = scaler_Y.inverse_transform(Yn_pred_val.reshape(-1,1)).ravel()
Yp_te  = scaler_Y.inverse_transform(Yn_pred_te.reshape(-1,1)).ravel()

# ── 6. METRICS ────────────────────────────────────────────────
R2   = r2_score(Y, Y_pred)
MSE  = mean_squared_error(Y, Y_pred)
RMSE = np.sqrt(MSE)
MAE  = np.mean(np.abs(Y - Y_pred))
R2_tr  = r2_score(Y_tr,  Yp_tr)
R2_val = r2_score(Y_val, Yp_val)
R2_te  = r2_score(Y_te,  Yp_te)

print(f"\n{'='*45}")
print(f"  Overall R2  = {R2:.4f}")
print(f"  MSE         = {MSE:.4f}")
print(f"  RMSE        = {RMSE:.4f}")
print(f"  MAE         = {MAE:.4f}")
print(f"  Train R2    = {R2_tr:.4f}")
print(f"  Val   R2    = {R2_val:.4f}")
print(f"  Test  R2    = {R2_te:.4f}")
print(f"{'='*45}")
if   R2 >= 0.95: print("  Excellent model (R2 > 0.95)")
elif R2 >= 0.90: print("  Very good model (R2 > 0.90)")
elif R2 >= 0.80: print("  Good model (R2 > 0.80)")
else:            print("  Acceptable")

# ── 7. PLOT 1: Performance (loss vs iterations) ───────────────
tr_err, val_err = [], []
iters = list(range(100, 5001, 100))
for n in iters:
    m = MLPRegressor(hidden_layer_sizes=(best_n,), activation='tanh',
                     solver='lbfgs', max_iter=n, random_state=7, tol=0)
    m.fit(Xn_tr, Yn_tr)
    tr_err.append(mean_squared_error(Yn_tr,  m.predict(Xn_tr)))
    val_err.append(mean_squared_error(Yn_val, m.predict(Xn_val)))

plt.figure(figsize=(7,4))
plt.semilogy(iters, tr_err,  label='Train',      color='royalblue',  lw=2)
plt.semilogy(iters, val_err, label='Validation', color='darkorange', lw=2)
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('MSE (log scale)', fontsize=12)
plt.title('ANN Training Performance', fontsize=13, fontweight='bold')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('01_performance.png', dpi=150); plt.show()
print("Saved: 01_performance.png")

# ── 8. PLOT 2: Error Histogram ────────────────────────────────
err_norm = Yn - Yn_pred_all
plt.figure(figsize=(7,4))
plt.hist(err_norm, bins=20, color='steelblue', edgecolor='white', alpha=0.85)
plt.axvline(0, color='red', linestyle='--', lw=1.5, label='Zero error')
plt.xlabel('Error (Normalized)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Error Histogram', fontsize=13, fontweight='bold')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('02_error_histogram.png', dpi=150); plt.show()
print("Saved: 02_error_histogram.png")

# ── 9. PLOT 3: Regression (Train / Val / Test) ────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
sets = [
    (Yn_tr,  Yn_pred_tr,  f'Training\nR2={R2_tr:.4f}',   'royalblue'),
    (Yn_val, Yn_pred_val, f'Validation\nR2={R2_val:.4f}', 'darkorange'),
    (Yn_te,  Yn_pred_te,  f'Testing\nR2={R2_te:.4f}',     'forestgreen'),
]
for ax, (yt, yp, lbl, col) in zip(axes, sets):
    ax.scatter(yt, yp, color=col, alpha=0.75, s=55, edgecolors='white')
    lo, hi = min(yt.min(),yp.min()), max(yt.max(),yp.max())
    ax.plot([lo,hi],[lo,hi],'k--',lw=1.5)
    ax.set_xlabel('Target (Normalized)', fontsize=10)
    ax.set_ylabel('Output (Normalized)', fontsize=10)
    ax.set_title(lbl, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
fig.suptitle('Regression Plot - ANN Model', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('03_regression.png', dpi=150); plt.show()
print("Saved: 03_regression.png")

# ── 10. PLOT 4: Experimental vs Predicted ─────────────────────
plt.figure(figsize=(6,6))
plt.scatter(Y, Y_pred, s=70, color='steelblue',
            edgecolors='navy', alpha=0.8, label='Data points')
lo = min(Y.min(), Y_pred.min()) - 2
hi = max(Y.max(), Y_pred.max()) + 2
plt.plot([lo,hi],[lo,hi],'r--',lw=2,label='Perfect fit (y=x)')
plt.xlabel('Experimental Removal (%)', fontsize=12)
plt.ylabel('Predicted Removal (%)', fontsize=12)
plt.title(f'Experimental vs Predicted\nR2 = {R2:.4f}', fontsize=13, fontweight='bold')
plt.legend(fontsize=10); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('04_exp_vs_pred.png', dpi=150); plt.show()
print("Saved: 04_exp_vs_pred.png")

# ── 11. PLOT 5: Feature Importance ────────────────────────────
from sklearn.inspection import permutation_importance
res = permutation_importance(net, Xn, Yn, n_repeats=20, random_state=7)
feat_names = ['Time', 'Conc', 'pH', 'Dosage', 'Temp', 'Metal']
imp = res.importances_mean

plt.figure(figsize=(8,4))
colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
bars = plt.bar(feat_names, imp, color=colors, edgecolor='white')
plt.xlabel('Input Feature', fontsize=12)
plt.ylabel('Importance Score', fontsize=12)
plt.title('Feature Importance\n(Effect on Removal Efficiency)', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, imp):
    plt.text(bar.get_x()+bar.get_width()/2,
             bar.get_height()+0.0005,
             f'{val:.4f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('05_feature_importance.png', dpi=150); plt.show()
print("Saved: 05_feature_importance.png")

# ── 12. SAVE PREDICTIONS ──────────────────────────────────────
pd.DataFrame({
    'Experimental_Removal_%': np.round(Y, 2),
    'Predicted_Removal_%':    np.round(Y_pred, 2),
    'Error_%':                np.round(Y - Y_pred, 2)
}).to_excel('ANN_Predictions.xlsx', index=False)
print("Saved: ANN_Predictions.xlsx")

# ── 13. PREDICT FUNCTION ──────────────────────────────────────
def predict_removal(time, conc, ph, dosage, temp, metal='Cu'):
    """metal = 'Cu' or 'Cr' """
    code = 0.0 if metal == 'Cu' else 1.0
    inp = scaler_X.transform([[time, conc, ph, dosage, temp, code]])
    out = net.predict(inp)
    val = float(np.clip(scaler_Y.inverse_transform(out.reshape(-1,1))[0][0], 0, 100))
    print(f"[{metal}] Predicted Removal = {val:.2f}%")
    return val

print("\n--- Example Predictions ---")
predict_removal(30, 20, 7, 0.5, 35, metal='Cu')
predict_removal(30, 20, 2, 0.5, 35, metal='Cr')

print("\nANN Model Training Completed Successfully")

# ── SAVE MODEL FOR PREDICTOR ──────────────────────────────────
import joblib
joblib.dump({'net': net, 'scaler_X': scaler_X, 'scaler_Y': scaler_Y}, 'model.pkl')
print('Model saved: model.pkl')
