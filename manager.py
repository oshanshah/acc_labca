import matplotlib
matplotlib.use('Agg')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("=== 1. Generating workload ===")
np.random.seed(42)
N = 7*24*60; t = np.arange(N)
load = 400 + 250*np.sin(2*np.pi*t/1440 - np.pi/2) + np.random.normal(0, 20, N)
for s in np.random.choice(np.arange(500, N-100), 12, replace=False):
    load[s:s+15] += np.random.uniform(300, 500)
load = np.clip(load, 10, None)
plt.figure(figsize=(12,3)); plt.plot(load[:2880]); plt.title("Workload (first 2 days)")
plt.xlabel("minute"); plt.ylabel("requests/min"); plt.tight_layout(); plt.savefig('fig1_workload.png', dpi=120); plt.close()

print("=== 2. Feature engineering ===")
df = pd.DataFrame({'load': load})
for l in [1, 5, 10, 15, 30]: df[f'lag{l}'] = df.load.shift(l)
df['rm10'] = df.load.rolling(10).mean(); df['rs10'] = df.load.rolling(10).std()
df['d1'] = df.load - df.lag1
df['hs'] = np.sin(2*np.pi*t/1440); df['hc'] = np.cos(2*np.pi*t/1440)
H = 10
df['y'] = df.load.shift(-H)
df = df.dropna().reset_index(drop=True)
load = df.load.values; n = len(df); split = int(0.7*n)
feats = [c for c in df.columns if c != 'y']
print("Features:", feats); print("Train rows:", split, "Test rows:", n-split)

print("\n=== 3. Forecaster (Random Forest, 10-min horizon) ===")
rf = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=0)
rf.fit(df[feats][:split], df.y[:split])
pred = rf.predict(df[feats])
yt, yp = df.y[split:], pred[split:]
mae = mean_absolute_error(yt, yp); rmse = np.sqrt(mean_squared_error(yt, yp))
mape = np.mean(np.abs(yt-yp)/yt)*100
naive = mean_absolute_error(yt, df.load[split:])
print(f"MAE={mae:.1f}  RMSE={rmse:.1f}  MAPE={mape:.1f}%  | Naive baseline MAE={naive:.1f}")

print("\n=== 4. Anomaly detector (Isolation Forest) ===")
af = ['load', 'rs10', 'd1']
iso = IsolationForest(contamination=0.02, random_state=0).fit(df[af][:split])
anom = iso.predict(df[af]) == -1
print("Anomalies flagged in test period:", anom[split:].sum(), "of", n-split, "minutes")

print("\n=== 5. Auto-scaling simulation: Reactive vs ML-based ===")
CAP, TU, DELAY = 100, 0.7, 5
def simulate(mode, p=None):
    p = pred if p is None else p
    dec = np.zeros(n, int); active = np.zeros(n, int)
    dec[:split] = int(np.ceil(load[split]/(CAP*TU)))
    for i in range(split, n):
        need = int(np.ceil(load[i]/(CAP*TU)))
        if mode == 'ml':
            s = np.arange(i-H-30, i-H+1)
            err = np.mean(np.abs(p[s]-load[s+H])/load[s+H])
            buf = np.clip(err, 0.05, 0.30)
            need = max(need, int(np.ceil(p[i]*(1+buf)/(CAP*TU))))
            if anom[i]:
                need = max(need, int(np.ceil(load[i]*1.3/(CAP*TU))))
        dec[i] = max(need, dec[i-1]-1, 2)
        active[i] = dec[i-DELAY]
    a, l = active[split:], load[split:]
    rho = l/(a*CAP); lat = 50/(1-np.minimum(rho, 0.99))
    util = np.minimum(rho, 1)
    return dict(avg_util_pct=util.mean()*100, avg_latency_ms=lat.mean(),
                p95_latency_ms=np.percentile(lat, 95), sla_violation_pct=(lat > 500).mean()*100,
                vm_hours=a.sum()/60, cost_usd=a.sum()/60*0.05,
                energy_kwh=(a*(100+100*util)).sum()/60/1000), a

res_r, a_r = simulate('reactive'); res_m, a_m = simulate('ml')
results = pd.DataFrame({'Reactive': res_r, 'ML-based': res_m}).round(2)
print(results); results.to_csv('results.csv')

print("\n=== 6. Graphs ===")
fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
x = np.arange(split, n)
ax[0].plot(x, load[split:], label='Actual'); ax[0].plot(x, pred[split-H:n-H], label='Predicted', alpha=.7)
ax[0].scatter(x[anom[split:]], load[split:][anom[split:]], c='r', s=8, label='Anomaly')
ax[0].set_ylabel('Requests/min'); ax[0].set_title('Forecast vs actual'); ax[0].legend()
ax[1].plot(x, a_r, label='Reactive'); ax[1].plot(x, a_m, label='ML-based')
ax[1].set_ylabel('Active VMs'); ax[1].set_title('Instances over time'); ax[1].legend()
ax[2].plot(x, load[split:]/(a_r*CAP), label='Reactive'); ax[2].plot(x, load[split:]/(a_m*CAP), label='ML-based')
ax[2].axhline(0.9, c='k', ls='--'); ax[2].set_ylabel('Utilization'); ax[2].set_title('Utilization')
ax[2].legend(); plt.tight_layout(); plt.savefig('fig2_results.png', dpi=120); plt.close()

w = slice(0, 400)
plt.figure(figsize=(12,3)); plt.plot(x[w], a_r[w], label='Reactive'); plt.plot(x[w], a_m[w], label='ML-based')
plt.legend(); plt.title("Zoom: VMs during a spike"); plt.tight_layout(); plt.savefig('fig3_zoom.png', dpi=120); plt.close()

print("\n=== 7. What if the prediction is wrong? ===")
scen = {'Accurate forecast': pred, 'Under-predict (-30%)': pred*0.7, 'Over-predict (+40%)': pred*1.4}
rows = {k: simulate('ml', v)[0] for k, v in scen.items()}
rows['Reactive (no ML)'] = res_r
wrong = pd.DataFrame(rows).round(2); print(wrong); wrong.to_csv('wrong_prediction.csv')
print("\nDONE. Files: fig1_workload.png fig2_results.png fig3_zoom.png results.csv wrong_prediction.csv")

