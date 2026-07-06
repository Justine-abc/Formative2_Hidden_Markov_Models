import os
import numpy as np
import pandas as pd
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

print("🚀 Initializing HMM Pipeline...")

# 1. LOAD AND MERGE TRACKING FILES
base_path = "./" 
folders = {'still': 'still', 'standing': 'standing', 'walking': 'walking', 'jumping': 'jumping'}
processed_data = {}

for activity, folder_name in folders.items():
    folder_path = os.path.join(base_path, folder_name)
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder '{folder_name}' not found. Check structure!")
        continue
    acc_df = pd.read_csv(os.path.join(folder_path, 'Accelerometer.csv'))
    gyro_df = pd.read_csv(os.path.join(folder_path, 'Gyroscope.csv'))
    
    acc_df = acc_df.rename(columns={'x': 'acc_x', 'y': 'acc_y', 'z': 'acc_z'}).sort_values('time')
    gyro_df = gyro_df.rename(columns={'x': 'gyro_x', 'y': 'gyro_y', 'z': 'gyro_z'}).sort_values('time')
    
    merged_df = pd.merge_asof(acc_df, gyro_df[['time', 'gyro_x', 'gyro_y', 'gyro_z']], on='time', direction='nearest')
    processed_data[activity] = merged_df.dropna()
    print(f"✅ Loaded {activity}: {processed_data[activity].shape[0]} rows merged.")

# 2. FEATURE EXTRACTION
def extract_window_features(df, window_size=100, step_size=50):
    window_features = []
    for start in range(0, len(df) - window_size, step_size):
        end = start + window_size
        window = df.iloc[start:end]
        feature_dict = {}
        
        for col in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']:
            feature_dict[f'{col}_mean'] = np.mean(window[col])
            feature_dict[f'{col}_var'] = np.var(window[col])
            feature_dict[f'{col}_rms'] = np.sqrt(np.mean(window[col]**2))
            
        for col in ['acc_x', 'acc_y', 'acc_z']:
            fft_values = np.abs(fft(window[col].values))
            feature_dict[f'{col}_dom_freq'] = np.argmax(fft_values[1:len(fft_values)//2]) + 1
            feature_dict[f'{col}_energy'] = np.sum(fft_values**2) / len(fft_values)
            
        window_features.append(feature_dict)
    return pd.DataFrame(window_features)

X_list, y_list = [], []
state_mapping = {'still': 0, 'standing': 1, 'walking': 2, 'jumping': 3}
lengths = []

for activity, df in processed_data.items():
    features_df = extract_window_features(df)
    X_list.append(features_df)
    y_list.append(np.full(len(features_df), state_mapping[activity]))
    lengths.append(len(features_df))

X_raw = pd.concat(X_list, axis=0).values
y_true = np.concatenate(y_list)
X = StandardScaler().fit_transform(X_raw)

print(f"📊 Features extracted. Matrix size: {X.shape}")

# 3. HMM MODEL IMPLEMENTATION & TRAINING
print("⚙️ Training Hidden Markov Model (Baum-Welch optimization)...")
model = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=100, tol=0.01, random_state=42)
model.fit(X, lengths=lengths)
print("🎯 HMM training complete!")

# 4. DECODING SEQUENCES (Viterbi)
print("🔮 Decoding sequence states using Viterbi...")
y_pred = model.predict(X, lengths=lengths)

# 5. VISUALIZATIONS & METRICS PRODUCTION
# Plot 1: Transition Probability Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(model.transmat_, annot=True, cmap='Blues', xticklabels=state_mapping.keys(), yticklabels=state_mapping.keys())
plt.title('HMM State Transition Matrix')
plt.ylabel('Current State')
plt.xlabel('Predicted Next State')
plt.tight_layout()
plt.savefig('transition_heatmap.png')
print("💾 Saved: transition_heatmap.png")

# Plot 2: Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=state_mapping.keys(), yticklabels=state_mapping.keys())
plt.title('Activity Classification Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("💾 Saved: confusion_matrix.png")

# Display numerical performance summary metrics
print("\n📋 MODEL PERFORMANCE SUMMARY REPORT:")
print(classification_report(y_true, y_pred, target_names=state_mapping.keys()))