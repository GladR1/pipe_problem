import numpy as np
import matplotlib.pyplot as plt


def plot_random_series(train_data, label, n=3):
    ids = train_data[train_data['class'] == label]['id'].unique()
    sample_ids = np.random.choice(ids, size=min(n, len(ids)), replace=False)
    
    plt.figure(figsize=(15, 4 * n))
    for i, sample_id in enumerate(sample_ids):
        sample = train_data[train_data['id'] == sample_id]
        plt.subplot(n, 1, i+1)
        plt.plot(sample['time'], sample['ch0'], label='ch0')
        plt.plot(sample['time'], sample['ch1'], label='ch1')
        plt.plot(sample['time'], sample['ch2'], label='ch2')
        plt.title(f'Пример сигнала для id={sample_id}, класс={label}')
        plt.xlabel('Время')
        plt.ylabel('Амплитуда')
        plt.legend()
    plt.tight_layout()
    plt.show()

def plot_mean_profile(train_data, label,channels):
    ids = train_data[train_data['class'] == label]['id'].unique()
    samples = train_data[train_data['id'].isin(ids)]
    
    plt.figure(figsize=(14, 4))
    for ch in channels:
        mean_profile = samples.groupby('time')[ch].mean()
        plt.plot(mean_profile.index, mean_profile.values, label=f'{ch}')
    plt.title(f'Средний профиль сигналов для класса {label}')
    plt.xlabel('Время')
    plt.ylabel('Среднее значение канала')
    plt.legend()
    plt.show()

def predict_cv(models, X_test):
    preds = []
    for model in models:
        pred = model.predict_proba(X_test)[:, 1]
        preds.append(pred)

    mean_preds = np.mean(preds, axis=0)
    final_preds = (mean_preds > 0.5).astype(int)

    return final_preds

def fourier_transform_features(df, n_freq=50):

    features = []
    ids = df['id'].unique()
    original_series = []
    for cur_id in ids:
        sample = df[df['id'] == cur_id].sort_values('time')
        fft_ch0 = np.abs(np.fft.fft(sample['ch0'].values))[:n_freq]
        fft_ch1 = np.abs(np.fft.fft(sample['ch1'].values))[:n_freq]
        fft_ch2 = np.abs(np.fft.fft(sample['ch2'].values))[:n_freq]
        feature_vector = np.hstack([fft_ch0, fft_ch1, fft_ch2])
        features.append(feature_vector)
        original_series.append(sample[['ch0', 'ch1', 'ch2']].values.flatten())
    features = np.array(features)
    return ids, features, original_series