#!/usr/bin/env python3
# train_bilstm.py
import argparse, os, json, time
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout, Layer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

class Attention(Layer):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)
    def call(self, hidden_states):
        query = hidden_states[:, -1, :]
        query_with_time = tf.expand_dims(query, 1)
        score = tf.nn.tanh(self.W1(hidden_states) + self.W2(query_with_time))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * hidden_states
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

def build_model(seq_len, n_feats):
    inp = Input(shape=(seq_len, n_feats))
    x = Bidirectional(LSTM(128, return_sequences=True))(inp)
    context = Attention(64)(x)
    x = Dense(64, activation='relu')(context)
    x = Dropout(0.2)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')])
    return model

def load_and_build_sequences(feat_file, features, seq_len):
    df = pd.read_parquet(feat_file)
    df = df.dropna(subset=features + ['pit_next'])
    seqs = []
    targets = []
    for (driver, stint), g in df.groupby(['Driver','Stint']):
        g = g.sort_values('LapNumber')
        vals = g[features].values
        t = g['pit_next'].values
        for i in range(len(vals)-1):
            start = max(0, i-seq_len+1)
            seq = vals[start:i+1]
            if seq.shape[0] < seq_len:
                pad = np.zeros((seq_len - seq.shape[0], seq.shape[1]))
                seq = np.vstack([pad, seq])
            seqs.append(seq)
            targets.append(t[i+1])
    X = np.array(seqs)
    y = np.array(targets).astype(int)
    return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, required=True)
    parser.add_argument('--seq_len', type=int, default=40)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--outdir', type=str, default='models/bilstm')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    logdir = os.path.join('logs', 'bilstm_' + time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(logdir, exist_ok=True)

    features = ['lap_sec','tyre_age','lap_delta','rolling_mean_3']
    X, y = load_and_build_sequences(args.features, features, args.seq_len)
    print('X shape, y shape:', X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # scale
    n_feats = X.shape[2]
    for i in range(n_feats):
        s = StandardScaler()
        s.fit(X_train[:,:,i].reshape(-1,1))
        X_train[:,:,i] = s.transform(X_train[:,:,i].reshape(-1,1)).reshape(-1, args.seq_len)
        X_test[:,:,i] = s.transform(X_test[:,:,i].reshape(-1,1)).reshape(-1, args.seq_len)

    model = build_model(args.seq_len, n_feats)
    es = EarlyStopping(patience=6, restore_best_weights=True, monitor='val_auc', mode='max')
    ckpt = ModelCheckpoint(os.path.join(args.outdir, 'bilstm_best.h5'), save_best_only=True, monitor='val_auc', mode='max')
    tb = TensorBoard(log_dir=logdir)

    history = model.fit(X_train, y_train, validation_split=0.1, epochs=args.epochs, batch_size=args.batch, callbacks=[es,ckpt,tb])
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob > 0.5).astype(int)
    auc = roc_auc_score(y_test, y_prob)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    metrics = {'auc': float(auc), 'precision': float(precision), 'recall': float(recall), 'f1': float(f1)}
    with open(os.path.join(args.outdir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print('Training complete. Metrics saved to', os.path.join(args.outdir, 'metrics.json'))

if __name__ == '__main__':
    main()
