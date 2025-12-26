import os
import numpy as np
import cv2
import struct
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from matplotlib import pyplot as plt

# --- MNIST Verilerini Okumak İçin Yardımcı Fonksiyonlar ---
def load_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)

def load_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        return np.fromfile(f, dtype=np.uint8)

# --- Veri Yollarını Tanımlama ---
# 'MNIST-dataset' klasörünün Python dosyanla aynı yerde olduğundan emin ol
train_img_path = os.path.join("MNIST-dataset", "train-images.idx3-ubyte")
train_label_path = os.path.join("MNIST-dataset", "train-labels.idx1-ubyte")
test_img_path = os.path.join("MNIST-dataset", "t10k-images.idx3-ubyte")
test_label_path = os.path.join("MNIST-dataset", "t10k-labels.idx1-ubyte")

# Verileri Yükle
train_images = load_images(train_img_path)
train_labels = load_labels(train_label_path)
test_images = load_images(test_img_path)
test_labels = load_labels(test_label_path)

# --- Hu Moments Öznitelik Çıkarımı ---
# Her resimden 7 tane Hu Moment değeri çıkarıyoruz
train_huMoments = np.empty((len(train_images), 7))
test_huMoments = np.empty((len(test_images), 7))

print("Öznitelikler çıkarılıyor (Hu Moments)...")
for train_idx, train_img in enumerate(train_images):
    train_moments = cv2.moments(train_img, True)
    train_huMoments[train_idx] = cv2.HuMoments(train_moments).reshape(7)

for test_idx, test_img in enumerate(test_images):
    test_moments = cv2.moments(test_img, True)
    test_huMoments[test_idx] = cv2.HuMoments(test_moments).reshape(7)

# --- Standardizasyon (Normalization) ---
# Veriyi ortalama 0, standart sapma 1 olacak şekilde ölçeklendiriyoruz
features_mean = np.mean(train_huMoments, axis=0)
features_std = np.std(train_huMoments, axis=0)
train_huMoments = (train_huMoments - features_mean) / features_std
test_huMoments = (test_huMoments - features_mean) / features_std

# --- Model Kurulumu (Tek Nöron) ---
# Sigmoid aktivasyonlu tek bir Dense katmanı
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=[7], activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

# --- Etiketleri Binarize Etme ---
# 0 rakamı -> 0, diğerleri (1-9) -> 1 ("0 değil") yapılıyor
train_labels[train_labels != 0] = 1
test_labels[test_labels != 0] = 1

# --- Eğitim ---
# class_weight: '0' rakamı az olduğu için ona 8 kat daha fazla önem veriyoruz
model.fit(train_huMoments,
          train_labels,
          batch_size=128,
          epochs=50,
          class_weight={0: 8, 1: 1},
          verbose=1)

# --- Tahmin ve Değerlendirme ---
perceptron_preds = model.predict(test_huMoments)
conf_matrix = confusion_matrix(test_labels, perceptron_preds > 0.5)

# Confusion Matrix Çizimi
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
cm_display.plot()
plt.title("Single Neuron Classifier Confusion Matrix")
plt.show()

# Modeli Kaydet
model.save("mnist_single_neuron.h5")
print("Model 'mnist_single_neuron.h5' olarak kaydedildi.")