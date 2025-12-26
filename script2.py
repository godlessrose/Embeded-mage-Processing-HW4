import os
import numpy as np
import cv2
import struct
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

# --- MNIST Veri Yükleme (Q1 ile aynı) ---
def load_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)

def load_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        return np.fromfile(f, dtype=np.uint8)

# --- Yolları Tanımla ---
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
train_huMoments = np.empty((len(train_images), 7))
test_huMoments = np.empty((len(test_images), 7))

print("Q2: 10 Sınıf için Hu Moments çıkarılıyor...")
for i, img in enumerate(train_images):
    m = cv2.moments(img, True)
    train_huMoments[i] = cv2.HuMoments(m).reshape(7)

for i, img in enumerate(test_images):
    m = cv2.moments(img, True)
    test_huMoments[i] = cv2.HuMoments(m).reshape(7)

# Standardizasyon
f_mean = np.mean(train_huMoments, axis=0)
f_std = np.std(train_huMoments, axis=0)
train_huMoments = (train_huMoments - f_mean) / f_std
test_huMoments = (test_huMoments - f_mean) / f_std

# --- Çok Katmanlı Model (MLP) Yapısı ---
# Q1'den farkı: Gizli katmanlar ekleniyor ve çıkış 10 nöron oluyor
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_shape=[7], activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax') 
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- Eğitim ---
# Burada artık label'ları binarize etmiyoruz (0-9 arası kalıyorlar)
print("Q2: Model eğitiliyor (10 Rakam)...")
model.fit(train_huMoments, train_labels, epochs=50, batch_size=128, verbose=1)

# --- Değerlendirme ---
preds = np.argmax(model.predict(test_huMoments), axis=1)
conf_matrix = confusion_matrix(test_labels, preds)

# Çizim
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=range(10))
disp.plot()
plt.title("MLP (10 Classes) Confusion Matrix")
plt.show()

model.save("mnist_mlp_model.h5")
print("Q2 Modeli 'mnist_mlp_model.h5' olarak kaydedildi.")