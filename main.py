import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from model_fusion import build_fusion_model
from dataset import load_feature_matrix
import numpy as np

# Load anonymized feature data
clinical, radiomics, deep, y = load_feature_matrix('feature_matrix_example.csv')
y_cat = tf.keras.utils.to_categorical(y, num_classes=4)

# Split data (70% train, 20% val, 10% test)
X = np.concatenate([clinical, radiomics, deep], axis=1)
X_train, X_temp, y_train, y_temp = train_test_split(X, y_cat, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

def split_feats(X):
    return [X[:, :42], X[:, 42:42+76], X[:, 42+76:42+76+40]]

X_train_split = split_feats(X_train)
X_val_split = split_feats(X_val)
X_test_split = split_feats(X_test)

# Build model
model = build_fusion_model(clinical_dim=42, radiomics_dim=76, img_shape=(224,224,3))
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Dummy image input for demonstration (all zeros, shape: n_samples, 224,224,3)
img_dummy_train = np.zeros((X_train_split[2].shape[0],224,224,3), dtype=np.float32)
img_dummy_val = np.zeros((X_val_split[2].shape[0],224,224,3), dtype=np.float32)
img_dummy_test = np.zeros((X_test_split[2].shape[0],224,224,3), dtype=np.float32)

# Train
model.fit(
    {'clinical_input': X_train_split[0], 'radiomics_input': X_train_split[1], 'input_1': img_dummy_train},
    y_train,
    validation_data=(
        {'clinical_input': X_val_split[0], 'radiomics_input': X_val_split[1], 'input_1': img_dummy_val},
        y_val),
    epochs=100,
    batch_size=32,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
)

# Evaluate
y_pred = model.predict({'clinical_input': X_test_split[0], 'radiomics_input': X_test_split[1], 'input_1': img_dummy_test})
print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
print('Macro-AUC:', roc_auc_score(y_test, y_pred, multi_class='ovr', average='macro'))

# Save model weights
model.save_weights('model_weights_fusion.h5')