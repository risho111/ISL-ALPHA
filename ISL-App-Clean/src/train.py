import json
import shutil
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    X_PATH,
    Y_PATH,
    LABELS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    MODEL_LABELS_PATH,
    SEQUENCE_LENGTH,
    FEATURE_SIZE,
)


def build_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, FEATURE_SIZE)),

        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.LSTM(32),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)

    num_classes = len(labels)

    print("Dataset loaded.")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Labels: {labels}")

    if num_classes < 2:
        print("Training needs at least 2 classes.")
        print("Use yes + idle first.")
        return

    class_counts = {}
    for i, label in enumerate(labels):
        class_counts[label] = int(np.sum(y == i))

    print("\nClass counts:")
    for label, count in class_counts.items():
        print(f"{label}: {count}")

    min_count = min(class_counts.values())

    if min_count < 5:
        print("\nWarning: Very low videos in one or more classes.")
        print("Use at least 30 videos per class for decent training.")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    class_weight_dict = {
        int(class_id): float(weight)
        for class_id, weight in zip(np.unique(y_train), class_weights)
    }

    model = build_model(num_classes)

    print("\nModel summary:")
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
    filepath=str(MODEL_PATH),
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=False
)
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=8,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )

    model = tf.keras.models.load_model(MODEL_PATH)

    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)

    print("\nTraining completed.")
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
    print(f"Validation loss: {val_loss:.4f}")

    predictions = model.predict(X_val)
    predicted_classes = np.argmax(predictions, axis=1)

    print("\nClassification report:")
    print(classification_report(y_val, predicted_classes, target_names=labels))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_val, predicted_classes))

    shutil.copy(LABELS_PATH, MODEL_LABELS_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Labels saved to: {MODEL_LABELS_PATH}")


if __name__ == "__main__":
    main()