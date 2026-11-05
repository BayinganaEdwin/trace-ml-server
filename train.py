"""
SiteLog Construction Stage Classifier — MobileNetV2 Transfer Learning
=====================================================================
4 classes:
    0 → formwork_erected
    1 → reinforcement_placed
    2 → concrete_poured
    3 → structure_complete

Usage
-----
1. Put images in  data/<class_name>/*.jpg  (see prepare_dataset.py to download)
2. pip install tensorflow pillow scikit-learn matplotlib
3. python train.py

The script:
  • Splits data 80/10/10 (train/val/test) with stratification
  • Loads MobileNetV2 (ImageNet weights), freezes base, trains top
  • Phase 2: unfreezes top-30 layers for fine-tuning
  • Saves best model to  checkpoints/best_model.keras
  • Exports TFLite model to  checkpoints/model.tflite  (for on-device use)
  • Prints per-class test accuracy + confusion matrix
"""

import os
import shutil
import random
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
SEED          = 42
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
PHASE1_EPOCHS = 20    # train top layers only
PHASE2_EPOCHS = 30    # fine-tune top-50 layers of base

CLASS_NAMES = [
    "formwork_erected",
    "reinforcement_placed",
    "concrete_poured",
    "structure_complete",
    "not_construction",   # negative examples — flowers, animals, food, etc.
]

DATA_DIR        = pathlib.Path("data")
CHECKPOINT_DIR  = pathlib.Path("checkpoints")
LOG_DIR         = pathlib.Path("logs")
SPLIT_DIR       = pathlib.Path("data_split")   # temp split used during training

CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ── 1. Verify data ─────────────────────────────────────────────────────────
def count_images():
    counts = {}
    for cls in CLASS_NAMES:
        cls_dir = DATA_DIR / cls
        if not cls_dir.exists():
            counts[cls] = 0
            continue
        imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg")) + list(cls_dir.glob("*.png"))
        counts[cls] = len(imgs)
    return counts


def verify_data():
    counts = count_images()
    print("\n── Dataset summary ──")
    total = 0
    for cls, n in counts.items():
        print(f"  {cls}: {n} images")
        total += n
    print(f"  Total: {total}")
    if total == 0:
        raise RuntimeError(
            "No images found. Run  python prepare_dataset.py  first, or "
            "manually place images in  data/<class_name>/  folders."
        )
    if any(v == 0 for v in counts.values()):
        missing = [k for k, v in counts.items() if v == 0]
        print(f"\n  WARNING: Missing classes: {missing}")
        print("  Training will proceed but the model won't recognise these stages.")
    return counts


# ── 2. Build stratified train/val/test split ───────────────────────────────
def build_split():
    """Copy images into data_split/{train,val,test}/<class>/ for ImageDataGenerator."""
    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)

    all_paths, all_labels = [], []
    for idx, cls in enumerate(CLASS_NAMES):
        cls_dir = DATA_DIR / cls
        if not cls_dir.exists():
            continue
        imgs = (
            list(cls_dir.glob("*.jpg"))
            + list(cls_dir.glob("*.jpeg"))
            + list(cls_dir.glob("*.png"))
        )
        all_paths.extend(imgs)
        all_labels.extend([idx] * len(imgs))

    # stratified split: 80 / 10 / 10
    paths_train, paths_tmp, y_train, y_tmp = train_test_split(
        all_paths, all_labels, test_size=0.20, stratify=all_labels, random_state=SEED
    )
    paths_val, paths_test, y_val, y_test = train_test_split(
        paths_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED
    )

    def copy_split(paths, labels, split_name):
        for p, lbl in zip(paths, labels):
            dst_dir = SPLIT_DIR / split_name / CLASS_NAMES[lbl]
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst_dir / p.name)

    copy_split(paths_train, y_train, "train")
    copy_split(paths_val,   y_val,   "val")
    copy_split(paths_test,  y_test,  "test")

    print(f"\n── Split: train={len(paths_train)}, val={len(paths_val)}, test={len(paths_test)}")
    return len(paths_train), len(paths_val), len(paths_test)


# ── 3. Data generators ─────────────────────────────────────────────────────
def make_generators():
    train_aug = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.20,
        height_shift_range=0.20,
        shear_range=0.15,
        zoom_range=0.25,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.7, 1.3],
        channel_shift_range=20.0,
        fill_mode="nearest",
    )
    val_aug = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_aug.flow_from_directory(
        SPLIT_DIR / "train",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        seed=SEED,
    )
    val_gen = val_aug.flow_from_directory(
        SPLIT_DIR / "val",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        seed=SEED,
    )
    test_gen = val_aug.flow_from_directory(
        SPLIT_DIR / "test",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )
    return train_gen, val_gen, test_gen


# ── 4. Model ───────────────────────────────────────────────────────────────
def build_model(num_classes=len(CLASS_NAMES)):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # freeze for phase 1

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base


# ── 5. Training ────────────────────────────────────────────────────────────
def get_callbacks(phase):
    return [
        callbacks.ModelCheckpoint(
            str(CHECKPOINT_DIR / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.TensorBoard(
            log_dir=str(LOG_DIR / f"phase{phase}"),
            histogram_freq=0,
        ),
    ]


def plot_history(hist1, hist2=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for key, ax in zip(["accuracy", "loss"], axes):
        ax.plot(hist1.history[key], label=f"Phase1 train {key}")
        ax.plot(hist1.history[f"val_{key}"], label=f"Phase1 val {key}")
        if hist2 is not None:
            offset = len(hist1.history[key])
            ax.plot(range(offset, offset + len(hist2.history[key])),
                    hist2.history[key], label=f"Phase2 train {key}")
            ax.plot(range(offset, offset + len(hist2.history[f"val_{key}"])),
                    hist2.history[f"val_{key}"], label=f"Phase2 val {key}")
        ax.set_title(key.capitalize())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(LOG_DIR / "training_history.png"), dpi=120)
    print(f"\n  Plot saved to {LOG_DIR / 'training_history.png'}")


# ── 6. Evaluation ─────────────────────────────────────────────────────────
def evaluate(model, test_gen, n_test):
    print("\n── Test set evaluation ──")
    results = model.evaluate(test_gen, verbose=1)
    print(f"  Test loss: {results[0]:.4f}  |  Test accuracy: {results[1]*100:.2f}%")

    test_gen.reset()
    preds = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes

    print("\n── Per-class report ──")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    print("── Confusion matrix ──")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    return results[1]


# ── 7. Export TFLite ──────────────────────────────────────────────────────
def export_tflite():
    """Convert saved .keras model → float16 TFLite for React Native on-device use."""
    print("\n── Exporting TFLite ──")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(CHECKPOINT_DIR / "best_model.keras"))
    # float16 quantisation: ~2× smaller, negligible accuracy loss
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    tflite_path = CHECKPOINT_DIR / "model.tflite"
    tflite_path.write_bytes(tflite_model)
    size_mb = tflite_path.stat().st_size / 1_000_000
    print(f"  Saved {tflite_path}  ({size_mb:.1f} MB)")

    # Save class names alongside the model
    labels_path = CHECKPOINT_DIR / "labels.txt"
    labels_path.write_text("\n".join(CLASS_NAMES))
    print(f"  Saved {labels_path}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("SiteLog — MobileNetV2 Construction Stage Classifier")
    print("=" * 60)

    # 1. Verify + split
    verify_data()
    n_train, n_val, n_test = build_split()

    # 2. Generators
    train_gen, val_gen, test_gen = make_generators()

    # 3. Build model
    model, base = build_model()
    model.summary()

    # ── Phase 1: train top layers only ──────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"PHASE 1 — Training top layers  ({PHASE1_EPOCHS} epochs max)")
    print(f"{'─'*50}")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    hist1 = model.fit(
        train_gen,
        steps_per_epoch=max(1, n_train // BATCH_SIZE),
        validation_data=val_gen,
        validation_steps=max(1, n_val // BATCH_SIZE),
        epochs=PHASE1_EPOCHS,
        callbacks=get_callbacks(1),
    )

    # ── Phase 2: fine-tune top-30 layers of base ─────────────────────────
    print(f"\n{'─'*50}")
    print(f"PHASE 2 — Fine-tuning top 30 base layers  ({PHASE2_EPOCHS} epochs max)")
    print(f"{'─'*50}")
    base.trainable = True
    for layer in base.layers[:-50]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    hist2 = model.fit(
        train_gen,
        steps_per_epoch=max(1, n_train // BATCH_SIZE),
        validation_data=val_gen,
        validation_steps=max(1, n_val // BATCH_SIZE),
        epochs=PHASE2_EPOCHS,
        callbacks=get_callbacks(2),
    )

    # ── Evaluate ─────────────────────────────────────────────────────────
    # Load best checkpoint before evaluating
    best = tf.keras.models.load_model(str(CHECKPOINT_DIR / "best_model.keras"))
    acc = evaluate(best, test_gen, n_test)

    # ── Plots + TFLite ────────────────────────────────────────────────────
    plot_history(hist1, hist2)
    try:
        export_tflite()
    except Exception as e:
        print(f"\n  TFLite export skipped: {e}")
        print("  (Run on a machine with full TensorFlow to export TFLite.)")

    print(f"\n{'='*60}")
    print(f"Training complete.  Best test accuracy: {acc*100:.2f}%")
    print(f"Model saved to: {CHECKPOINT_DIR / 'best_model.keras'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

# Accuracy targets by training phase:
# Phase 1 (frozen base):   val_accuracy >= 0.80 within 10 epochs
# Phase 2 (fine-tuned):    val_accuracy >= 0.90 within 20 epochs
# If targets not met, review dataset balance and augmentation parameters.
