import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ----------------- PATHS -----------------
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "train")  # train/Closed_Eyes, train/Open_Eyes
MODEL_OUT_PATH = os.path.join(SCRIPT_DIR, "drowsiness_model.h5")

IMG_HEIGHT = 24
IMG_WIDTH = 24
BATCH_SIZE = 32
EPOCHS = 10  # later you can increase

if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(
        f"Training folder not found at {DATA_DIR}.\n"
        "You must have:\n"
        "  train/Closed_Eyes\n"
        "  train/Open_Eyes\n"
        "with images inside."
    )

# ----------------- DATA GENERATORS -----------------
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
)

print("Class indices:", train_gen.class_indices)
# should be {'Closed_Eyes': 0, 'Open_Eyes': 1}

# ----------------- CNN MODEL -----------------
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),  # 0 = Closed_Eyes, 1 = Open_Eyes
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
)

model.save(MODEL_OUT_PATH)
print(f"Training complete. Model saved to: {MODEL_OUT_PATH}")

