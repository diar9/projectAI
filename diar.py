from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageOps, ImageTk
from tensorflow.keras.models import load_model

# ============================================================
# SETTINGS
# ============================================================

APP_TITLE = "AI Rock Paper Scissors"

MODEL_FOLDER = "model"
MODEL_FILE = "keras_model.h5"
LABELS_FILE = "labels.txt"

WINDOW_SIZE = "650x800"
PREVIEW_SIZE = (500, 500)
IMAGE_SIZE = (224, 224)

# ============================================================
# COLORS
# ============================================================

BG_COLOR = "#1E1E2E"
CARD_COLOR = "#2A2A3C"
ACCENT_COLOR = "#00C896"
ACCENT_HOVER = "#00A67E"
TEXT_COLOR = "#FFFFFF"
SECONDARY_TEXT = "#CFCFCF"

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / MODEL_FOLDER / MODEL_FILE
LABELS_PATH = BASE_DIR / MODEL_FOLDER / LABELS_FILE

# ============================================================
# LOAD LABELS
# ============================================================

def load_labels():

    labels = []

    with open(LABELS_PATH, "r", encoding="utf-8") as file:

        for line in file:

            label = line.strip()

            if not label:
                continue

            parts = label.split(" ", 1)

            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1])
            else:
                labels.append(label)

    return labels

# ============================================================
# PREPARE IMAGE
# ============================================================

def prepare_image(image_path):

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = Image.open(image_path).convert("RGB")

    image = ImageOps.fit(
        image,
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(image)

    normalized = (image_array.astype(np.float32) / 127.5) - 1

    data[0] = normalized

    return data

# ============================================================
# PREDICTION
# ============================================================

def classify_image(image_path):

    data = prepare_image(image_path)

    prediction = model.predict(data, verbose=0)

    index = int(np.argmax(prediction[0]))

    gesture = class_names[index]

    confidence = float(prediction[0][index])

    return index, gesture, confidence

# ============================================================
# SHOW IMAGE
# ============================================================

def show_image(file_path):

    image = Image.open(file_path).convert("RGB")

    image.thumbnail(PREVIEW_SIZE)

    photo = ImageTk.PhotoImage(image)

    image_label.config(image=photo, text="")

    image_label.image = photo

# ============================================================
# HIGHLIGHT RESULT
# ============================================================

def highlight_choice(index):

    for i, lbl in enumerate(choice_labels):

        if i == index:

            lbl.config(
                bg=ACCENT_COLOR,
                fg="white",
                font=("Arial",13,"bold")
            )

        else:

            lbl.config(
                bg=CARD_COLOR,
                fg=TEXT_COLOR,
                font=("Arial",12)
            )

# ============================================================
# UPLOAD IMAGE
# ============================================================

def upload_image():

    file_path = filedialog.askopenfilename(

        title="Choose Image",

        filetypes=[
            ("Images","*.png;*.jpg;*.jpeg;*.bmp;*.webp"),
            ("All Files","*.*")
        ]
    )

    if not file_path:
        return

    try:

        show_image(file_path)

        index, gesture, confidence = classify_image(file_path)

    except Exception as error:

        messagebox.showerror(
            "Prediction Error",
            str(error)
        )

        return

    result_label.config(

        text=f"Prediction: {gesture}\nConfidence: {confidence*100:.2f}%"

    )

    highlight_choice(index)

# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model(MODEL_PATH, compile=False)

    class_names = load_labels()

except Exception as error:

    messagebox.showerror(
        "Model Error",
        str(error)
    )

    raise

# ============================================================
# GUI
# ============================================================

app = tk.Tk()

app.title(APP_TITLE)

app.geometry(WINDOW_SIZE)

app.configure(bg=BG_COLOR)

# ============================================================
# TITLE
# ============================================================

title = tk.Label(

    app,

    text=APP_TITLE,

    font=("Arial",22,"bold"),

    bg=BG_COLOR,

    fg=TEXT_COLOR

)

title.pack(pady=15)

# ============================================================
# IMAGE
# ============================================================

image_label = tk.Label(

    app,

    text="Upload an image",

    bg=CARD_COLOR,

    fg=SECONDARY_TEXT,

    width=45,

    height=14,

    font=("Arial",14)

)

image_label.pack(pady=20)

# ============================================================
# BUTTON
# ============================================================

button = tk.Button(

    app,

    text="Upload Image",

    command=upload_image,

    bg=ACCENT_COLOR,

    fg="white",

    activebackground=ACCENT_HOVER,

    activeforeground="white",

    relief="flat",

    cursor="hand2",

    font=("Arial",14,"bold"),

    padx=25,

    pady=12

)

button.pack()

# ============================================================
# RESULT
# ============================================================

result_label = tk.Label(

    app,

    text="Prediction will appear here",

    bg=BG_COLOR,

    fg=TEXT_COLOR,

    font=("Arial",16)

)

result_label.pack(pady=20)

# ============================================================
# CLASSES TITLE
# ============================================================

title2 = tk.Label(

    app,

    text="Available Classes",

    bg=BG_COLOR,

    fg=TEXT_COLOR,

    font=("Arial",16,"bold")

)

title2.pack()

# ============================================================
# CLASSES FRAME
# ============================================================

frame = tk.Frame(

    app,

    bg=BG_COLOR

)

frame.pack(pady=10)

choice_labels = []

for gesture in class_names:

    lbl = tk.Label(

        frame,

        text=gesture,

        width=25,

        bg=CARD_COLOR,

        fg=TEXT_COLOR,

        pady=6,

        font=("Arial",12)

    )

    lbl.pack(pady=3)

    choice_labels.append(lbl)

# ============================================================
# START
# ============================================================

app.mainloop()
