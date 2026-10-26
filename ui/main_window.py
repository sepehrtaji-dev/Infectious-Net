import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from predict import Predictor

PREVIEW_SIZE = (320, 320)


class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face-infectious - Skin Disease Classifier")
        self.resizable(False, False)
        self.predictor = None
        self._photo = None

        self._build()
        self.after(100, self._load_model)

    def _build(self):
        pad = {"padx": 12, "pady": 6}

        self.preview = tk.Label(self, text="No image selected", width=40, height=18,
                                relief="groove", bg="#f2f2f2")
        self.preview.grid(row=0, column=0, rowspan=2, **pad)

        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="n", **pad)

        self.btn = ttk.Button(right, text="Choose image...", command=self.choose_image,
                              state="disabled")
        self.btn.pack(fill="x", pady=(0, 10))

        self.result_label = ttk.Label(right, text="Loading model...", font=("Segoe UI", 13, "bold"))
        self.result_label.pack(anchor="w", pady=(0, 8))

        self.bars = {}
        self.bar_frame = ttk.Frame(right)
        self.bar_frame.pack(fill="x")

    def _load_model(self):
        try:
            self.predictor = Predictor()
        except Exception as e:  # missing checkpoint etc.
            self.result_label.config(text="Model not loaded")
            messagebox.showerror("Model error", str(e))
            return

        for name in self.predictor.class_names:
            row = ttk.Frame(self.bar_frame)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=name, width=12).pack(side="left")
            bar = ttk.Progressbar(row, length=180, maximum=100)
            bar.pack(side="left", padx=6)
            pct = ttk.Label(row, text="0.0%", width=7)
            pct.pack(side="left")
            self.bars[name] = (bar, pct)

        self.result_label.config(text="Choose an image to start")
        self.btn.config(state="normal")

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")])
        if not path:
            return
        try:
            img = Image.open(path)
            self._show_preview(img)
            results = self.predictor.predict(img)
        except Exception as e:
            messagebox.showerror("Error", f"Could not process image:\n{e}")
            return

        top_name, top_p = results[0]
        self.result_label.config(text=f"{top_name} ({top_p * 100:.1f}%)")
        for name, p in results:
            bar, pct = self.bars[name]
            bar["value"] = p * 100
            pct.config(text=f"{p * 100:.1f}%")

    def _show_preview(self, img):
        thumb = img.convert("RGB")
        thumb.thumbnail(PREVIEW_SIZE)
        self._photo = ImageTk.PhotoImage(thumb)
        self.preview.config(image=self._photo, text="", width=thumb.width, height=thumb.height)


def run():
    MainWindow().mainloop()