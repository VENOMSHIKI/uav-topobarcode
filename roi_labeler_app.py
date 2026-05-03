import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}

CLASSES = [
    "bridge",
    "river",
    "street",
    "roof",
    "city_block",
]


class RoiLabelerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ROI Labeler — TheWorld50")
        self.root.geometry("1200x800")

        self.source_dir = tk.StringVar(value="data/raw_web/TheWorld_bridge")
        self.output_root = tk.StringVar(value="data/reference_TheWorld50")
        self.current_class = tk.StringVar(value="bridge")
        self.output_size = tk.IntVar(value=128)
        self.force_square = tk.BooleanVar(value=True)

        self.image_paths = []
        self.current_index = 0

        self.original_image = None
        self.display_image = None
        self.tk_image = None

        self.scale = 1.0
        self.image_on_canvas_id = None

        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.current_rect = None

        self.build_ui()
        self.refresh_file_list()

    def build_ui(self):
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        tk.Label(top, text="Source folder:").grid(row=0, column=0, sticky="w")
        tk.Entry(top, textvariable=self.source_dir, width=55).grid(row=0, column=1, padx=4)
        tk.Button(top, text="Choose", command=self.choose_source_dir).grid(row=0, column=2, padx=4)

        tk.Label(top, text="Output root:").grid(row=1, column=0, sticky="w")
        tk.Entry(top, textvariable=self.output_root, width=55).grid(row=1, column=1, padx=4)
        tk.Button(top, text="Choose", command=self.choose_output_root).grid(row=1, column=2, padx=4)

        tk.Label(top, text="Class:").grid(row=0, column=3, sticky="w", padx=(20, 4))
        class_menu = tk.OptionMenu(top, self.current_class, *CLASSES)
        class_menu.config(width=12)
        class_menu.grid(row=0, column=4, padx=4)

        tk.Label(top, text="ROI size:").grid(row=1, column=3, sticky="w", padx=(20, 4))
        size_menu = tk.OptionMenu(top, self.output_size, 128, 160, 192, 224, 256)
        size_menu.config(width=12)
        size_menu.grid(row=1, column=4, padx=4)

        tk.Checkbutton(
            top,
            text="Force square crop",
            variable=self.force_square
        ).grid(row=0, column=5, padx=12)

        tk.Button(top, text="Refresh", command=self.refresh_file_list).grid(row=1, column=5, padx=4)

        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(main, width=330)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=6)

        tk.Label(left, text="Images").pack(anchor="w")

        self.listbox = tk.Listbox(left, width=45, height=30)
        self.listbox.pack(fill=tk.Y, expand=True)

        self.listbox.bind("<<ListboxSelect>>", self.on_list_select)

        nav = tk.Frame(left)
        nav.pack(fill=tk.X, pady=6)

        tk.Button(nav, text="Previous", command=self.prev_image).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(nav, text="Next", command=self.next_image).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        tk.Button(left, text="Save ROI", command=self.save_roi, height=2).pack(fill=tk.X, pady=4)
        tk.Button(left, text="Clear selection", command=self.clear_selection).pack(fill=tk.X, pady=4)
        tk.Button(left, text="Delete last ROI in class", command=self.delete_last_roi).pack(fill=tk.X, pady=4)

        self.info_label = tk.Label(left, text="", justify=tk.LEFT, anchor="w")
        self.info_label.pack(fill=tk.X, pady=8)

        canvas_frame = tk.Frame(main)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=6)

        self.canvas = tk.Canvas(canvas_frame, bg="#222222", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        bottom = tk.Label(
            self.root,
            text="ЛКМ: выделить ROI | Save ROI: сохранить | Next: следующее изображение",
            anchor="w"
        )
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=4)

    def choose_source_dir(self):
        folder = filedialog.askdirectory(initialdir=os.getcwd())
        if folder:
            self.source_dir.set(folder)
            self.refresh_file_list()

    def choose_output_root(self):
        folder = filedialog.askdirectory(initialdir=os.getcwd())
        if folder:
            self.output_root.set(folder)

    def refresh_file_list(self):
        folder = self.source_dir.get()

        if not os.path.isdir(folder):
            self.image_paths = []
            self.listbox.delete(0, tk.END)
            self.set_info(f"Папка не найдена:\n{folder}")
            return

        self.image_paths = self.find_images(folder)
        self.listbox.delete(0, tk.END)

        for path in self.image_paths:
            self.listbox.insert(tk.END, os.path.basename(path))

        if self.image_paths:
            self.current_index = 0
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(0)
            self.load_current_image()
        else:
            self.clear_canvas()
            self.set_info("В папке нет изображений.")

    def find_images(self, folder):
        result = []
        for root, _, files in os.walk(folder):
            for name in sorted(files):
                ext = Path(name).suffix.lower()
                if ext in IMAGE_EXTENSIONS:
                    result.append(os.path.join(root, name))
        return result

    def on_list_select(self, event):
        selection = self.listbox.curselection()
        if not selection:
            return

        self.current_index = selection[0]
        self.load_current_image()

    def prev_image(self):
        if not self.image_paths:
            return

        self.current_index = max(0, self.current_index - 1)
        self.select_listbox_index(self.current_index)
        self.load_current_image()

    def next_image(self):
        if not self.image_paths:
            return

        self.current_index = min(len(self.image_paths) - 1, self.current_index + 1)
        self.select_listbox_index(self.current_index)
        self.load_current_image()

    def select_listbox_index(self, index):
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(index)
        self.listbox.see(index)

    def load_current_image(self):
        if not self.image_paths:
            return

        path = self.image_paths[self.current_index]

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть изображение:\n{path}\n\n{e}")
            return

        self.original_image = img
        self.current_rect = None
        self.clear_selection()

        self.display_image, self.scale = self.make_display_image(img)
        self.tk_image = ImageTk.PhotoImage(self.display_image)

        self.clear_canvas()
        self.image_on_canvas_id = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        self.set_info(
            f"Image {self.current_index + 1} / {len(self.image_paths)}\n"
            f"{os.path.basename(path)}\n"
            f"Original: {img.width} x {img.height}\n"
            f"Display scale: {self.scale:.3f}\n"
            f"Class: {self.current_class.get()}\n"
            f"Output size: {self.output_size.get()} x {self.output_size.get()}"
        )

    def make_display_image(self, img):
        max_w = 830
        max_h = 700

        scale = min(max_w / img.width, max_h / img.height, 1.0)

        if scale >= 1.0:
            return img.copy(), 1.0

        new_w = int(img.width * scale)
        new_h = int(img.height * scale)

        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return resized, scale

    def clear_canvas(self):
        self.canvas.delete("all")
        self.rect_id = None

    def on_mouse_down(self, event):
        if self.original_image is None:
            return

        self.start_x = event.x
        self.start_y = event.y

        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)

        self.rect_id = self.canvas.create_rectangle(
            self.start_x,
            self.start_y,
            self.start_x,
            self.start_y,
            outline="red",
            width=2
        )

    def on_mouse_drag(self, event):
        if self.rect_id is None:
            return

        self.canvas.coords(
            self.rect_id,
            self.start_x,
            self.start_y,
            event.x,
            event.y
        )

    def on_mouse_up(self, event):
        if self.rect_id is None:
            return

        x0 = min(self.start_x, event.x)
        y0 = min(self.start_y, event.y)
        x1 = max(self.start_x, event.x)
        y1 = max(self.start_y, event.y)

        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
            self.current_rect = None
            return

        self.current_rect = (x0, y0, x1, y1)

        self.set_info(
            self.info_label.cget("text")
            + f"\nSelected display ROI: x={x0}, y={y0}, w={x1 - x0}, h={y1 - y0}"
        )

    def clear_selection(self):
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
        self.rect_id = None
        self.current_rect = None

    def save_roi(self):
        if self.original_image is None:
            messagebox.showwarning("Нет изображения", "Сначала выбери изображение.")
            return

        if self.current_rect is None:
            messagebox.showwarning("Нет ROI", "Сначала выдели ROI мышкой.")
            return

        x0, y0, x1, y1 = self.current_rect

        # Координаты с canvas/display переводим в оригинальное изображение
        ox0 = int(x0 / self.scale)
        oy0 = int(y0 / self.scale)
        ox1 = int(x1 / self.scale)
        oy1 = int(y1 / self.scale)

        ox0 = max(0, ox0)
        oy0 = max(0, oy0)
        ox1 = min(self.original_image.width, ox1)
        oy1 = min(self.original_image.height, oy1)

        if self.force_square.get():
            ox0, oy0, ox1, oy1 = self.make_square_box(
                ox0,
                oy0,
                ox1,
                oy1,
                self.original_image.width,
                self.original_image.height
            )

        crop = self.original_image.crop((ox0, oy0, ox1, oy1))

        size = self.output_size.get()
        crop = crop.resize((size, size), Image.Resampling.LANCZOS)

        class_name = self.current_class.get()
        out_dir = os.path.join(self.output_root.get(), class_name)
        os.makedirs(out_dir, exist_ok=True)

        index = self.next_index(out_dir, class_name)
        out_name = f"{class_name}_{index:02d}.png"
        out_path = os.path.join(out_dir, out_name)

        crop.save(out_path, "PNG")

        self.set_info(
            f"Saved:\n{out_path}\n"
            f"Original ROI: x={ox0}, y={oy0}, w={ox1 - ox0}, h={oy1 - oy0}\n"
            f"Output: {size} x {size}\n"
            f"Class: {class_name}"
        )

        self.clear_selection()

    def make_square_box(self, x0, y0, x1, y1, img_w, img_h):
        w = x1 - x0
        h = y1 - y0
        side = max(w, h)

        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2

        sx0 = cx - side // 2
        sy0 = cy - side // 2
        sx1 = sx0 + side
        sy1 = sy0 + side

        if sx0 < 0:
            sx1 -= sx0
            sx0 = 0

        if sy0 < 0:
            sy1 -= sy0
            sy0 = 0

        if sx1 > img_w:
            shift = sx1 - img_w
            sx0 -= shift
            sx1 = img_w

        if sy1 > img_h:
            shift = sy1 - img_h
            sy0 -= shift
            sy1 = img_h

        sx0 = max(0, sx0)
        sy0 = max(0, sy0)
        sx1 = min(img_w, sx1)
        sy1 = min(img_h, sy1)

        return sx0, sy0, sx1, sy1

    def next_index(self, out_dir, prefix):
        nums = []

        if not os.path.isdir(out_dir):
            return 1

        for name in os.listdir(out_dir):
            if not name.lower().endswith(".png"):
                continue

            stem = os.path.splitext(name)[0]

            if not stem.startswith(prefix + "_"):
                continue

            tail = stem.replace(prefix + "_", "")

            try:
                nums.append(int(tail))
            except ValueError:
                pass

        return max(nums, default=0) + 1

    def delete_last_roi(self):
        class_name = self.current_class.get()
        out_dir = os.path.join(self.output_root.get(), class_name)

        if not os.path.isdir(out_dir):
            messagebox.showinfo("Нет папки", f"Папка не найдена:\n{out_dir}")
            return

        files = []

        for name in os.listdir(out_dir):
            if name.startswith(class_name + "_") and name.lower().endswith(".png"):
                files.append(name)

        if not files:
            messagebox.showinfo("Нет ROI", "В этом классе пока нет ROI.")
            return

        files.sort()
        last_file = files[-1]
        path = os.path.join(out_dir, last_file)

        answer = messagebox.askyesno(
            "Удалить последний ROI?",
            f"Удалить файл?\n{path}"
        )

        if answer:
            os.remove(path)
            self.set_info(f"Удален файл:\n{path}")

    def set_info(self, text):
        self.info_label.config(text=text)


def main():
    root = tk.Tk()
    app = RoiLabelerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()