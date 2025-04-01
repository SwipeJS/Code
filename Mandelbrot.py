import numpy as np
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk
from numba import cuda
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# CUDA kernel for Mandelbrot computation
@cuda.jit
def mandelbrot_kernel(xmin, xmax, ymin, ymax, width, height, max_iter, output):
    x_idx, y_idx = cuda.grid(2)

    if x_idx < width and y_idx < height:
        real = xmin + x_idx * (xmax - xmin) / width
        imag = ymin + y_idx * (ymax - ymin) / height
        c = complex(real, imag)
        z = 0 + 0j
        count = 0

        while abs(z) <= 2 and count < max_iter:
            z = z * z + c
            count += 1

        output[y_idx, x_idx] = count

# Function to compute the Mandelbrot set using CUDA
def mandelbrot_set_cuda(xmin, xmax, ymin, ymax, width, height, max_iter):
    output = np.zeros((height, width), dtype=np.int32)
    d_output = cuda.to_device(output)

    # Define grid and block dimensions
    threads_per_block = (16, 16)
    blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Launch the kernel
    mandelbrot_kernel[blocks_per_grid, threads_per_block](
        xmin, xmax, ymin, ymax, width, height, max_iter, d_output
    )
    d_output.copy_to_host(output)

    return output

# Create Mandelbrot plot in Tkinter canvas
class MandelbrotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mandelbrot Set Viewer")
        self.root.attributes("-fullscreen", True) 
        self.root.overrideredirect(True)  
        self.root.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0")
        
        self.width = self.root.winfo_screenwidth()  # Use screen width
        self.height = self.root.winfo_screenheight()  # Use screen height
        self.max_iter = 1048

        # Canvas for displaying Mandelbrot
        self.canvas = Canvas(root, width=self.width, height=self.height, bd=0, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Initial zoom area for Mandelbrot set
        self.xmin, self.xmax, self.ymin, self.ymax = -3.0, 2.0, -1.5, 1.5

        # Variables for rectangle selection
        self.rect_id = None
        self.start_x = self.start_y = 0

        # Generate the Mandelbrot set image
        self.generate_and_draw()

        # Bindings for mouse interaction
        self.canvas.bind("<ButtonPress-1>", self.start_rectangle)
        self.canvas.bind("<B1-Motion>", self.draw_rectangle)
        self.canvas.bind("<ButtonRelease-1>", self.zoom_to_rectangle)

    def generate_and_draw(self):
        # Compute the Mandelbrot set using CUDA
        data = mandelbrot_set_cuda(
            self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter
        )

        # Normalize data for colormap
        normalized_data = data / self.max_iter

        # Apply colormap
        colormap = cm.CMRmap
        img_colored = colormap(normalized_data)
        img_colored = (img_colored[:, :, :3] * 255).astype(np.uint8)

        # Convert to Image
        img = Image.fromarray(img_colored)

        # Convert to Tkinter-compatible format
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

    def start_rectangle(self, event):
        self.start_x, self.start_y = event.x, event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")

    def draw_rectangle(self, event):
        end_x, end_y = event.x, event.y
        aspect_ratio = 16 / 9
        rect_width = abs(end_x - self.start_x)
        rect_height = rect_width / aspect_ratio
        if end_y < self.start_y:
            end_y = self.start_y - rect_height
        else:
            end_y = self.start_y + rect_height

        self.canvas.coords(self.rect_id, self.start_x, self.start_y, end_x, end_y)

    def zoom_to_rectangle(self, event):
        if not self.rect_id:
            return

        coords = self.canvas.coords(self.rect_id)
        x1, y1, x2, y2 = coords
        self.canvas.delete(self.rect_id)

        x_min = min(x1, x2)
        x_max = max(x1, x2)
        y_min = min(y1, y2)
        y_max = max(y1, y2)

        # Map canvas coordinates to Mandelbrot set coordinates
        x_min_mapped = self.xmin + (x_min / self.width) * (self.xmax - self.xmin)
        x_max_mapped = self.xmin + (x_max / self.width) * (self.xmax - self.xmin)
        y_min_mapped = self.ymin + (y_min / self.height) * (self.ymax - self.ymin)
        y_max_mapped = self.ymin + (y_max / self.height) * (self.ymax - self.ymin)

        self.xmin, self.xmax = x_min_mapped, x_max_mapped
        self.ymin, self.ymax = y_min_mapped, y_max_mapped

        # Redraw the Mandelbrot set
        self.generate_and_draw()

# Run via Tkinter
if __name__ == "__main__":
    root = tk.Tk()
    app = MandelbrotApp(root)
    root.mainloop()
