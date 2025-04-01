import numpy as np
import tkinter as tk
from tkinter import Canvas, simpledialog
from PIL import Image, ImageTk
from numba import cuda
import matplotlib.cm as cm

# CUDA kernel for Julia set computation
@cuda.jit
def julia_kernel(xmin, xmax, ymin, ymax, width, height, max_iter, c_real, c_imag, output):
    x_idx, y_idx = cuda.grid(2)
    if x_idx < width and y_idx < height:
        real = xmin + x_idx * (xmax - xmin) / width
        imag = ymin + y_idx * (ymax - ymin) / height
        z = complex(real, imag)
        c = complex(c_real, c_imag)
        count = 0
        while abs(z) <= 2 and count < max_iter:
            z = z * z + c
            count += 1
        output[y_idx, x_idx] = count

# Function to compute the Julia set using CUDA
def julia_set_cuda(xmin, xmax, ymin, ymax, width, height, max_iter, c_real, c_imag):
    output = np.zeros((height, width), dtype=np.int32)
    d_output = cuda.to_device(output)

    threads_per_block = (16, 16)
    blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    julia_kernel[blocks_per_grid, threads_per_block](
        xmin, xmax, ymin, ymax, width, height, max_iter, c_real, c_imag, d_output
    )
    d_output.copy_to_host(output)
    return output

# Julia Set Viewer Class
class JuliaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Julia Set Viewer")

        # Get screen size for correct aspect ratio
        self.width = self.root.winfo_screenwidth()
        self.height = self.root.winfo_screenheight()

        # Set window to fullscreen
        self.root.attributes("-fullscreen", True)
        self.root.geometry(f"{self.width}x{self.height}+0+0")
        self.root.resizable(False, False) 

        self.max_iter = 1048  

        self.canvas = Canvas(root, width=self.width, height=self.height, bd=0, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Initial Julia set parameters
        self.xmin, self.xmax, self.ymin, self.ymax = -3.0, 2.0, -1.5, 1.5
        
        # Get user input for Julia set parameters
        self.c_real = simpledialog.askfloat("Julia Set Input", "Enter real part of c:", minvalue=-2, maxvalue=2)
        self.c_imag = simpledialog.askfloat("Julia Set Input", "Enter imaginary part of c:", minvalue=-2, maxvalue=2)
        
        # Ensure valid inputs
        if self.c_real is None:
            self.c_real = -0.7  # Default value
        if self.c_imag is None:
            self.c_imag = 0.27015  # Default value
        
        print(f"Using c = {self.c_real} + {self.c_imag}i") 
        
        self.zoom_stack = [(self.xmin, self.xmax, self.ymin, self.ymax)]

        self.rect_id = None
        self.start_x = self.start_y = 0

        self.generate_and_draw()
        
        # Bindings for zooming and right-click reset
        self.canvas.bind("<ButtonPress-1>", self.start_rectangle)
        self.canvas.bind("<B1-Motion>", self.draw_rectangle)
        self.canvas.bind("<ButtonRelease-1>", self.zoom_to_rectangle)
        self.canvas.bind("<Button-3>", self.reset_zoom) 

    # Function for drawing Julia Set
    def generate_and_draw(self):
        data = julia_set_cuda(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter, self.c_real, self.c_imag)
        normalized_data = data / self.max_iter
        colormap = cm.CMRmap
        img_colored = colormap(normalized_data)
        img_colored = (img_colored[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(img_colored)
        
        self.tk_img = ImageTk.PhotoImage(img) 
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        
        self.root.update_idletasks() 
    
    def start_rectangle(self, event):
        self.start_x, self.start_y = event.x, event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")

    def draw_rectangle(self, event):
        end_x, end_y = event.x, event.y
        aspect_ratio = self.width / self.height
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

        x_min_mapped = self.xmin + (min(x1, x2) / self.width) * (self.xmax - self.xmin)
        x_max_mapped = self.xmin + (max(x1, x2) / self.width) * (self.xmax - self.xmin)
        y_min_mapped = self.ymin + (min(y1, y2) / self.height) * (self.ymax - self.ymin)
        y_max_mapped = self.ymin + (max(y1, y2) / self.height) * (self.ymax - self.ymin)

        self.zoom_stack.append((self.xmin, self.xmax, self.ymin, self.ymax))

        self.xmin, self.xmax = x_min_mapped, x_max_mapped
        self.ymin, self.ymax = y_min_mapped, y_max_mapped

        self.generate_and_draw()

    def reset_zoom(self, event):
        if len(self.zoom_stack) > 1:
            self.zoom_stack.pop() 
            self.xmin, self.xmax, self.ymin, self.ymax = self.zoom_stack[-1] 
            self.generate_and_draw()

# Run Tkinter application
if __name__ == "__main__":
    root = tk.Tk()
    app = JuliaApp(root)
    root.mainloop()
