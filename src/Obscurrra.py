import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
from ttkthemes import ThemedTk

import os
import logging
from PIL import Image, ImageTk
import threading
import sys
import cv2
import glob
import time
from mtcnn import MTCNN
from concurrent.futures import ThreadPoolExecutor


class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.bind_mouse_wheel(canvas)

    def bind_mouse_wheel(self, canvas):
        def _on_mouse_wheel(event):
            canvas.yview_scroll(-1 * int(event.delta / 120), "units")

        self.bind_all("<MouseWheel>", _on_mouse_wheel)
        self.bind_all("<Shift-MouseWheel>", _on_mouse_wheel)


class ObscurrraGUI(ThemedTk):
    """
    This class represents the main GUI application for Obscurrra.
    It is built using Tkinter and ThemedTk for themed widgets.
    """

    def __init__(self):
        """Initializes the Obscurrra GUI application."""
        super().__init__()

        # Set the title and window size
        self.title("Obscurrra")
        self.geometry("800x960")
        self.minsize(800, 960)

        # Initialize main components
        self.main_program = MainProgram()
        self.image_processor = ImageProcessor()
        self.cancel_flag = False
        self.zoom_factor = 1.0
        self.selected_files = []

        # Create a scrollable frame
        self.scrollable_frame = ScrollableFrame(self)
        self.scrollable_frame.pack(fill="both", expand=True)

        # Define available themes and set default theme
        self.available_themes = ['clam', 'alt', 'default', 'classic', 'vista', 'xpnative', 'winnative']
        self.selected_theme = tk.StringVar(value="vista")
        self.create_widgets()  # Create all widgets

        # Set default values for entries and sliders
        self.max_image_size_entry.insert(0, "500")
        self.blur_intensity_slider.set(50)
        self.mtcnn_var.set(True)

    def create_label_entry_pair(self, parent, label_text, row, column, entry_width=50):
        """Creates a label and entry widget pair."""
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=column, padx=5, pady=5, sticky="w")
        entry = ttk.Entry(parent, width=entry_width)
        entry.grid(row=row, column=column+1, padx=5, pady=5, sticky="ew")
        return label, entry

    def create_checkbox(self, parent, text, variable, row, column, sticky="w"):
        """Creates a checkbox widget."""
        checkbox = ttk.Checkbutton(parent, text=text, variable=variable)
        checkbox.grid(row=row, column=column, padx=5, pady=5, sticky=sticky)
        return checkbox

    def create_widgets(self):
        """Creates all the widgets for the GUI."""
        self.style = ttk.Style(self)
        self.style.theme_use(self.selected_theme.get())

        # Folder Selection Frame
        top_frame = ttk.LabelFrame(self.scrollable_frame.scrollable_frame, text="Folder Selection", padding="10")
        top_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        top_frame.columnconfigure(1, weight=1)

        self.input_folder_label, self.input_folder_entry = self.create_label_entry_pair(top_frame, "Input Folder:", 0, 0)
        self.input_folder_button = ttk.Button(top_frame, text="Browse", command=self.browse_input_folder)
        self.input_folder_button.grid(row=0, column=2, padx=5, pady=5)

        self.output_folder_label, self.output_folder_entry = self.create_label_entry_pair(top_frame, "Output Folder:", 1, 0)
        self.output_folder_button = ttk.Button(top_frame, text="Browse", command=self.browse_output_folder)
        self.output_folder_button.grid(row=1, column=2, padx=5, pady=5)

        self.select_images_button = ttk.Button(top_frame, text="Select Individual Images", command=self.browse_images)
        self.select_images_button.grid(row=2, column=0, columnspan=3, padx=5, pady=5)

        # Image and Model Selection Frame
        middle_frame = ttk.LabelFrame(self.scrollable_frame.scrollable_frame, text="Image and Model Selection", padding="10")
        middle_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        middle_frame.columnconfigure(1, weight=1)
        middle_frame.columnconfigure(2, weight=1)
        middle_frame.columnconfigure(3, weight=1)

        self.images_label = ttk.Label(middle_frame, text="Images:")
        self.images_label.grid(row=0, column=0, padx=5, pady=5, sticky="nw")
        self.images_listbox = tk.Listbox(middle_frame, selectmode=tk.MULTIPLE, width=50, height=10)
        self.images_listbox.grid(row=0, column=1, columnspan=3, padx=5, pady=5, sticky="ew")

        self.models_label = ttk.Label(middle_frame, text="Face Detection Models:")
        self.models_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.mtcnn_var = tk.BooleanVar()
        self.mtcnn_checkbox = self.create_checkbox(middle_frame, "MTCNN", self.mtcnn_var, 1, 1, sticky="w")
        self.frontalface_var = tk.BooleanVar()
        self.frontalface_checkbox = self.create_checkbox(middle_frame, "Frontal Face", self.frontalface_var, 1, 2, sticky="w")
        self.profileface_var = tk.BooleanVar()
        self.profileface_checkbox = self.create_checkbox(middle_frame, "Profile Face", self.profileface_var, 1, 3, sticky="w")

        self.dlib_var = tk.BooleanVar()
        self.dlib_checkbox = self.create_checkbox(middle_frame, "Dlib", self.dlib_var, 2, 1, sticky="w")
        self.facenet_var = tk.BooleanVar()
        self.facenet_checkbox = self.create_checkbox(middle_frame, "FaceNet", self.facenet_var, 2, 2, sticky="w")
        self.retinaface_var = tk.BooleanVar()
        self.retinaface_checkbox = self.create_checkbox(middle_frame, "RetinaFace", self.retinaface_var, 2, 3, sticky="w")

        # Preferences Frame
        settings_frame = ttk.LabelFrame(self.scrollable_frame.scrollable_frame, text="Preferences", padding="10")
        settings_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        settings_frame.columnconfigure(1, weight=1)

        self.max_image_size_label, self.max_image_size_entry = self.create_label_entry_pair(settings_frame, "Max Image Size (px):", 0, 0, 10)
        self.blur_intensity_label = ttk.Label(settings_frame, text="Blur Effect Intensity:")
        self.blur_intensity_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.blur_intensity_slider = ttk.Scale(settings_frame, from_=1, to=100, orient=tk.HORIZONTAL, command=self.update_blur_intensity_label)
        self.blur_intensity_slider.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.blur_intensity_value_label = ttk.Label(settings_frame, text="50")
        self.blur_intensity_value_label.grid(row=1, column=2, padx=5, pady=5, sticky="w")

        # Processing Control and Log Frame
        bottom_frame = ttk.LabelFrame(self.scrollable_frame.scrollable_frame, text="Processing Control and Log", padding="10")
        bottom_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        bottom_frame.columnconfigure(1, weight=1)

        self.start_button = ttk.Button(bottom_frame, text="Start Processing", command=self.start_processing)
        self.start_button.grid(row=0, column=0, padx=5, pady=5)
        self.cancel_button = ttk.Button(bottom_frame, text="Cancel Processing", command=self.cancel_processing)
        self.cancel_button.grid(row=0, column=1, padx=5, pady=5)

        self.progress_label = ttk.Label(bottom_frame, text="Progress:")
        self.progress_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.progress_bar = ttk.Progressbar(bottom_frame, length=200, mode='determinate')
        self.progress_bar.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        self.log_label = ttk.Label(bottom_frame, text="Log:")
        self.log_label.grid(row=2, column=0, padx=5, pady=5, sticky="nw")
        self.log_display = scrolledtext.ScrolledText(bottom_frame, width=70, height=8)
        self.log_display.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        self.total_images_label = ttk.Label(bottom_frame, text="Total Images Processed:")
        self.total_images_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.total_images_count = ttk.Label(bottom_frame, text="0")
        self.total_images_count.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        self.total_faces_label = ttk.Label(bottom_frame, text="Total Faces Detected:")
        self.total_faces_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.total_faces_count = ttk.Label(bottom_frame, text="0")
        self.total_faces_count.grid(row=4, column=1, padx=5, pady=5, sticky="ew")

        self.time_taken_label = ttk.Label(bottom_frame, text="Time Taken:")
        self.time_taken_label.grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.time_taken_count = ttk.Label(bottom_frame, text="0")
        self.time_taken_count.grid(row=5, column=1, padx=5, pady=5, sticky="ew")

        # Image Preview Frame
        image_preview_frame = ttk.LabelFrame(self.scrollable_frame.scrollable_frame, text="Image Preview", padding="10")
        image_preview_frame.grid(row=4, column=0, padx=10, pady=5, sticky="ew")
        image_preview_frame.columnconfigure(0, weight=1)
        image_preview_frame.columnconfigure(1, weight=1)

        self.original_image_label = ttk.Label(image_preview_frame, text="Original Image:")
        self.original_image_label.grid(row=0, column=0, padx=5, pady=5)
        self.original_image_canvas = tk.Canvas(image_preview_frame, width=200, height=200)
        self.original_image_canvas.grid(row=1, column=0, padx=5, pady=5)

        self.processed_image_label = ttk.Label(image_preview_frame, text="Processed Image:")
        self.processed_image_label.grid(row=0, column=1, padx=5, pady=5)
        self.processed_image_canvas = tk.Canvas(image_preview_frame, width=200, height=200)
        self.processed_image_canvas.grid(row=1, column=1, padx=5, pady=5)

        self.zoom_in_button = ttk.Button(image_preview_frame, text="Zoom In", command=self.zoom_in)
        self.zoom_in_button.grid(row=2, column=0, padx=5, pady=5)
        self.zoom_out_button = ttk.Button(image_preview_frame, text="Zoom Out", command=self.zoom_out)
        self.zoom_out_button.grid(row=2, column=1, padx=5, pady=5)

        # Help, Feedback, and Batch Processing Frame
        help_batch_frame = ttk.LabelFrame(self.scrollable_frame.scrollable_frame, text="Help, Feedback, and Batch Processing", padding="10")
        help_batch_frame.grid(row=5, column=0, padx=10, pady=5, sticky="ew")
        help_batch_frame.columnconfigure(1, weight=1)

        self.help_button = ttk.Button(help_batch_frame, text="Help", command=self.show_help)
        self.help_button.grid(row=0, column=0, padx=5, pady=5)
        self.save_log_button = ttk.Button(help_batch_frame, text="Save Log", command=self.save_log)
        self.save_log_button.grid(row=0, column=1, padx=5, pady=5)
        self.batch_process_button = ttk.Button(help_batch_frame, text="Batch Process", command=self.batch_process)
        self.batch_process_button.grid(row=0, column=2, padx=5, pady=5)

        self.theme_label = ttk.Label(help_batch_frame, text="Select Theme:")
        self.theme_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.theme_dropdown = ttk.Combobox(help_batch_frame, textvariable=self.selected_theme, values=self.available_themes)
        self.theme_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.theme_dropdown.bind("<<ComboboxSelected>>", self.change_theme)

        # Exit and Save Settings Frame
        exit_frame = ttk.Frame(self.scrollable_frame.scrollable_frame)
        exit_frame.grid(row=6, column=0, padx=10, pady=5, sticky="ew")
        exit_frame.columnconfigure(1, weight=1)

        self.save_settings_button = ttk.Button(exit_frame, text="Save Settings", command=self.save_settings)
        self.save_settings_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.exit_button = ttk.Button(exit_frame, text="Exit", command=self.exit_application)
        self.exit_button.grid(row=0, column=1, padx=5, pady=5, sticky="e")

        self.redirect_logging()

    def change_theme(self, event=None):
        """Changes the theme of the application based on user selection."""
        available_themes = self.style.theme_names()
        selected_theme = self.selected_theme.get()
        if selected_theme in available_themes:
            self.style.theme_use(selected_theme)
        else:
            print(f"Theme {selected_theme} is not available. Skipping...")
            self.style.theme_use('vista')  # Set default theme

    def redirect_logging(self):
        """Redirects logging output to the GUI's log display."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        log_handler = LogHandler(self.log_display)
        logging.getLogger().addHandler(log_handler)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    def update_blur_intensity_label(self, value):
        """Updates the blur intensity label based on the slider value."""
        self.blur_intensity_value_label.config(text=str(int(float(value))))

    def browse_input_folder(self):
        """Opens a dialog to browse and select the input folder."""
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.input_folder_entry.delete(0, tk.END)
            self.input_folder_entry.insert(0, folder_selected)
            self.selected_files = []
            self.load_images()

    def browse_output_folder(self):
        """Opens a dialog to browse and select the output folder."""
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.output_folder_entry.delete(0, tk.END)
            self.output_folder_entry.insert(0, folder_selected)

    def browse_images(self):
        """Opens a dialog to browse and select individual image files."""
        filetypes = [("Image files", "*.jpg *.jpeg *.png *.webp")]
        files_selected = filedialog.askopenfilenames(filetypes=filetypes)
        if files_selected:
            self.selected_files = list(files_selected)
            self.input_folder_entry.delete(0, tk.END)
            self.load_images()

    def load_images(self):
        """Loads selected images into the images listbox."""
        self.images_listbox.delete(0, tk.END)
        if self.selected_files:
            for image in self.selected_files:
                self.images_listbox.insert(tk.END, os.path.basename(image))
        else:
            input_folder = self.input_folder_entry.get()
            if os.path.isdir(input_folder):
                for image in os.listdir(input_folder):
                    if image.lower().endswith(('jpg', 'jpeg', 'png', 'webp')):
                        self.images_listbox.insert(tk.END, image)
            else:
                messagebox.showerror("Error", "Invalid input folder")

    def start_processing(self):
        """Starts the image processing in a separate thread."""
        input_folder = self.input_folder_entry.get()
        output_folder = self.output_folder_entry.get()
        models = []
        if self.mtcnn_var.get():
            models.append('mtcnn')
        if self.frontalface_var.get():
            models.append('frontalface')
        if self.profileface_var.get():
            models.append('profileface')
        if self.dlib_var.get():
            models.append('dlib')
        if self.facenet_var.get():
            models.append('facenet')
        if self.retinaface_var.get():
            models.append('retinaface')

        if not os.path.isdir(input_folder) and not self.selected_files:
            messagebox.showerror("Error", "Invalid input folder or no images selected")
            return

        if not os.path.isdir(output_folder):
            output_folder = os.path.join(input_folder, "Obscurrred")
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            self.output_folder_entry.delete(0, tk.END)
            self.output_folder_entry.insert(0, output_folder)

        self.cancel_flag = False
        self.log_display.delete('1.0', tk.END)

        threading.Thread(target=self.process_images, args=(input_folder, output_folder, models)).start()

    def process_images(self, input_folder, output_folder, models):
        """Processes all images in the input folder using the selected models and blurs detected faces."""
        total_images = 0
        total_faces = 0
        no_faces = 0
        image_files = self.selected_files if self.selected_files else [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(('jpg', 'jpeg', 'png', 'webp'))]

        blur_intensity = int(self.blur_intensity_value_label.cget("text"))
        blur_effect = (blur_intensity, blur_intensity)
        start_time = time.time()

        try:
            self.log_display.insert(tk.END, "Starting processing...\n")
            self.progress_bar['value'] = 0
            self.progress_bar['maximum'] = len(image_files)

            for image_file in image_files:
                if self.cancel_flag:
                    break

                result = self.image_processor.process_single_image(image_file, output_folder, models, blur_effect)
                total_images += 1
                total_faces += result['faces']
                if result['faces'] == 0:
                    no_faces += 1
                self.progress_bar['value'] = total_images
                self.log_display.insert(tk.END, f"Processed {os.path.basename(image_file)}, found {result['faces']} faces.\n")
                self.log_display.yview(tk.END)
                self.update_image_preview(image_file, result['output_path'])

            elapsed_time = time.time() - start_time
            self.time_taken_count.config(text=f"{elapsed_time:.2f} seconds")

            if not self.cancel_flag:
                self.log_display.insert(tk.END, f"Processing complete. Total images processed: {total_images}, Total faces detected: {total_faces}, Total images without faces: {no_faces}, Time taken: {elapsed_time:.2f} seconds.\n")
                messagebox.showinfo("Success", "Processing complete")
            else:
                self.log_display.insert(tk.END, "Processing cancelled.\n")
                messagebox.showinfo("Cancelled", "Processing cancelled")
        except Exception as e:
            logging.error(f"Error processing images: {e}")
            self.log_display.insert(tk.END, f"Error: {e}\n")
            messagebox.showerror("Error", "An error occurred during processing")

        self.total_images_count.config(text=str(total_images))
        self.total_faces_count.config(text=str(total_faces))

    def cancel_processing(self):
        """Sets the cancel flag to stop processing."""
        self.cancel_flag = True
        logging.info("Processing cancelled by user.")

    def zoom_in(self):
        """Zooms in on the image preview."""
        self.zoom_factor *= 1.2
        self.update_image_preview()

    def zoom_out(self):
        """Zooms out on the image preview."""
        self.zoom_factor /= 1.2
        self.update_image_preview()

    def update_image_preview(self, original_image_path=None, processed_image_path=None):
        """Updates the image preview with the original and processed images."""
        if original_image_path:
            original_image = Image.open(original_image_path)
            original_image.thumbnail((200, 200), Image.LANCZOS)
            self.original_image = ImageTk.PhotoImage(original_image)
            self.original_image_canvas.create_image(0, 0, anchor=tk.NW, image=self.original_image)

        if processed_image_path:
            processed_image = Image.open(processed_image_path)
            processed_image.thumbnail((200, 200), Image.LANCZOS)
            self.processed_image = ImageTk.PhotoImage(processed_image)
            self.processed_image_canvas.create_image(0, 0, anchor=tk.NW, image=self.processed_image)

    def batch_process(self):
        """Starts the batch processing of selected images."""
        input_folder = self.input_folder_entry.get()
        output_folder = self.output_folder_entry.get()
        models = []
        if self.mtcnn_var.get():
            models.append('mtcnn')
        if self.frontalface_var.get():
            models.append('frontalface')
        if self.profileface_var.get():
            models.append('profileface')
        if self.dlib_var.get():
            models.append('dlib')
        if self.facenet_var.get():
            models.append('facenet')
        if self.retinaface_var.get():
            models.append('retinaface')

        if not os.path.isdir(input_folder) and not self.selected_files:
            messagebox.showerror("Error", "Invalid input folder or no images selected")
            return

        if not os.path.isdir(output_folder):
            output_folder = os.path.join(input_folder, "Obscurred")
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            self.output_folder_entry.delete(0, tk.END)
            self.output_folder_entry.insert(0, output_folder)

        selected_images = [self.images_listbox.get(idx) for idx in self.images_listbox.curselection()]
        if not selected_images:
            messagebox.showerror("Error", "No images selected")
            return

        self.cancel_flag = False

        threading.Thread(target=self.process_batch_images, args=(output_folder, models, selected_images)).start()

    def process_batch_images(self, output_folder, models, selected_images):
        """Processes a batch of selected images."""
        total_images = 0
        total_faces = 0
        no_faces = 0
        image_files = [os.path.join(self.input_folder_entry.get(), image) for image in selected_images]

        blur_intensity = int(self.blur_intensity_value_label.cget("text"))
        blur_effect = (blur_intensity, blur_intensity)
        start_time = time.time()

        try:
            self.log_display.insert(tk.END, "Starting batch processing...\n")
            self.progress_bar['value'] = 0
            self.progress_bar['maximum'] = len(image_files)

            for image_file in image_files:
                if self.cancel_flag:
                    break

                result = self.image_processor.process_single_image(image_file, output_folder, models, blur_effect)
                total_images += 1
                total_faces += result['faces']
                if result['faces'] == 0:
                    no_faces += 1
                self.progress_bar['value'] = total_images
                self.log_display.insert(tk.END, f"Processed {os.path.basename(image_file)}, found {result['faces']} faces.\n")
                self.log_display.yview(tk.END)
                self.update_image_preview(image_file, result['output_path'])

            elapsed_time = time.time() - start_time
            self.time_taken_count.config(text=f"{elapsed_time:.2f} seconds")

            if not self.cancel_flag:
                self.log_display.insert(tk.END, f"Batch processing complete. Total images processed: {total_images}, Total faces detected: {total_faces}, Total images without faces: {no_faces}, Time taken: {elapsed_time:.2f} seconds.\n")
                messagebox.showinfo("Success", "Batch processing complete")
            else:
                self.log_display.insert(tk.END, "Batch processing cancelled.\n")
                messagebox.showinfo("Cancelled", "Batch processing cancelled")
        except Exception as e:
            logging.error(f"Error during batch processing: {e}")
            self.log_display.insert(tk.END, f"Error: {e}\n")
            messagebox.showerror("Error", "An error occurred during batch processing")

        self.total_images_count.config(text=str(total_images))
        self.total_faces_count.config(text=str(total_faces))
        self.log_display.insert(tk.END, f"Total images processed: {total_images}, Total faces detected: {total_faces}, Total images without faces: {total_images - total_faces}\n")
        self.log_display.insert(tk.END, f"Time taken: {elapsed_time:.2f} seconds\n")

    def show_help(self):
        """Displays the help message."""
        help_message = (
            ''' 
Obscurrra GUI Help
---
Quick and Simple:
1. Select the input folder containing the images to be processed.
    - You can also select individual images by clicking 'Select Individual Images'.
2. Select the output folder where the processed images will be saved.
3. Click 'Start Processing' to begin face detection and blurring.
4. "Processing Complete" message will be displayed once the process is done.

Advanced Options:
- Prefferences: Adjust the max image size and blur effect intensity as needed.
- Image and Model Selection: Choose the face detection models to use.
- Image Preview: Compare original and processed images.
- Log: Check the log for real-time updates and progress.
- Save Settings: Save your current settings.
- Exit: Close the application safely.

Feedback and Support:
- [GitHub repo](https://github.com/jakujobi/obscurrra) for the source code
- [Website](obscurrra.jakujobi.com)

This software is provided as-is without any guarantees or warranties under the Hippocratic License 3.0 (HL3). 
Please use responsibly and respect privacy and ethical considerations.
'''
        )
        messagebox.showinfo("Help", help_message)

    def save_log(self):
        """Saves the log to a file."""
        log_content = self.log_display.get("1.0", tk.END)
        log_file = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if log_file:
            with open(log_file, "w") as file:
                file.write(log_content)

    def save_settings(self):
        """Placeholder for saving settings (not implemented)."""
        pass

    def exit_application(self):
        """Exits the application."""
        self.destroy()

class LogHandler(logging.Handler):
    """
    A custom logging handler that redirects log messages to a Tkinter ScrolledText widget.
    """
    def __init__(self, log_display):
        """
        Initializes the LogHandler.

        Args:
            log_display (ScrolledText): The ScrolledText widget to display log messages.
        """
        super().__init__()
        self.log_display = log_display

    def emit(self, record):
        """
        Emits a log record.

        Args:
            record (LogRecord): The log record to emit.
        """
        log_entry = self.format(record)
        self.log_display.insert(tk.END, log_entry + "\n")
        self.log_display.yview(tk.END)


class MainProgram:
    """
    Main program class that manages directory operations and image processing.
    """
    OUTPUT_FOLDER_NAME = 'blurred'

    def __init__(self):
        """
        Initializes the MainProgram.
        """
        self.directory_manager = DirectoryManager()
        self.image_processor = ImageProcessor()

    def run(self, models):
        """
        Runs the main program to process all images in the current directory using the specified models.

        Args:
            models (list): List of face detection models to use.
        """
        try:
            current_directory = self.directory_manager.get_current_directory()
            input_folder = current_directory
            output_folder = os.path.join(current_directory, MainProgram.OUTPUT_FOLDER_NAME)
            self.directory_manager.create_output_directory(output_folder)
            self.image_processor.process_all_images(input_folder, output_folder, models)
        except Exception as e:
            logging.error(f"Error running the program: {e}")
            raise e


class DirectoryManager:
    """
    Class for managing directory operations.
    """
    TEST_DIR_SUFFIX = '/test'

    @staticmethod
    def get_current_directory():
        """
        Gets the current directory path with a test directory suffix.

        Returns:
            str: The current directory path with the test directory suffix.
        """
        try:
            current_directory = os.path.dirname(os.path.realpath(__file__))
            logging.info(f"Getting current directory: {current_directory}")
            return current_directory + DirectoryManager.TEST_DIR_SUFFIX
        except Exception as e:
            logging.error(f"Error getting current directory: {e}")
            raise e

    @staticmethod
    def create_output_directory(output_folder):
        """
        Creates the output directory if it does not exist.

        Args:
            output_folder (str): The path to the output folder.
        """
        try:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                logging.info(f"Output directory created at: {output_folder}")
        except Exception as e:
            logging.error(f"Error creating output directory {output_folder}: {e}")
            raise e


class Preprocessor:
    """
    Class for preprocessing images before face detection.
    """
    @staticmethod
    def read_image(image_path):
        """
        Reads an image from the specified path.

        Args:
            image_path (str): The path to the image file.

        Returns:
            ndarray: The image as a NumPy array.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Error reading image {image_path}.")
        return img

    @staticmethod
    def resize_image(image, max_dimension):
        """
        Resizes an image to fit within the specified maximum dimension.

        Args:
            image (ndarray): The image to resize.
            max_dimension (int): The maximum dimension for resizing.

        Returns:
            ndarray: The resized image.
        """
        height, width = image.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_size = (int(width * scale), int(height * scale))
            resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            return resized_image
        return image

    @staticmethod
    def preprocess_image(image):
        """
        Converts an image to grayscale and equalizes its histogram.

        Args:
            image (ndarray): The image to preprocess.

        Returns:
            ndarray: The preprocessed image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        return gray


class FaceDetection:
    """
    Class for detecting faces in images using various models.
    """
    FRONT_FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    PROFILE_FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_profileface.xml'

    def __init__(self):
        """
        Initializes the FaceDetection class and loads face detection models.
        """
        self._front_face_cascade = self._load_face_detection_model(self.FRONT_FACE_CASCADE_PATH)
        self._profile_face_cascade = self._load_face_detection_model(self.PROFILE_FACE_CASCADE_PATH)
        self._mtcnn_detector = MTCNN()

    @staticmethod
    def _load_face_detection_model(model_path):
        """
        Loads a face detection model from the specified path.

        Args:
            model_path (str): The path to the model file.

        Returns:
            CascadeClassifier: The loaded face detection model.
        """
        try:
            model = cv2.CascadeClassifier(model_path)
            return model
        except Exception as e:
            logging.error(f"Error loading face detection model: {e}")
            raise e

    @property
    def front_face_cascade(self):
        """Returns the front face cascade model."""
        return self._front_face_cascade

    @property
    def profile_face_cascade(self):
        """Returns the profile face cascade model."""
        return self._profile_face_cascade

    @property
    def mtcnn_detector(self):
        """Returns the MTCNN detector."""
        return self._mtcnn_detector

    def choose_model(self, models, image, gray_image):
        """
        Chooses and applies the specified face detection models to detect faces in an image.

        Args:
            models (list): List of face detection models to use.
            image (ndarray): The original image.
            gray_image (ndarray): The preprocessed grayscale image.

        Returns:
            list: List of detected faces as (x, y, w, h) tuples.
        """
        faces = []
        if 'mtcnn' in models:
            faces.extend(self.detect_faces_mtcnn(image))
        if 'frontalface' in models:
            faces.extend(self.filter_faces(self.detect_faces(gray_image, self.front_face_cascade), faces))
        if 'profileface' in models:
            faces.extend(self.filter_faces(self.detect_faces(gray_image, self.profile_face_cascade), faces))
        return faces

    @staticmethod
    def filter_faces(new_faces, existing_faces):
        """
        Filters out overlapping faces to avoid duplicate detections.

        Args:
            new_faces (list): List of newly detected faces.
            existing_faces (list): List of already detected faces.

        Returns:
            list: Filtered list of faces.
        """
        filtered_faces = []
        for face in new_faces:
            if not any(FaceDetection.is_same_face(face, existing_face) for existing_face in existing_faces):
                filtered_faces.append(face)
        return filtered_faces

    @staticmethod
    def is_same_face(face1, face2):
        """
        Checks if two faces are the same based on their coordinates and sizes.

        Args:
            face1 (tuple): Coordinates and size of the first face (x, y, w, h).
            face2 (tuple): Coordinates and size of the second face (x, y, w, h).

        Returns:
            bool: True if the faces are the same, False otherwise.
        """
        x1, y1, w1, h1 = face1
        x2, y2, w2, h2 = face2
        return abs(x1 - x2) < w1 / 2 and abs(y1 - y2) < h1 / 2

    @staticmethod
    def detect_faces(gray, face_cascade):
        """
        Detects faces in a grayscale image using the specified face cascade.

        Args:
            gray (ndarray): The preprocessed grayscale image.
            face_cascade (CascadeClassifier): The face cascade to use for detection.

        Returns:
            list: List of detected faces as (x, y, w, h) tuples.
        """
        try:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            return faces
        except Exception as e:
            logging.error(f"Error detecting faces: {e}")
            raise e

    def detect_faces_mtcnn(self, image):
        """
        Detects faces in an image using the MTCNN detector.

        Args:
            image (ndarray): The original image.

        Returns:
            list: List of detected faces as (x, y, w, h) tuples.
        """
        try:
            results = self.mtcnn_detector.detect_faces(image)
            faces = [(result['box'][0], result['box'][1], result['box'][2], result['box'][3]) for result in results]
            return faces
        except Exception as e:
            logging.error(f"Error detecting faces with MTCNN: {e}")
            raise e


class FaceBlurrer:
    """
    Class for blurring detected faces in images.
    """
    @staticmethod
    def blur_faces(img, faces, blur_effect):
        """
        Applies a blur effect to the detected faces in an image.

        Args:
            img (ndarray): The image containing faces.
            faces (list): List of detected faces as (x, y, w, h) tuples.
            blur_effect (tuple): The blur effect to apply as (width, height).

        Returns:
            ndarray: The image with blurred faces.
        """
        try:
            # Ensure blur_effect is a tuple of integers
            blur_effect = (int(blur_effect[0]), int(blur_effect[1]))
            for (x, y, w, h) in faces:
                img[y:y+h, x:x+w] = cv2.blur(img[y:y+h, x:x+w], blur_effect)
            return img
        except Exception as e:
            logging.error(f"Error blurring faces: {e}")
            raise e

class ImageProcessor:
    """
    Class for processing images including resizing, face detection, and face blurring.
    """
    IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    _MAX_IMAGE_SIZE = 1000

    def __init__(self):
        """
        Initializes the ImageProcessor.
        """
        self.preprocessor = Preprocessor()
        self.face_detection = FaceDetection()
        self.face_blurrer = FaceBlurrer()

    @property
    def max_image_size(self):
        """Returns the maximum image size."""
        return self._MAX_IMAGE_SIZE

    @max_image_size.setter
    def max_image_size(self, value):
        """
        Sets the maximum image size.

        Args:
            value (int): The new maximum image size.
        """
        if value > 0:
            self._MAX_IMAGE_SIZE = value
        else:
            raise ValueError("Maximum image size must be greater than 0.")

    @staticmethod
    def get_output_path(image_path, output_folder):
        """
        Gets the output path for a processed image.

        Args:
            image_path (str): The path to the original image.
            output_folder (str): The folder to save the processed image.

        Returns:
            str: The output path for the processed image.
        """
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        return os.path.join(output_folder, f"{name}_b{ext}")

    def process_single_image(self, image_path, output_folder, models, blur_effect):
        """
        Processes a single image including resizing, face detection, and face blurring.

        Args:
            image_path (str): The path to the image file.
            output_folder (str): The folder to save the processed image.
            models (list): List of face detection models to use.
            blur_effect (tuple): The blur effect to apply as (width, height).

        Returns:
            dict: A dictionary with the number of faces detected and the output path.
        """
        try:
            logging.info(f"Processing {image_path}")
            original_img = self.preprocessor.read_image(image_path)
            resized_img = self.preprocessor.resize_image(original_img.copy(), self.max_image_size)
            gray = self.preprocessor.preprocess_image(resized_img)
            faces = self.face_detection.choose_model(models, resized_img, gray)

            scale_factor = max(original_img.shape[:2]) / max(resized_img.shape[:2])

            if faces:
                logging.info(f"Detected {len(faces)} face(s) in {image_path}.")
                faces = [(int(x*scale_factor), int(y*scale_factor), int(w*scale_factor), int(h*scale_factor)) for (x, y, w, h) in faces]
                self.face_blurrer.blur_faces(original_img, faces, blur_effect)
            else:
                logging.info(f"No faces detected in {image_path}.")
            output_path = self.get_output_path(image_path, output_folder)
            cv2.imwrite(output_path, original_img)
            logging.info(f"Processed and saved {output_path}")
            return {'faces': len(faces), 'output_path': output_path}
        except Exception as e:
            logging.error(f"Error processing {image_path}: {e}")
            raise e

    def process_all_images(self, input_folder, output_folder, models):
        """
        Processes all images in the input folder using the specified face detection models.

        Args:
            input_folder (str): The folder containing the images to process.
            output_folder (str): The folder to save the processed images.
            models (list): List of face detection models to use.
        """
        try:
            start_time = time.time()
            total_faces = 0
            total_images = 0
            with ThreadPoolExecutor() as executor:
                futures = []
                for extension in self.IMAGE_EXTENSIONS:
                    for filename in glob.glob(os.path.join(input_folder, extension)):
                        futures.append(executor.submit(self.process_single_image, filename, output_folder, models, blur_effect))
                for future in futures:
                    result = future.result()
                    total_faces += result['faces']
                    total_images += 1
            end_time = time.time()
            elapsed_time = end_time - start_time
            logging.info(f"Face blurring complete. Time taken: {elapsed_time} seconds.")
            logging.info(f"Total images processed: {total_images}")
            logging.info(f"Total faces found: {total_faces}")
            logging.info(f"Total images without faces: {total_images - total_faces}")
        except Exception as e:
            logging.error(f"Error processing all images: {e}")
            raise e


if __name__ == "__main__":
    app = ObscurrraGUI()
    app.mainloop()