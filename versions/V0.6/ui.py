import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import os
import logging
from main_program import MainProgram
from main_program import ImageProcessor  # Import ImageProcessor directly
from PIL import Image, ImageTk  # For handling image display and zoom
import threading
import sys

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
        """Enable scrolling with the mouse wheel."""
        def _on_mouse_wheel(event):
            canvas.yview_scroll(-1 * int(event.delta / 120), "units")

        self.bind_all("<MouseWheel>", _on_mouse_wheel)
        self.bind_all("<Shift-MouseWheel>", _on_mouse_wheel)

class ObscurrraGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # Set the title and size of the main window
        self.title("Obscurrra")
        self.geometry("760x960")
        self.minsize(760, 960)  # Set minimum size for better responsiveness

        # Initialize the main processing program and ImageProcessor
        self.main_program = MainProgram()
        self.image_processor = ImageProcessor()  # Initialize ImageProcessor
        self.cancel_flag = False  # Flag to signal cancellation
        self.zoom_factor = 1.0  # Initial zoom factor
        self.selected_files = []  # To store selected individual image files

        # Create a scrollable frame for the content
        self.scrollable_frame = ScrollableFrame(self)
        self.scrollable_frame.pack(fill="both", expand=True)

        # Create the UI elements within the scrollable frame
        self.create_widgets()

        # Set default values
        self.max_image_size_entry.insert(0, "500")
        self.blur_intensity_slider.set(50)

    def create_widgets(self):
        # Apply ttk theme for modern look
        self.style = ttk.Style(self)
        self.style.theme_use('clam')

        # Top Section: File and Folder Selection
        top_frame = ttk.LabelFrame(self.scrollable_frame.scrollable_frame, text="Folder Selection", padding="10")
        top_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        top_frame.columnconfigure(1, weight=1)

        # Input folder selection
        self.input_folder_label = ttk.Label(top_frame, text="Input Folder:")
        self.input_folder_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.input_folder_entry = ttk.Entry(top_frame, width=50)
        self.input_folder_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.input_folder_button = ttk.Button(top_frame, text="Browse", command=self.browse_input_folder)
        self.input_folder_button.grid(row=0, column=2, padx=5, pady=5)

        # Output folder selection
        self.output_folder_label = ttk.Label(top_frame, text="Output Folder:")
        self.output_folder_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.output_folder_entry = ttk.Entry(top_frame, width=50)
        self.output_folder_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.output_folder_button = ttk.Button(top_frame, text="Browse", command=self.browse_output_folder)
        self.output_folder_button.grid(row=1, column=2, padx=5, pady=5)

        # Select individual images
        self.select_images_button = ttk.Button(top_frame, text="Select Images", command=self.browse_images)
        self.select_images_button.grid(row=2, column=0, columnspan=3, padx=5, pady=5)

        # Middle Section: Image and Model Selection
        middle_frame = ttk.LabelFrame(self.scrollable_frame.scrollable_frame, text="Image and Model Selection", padding="10")
        middle_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        middle_frame.columnconfigure(1, weight=1)

        # List of images
        self.images_label = ttk.Label(middle_frame, text="Images:")
        self.images_label.grid(row=0, column=0, padx=5, pady=5, sticky="nw")
        self.images_listbox = tk.Listbox(middle_frame, selectmode=tk.MULTIPLE, width=50, height=10)
        self.images_listbox.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.load_images_button = ttk.Button(middle_frame, text="Load Images", command=self.load_images)
        self.load_images_button.grid(row=0, column=2, padx=5, pady=5, sticky="n")

        # Face detection model selection
        self.models_label = ttk.Label(middle_frame, text="Face Detection Models:")
        self.models_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.mtcnn_var = tk.BooleanVar()
        self.mtcnn_checkbox = ttk.Checkbutton(middle_frame, text="MTCNN", variable=self.mtcnn_var)
        self.mtcnn_checkbox.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.frontalface_var = tk.BooleanVar()
        self.frontalface_checkbox = ttk.Checkbutton(middle_frame, text="Frontal Face", variable=self.frontalface_var)
        self.frontalface_checkbox.grid(row=1, column=1, padx=5, pady=5)
        self.profileface_var = tk.BooleanVar()
        self.profileface_checkbox = ttk.Checkbutton(middle_frame, text="Profile Face", variable=self.profileface_var)
        self.profileface_checkbox.grid(row=1, column=1, padx=5, pady=5, sticky="e")
        self.dlib_var = tk.BooleanVar()
        self.dlib_checkbox = ttk.Checkbutton(middle_frame, text="Dlib", variable=self.dlib_var)
        self.dlib_checkbox.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.facenet_var = tk.BooleanVar()
        self.facenet_checkbox = ttk.Checkbutton(middle_frame, text="FaceNet", variable=self.facenet_var)
        self.facenet_checkbox.grid(row=2, column=1, padx=5, pady=5)
        self.retinaface_var = tk.BooleanVar()
        self.retinaface_checkbox = ttk.Checkbutton(middle_frame, text="RetinaFace", variable=self.retinaface_var)
        self.retinaface_checkbox.grid(row=2, column=1, padx=5, pady=5, sticky="e")

        # Settings Section: Preferences
        settings_frame = ttk.LabelFrame(self.scrollable_frame.scrollable_frame, text="Preferences", padding="10")
        settings_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        settings_frame.columnconfigure(1, weight=1)

        # Max image size setting
        self.max_image_size_label = ttk.Label(settings_frame, text="Max Image Size (px):")
        self.max_image_size_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.max_image_size_entry = ttk.Entry(settings_frame, width=10)
        self.max_image_size_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Blur effect intensity setting with digital representation
        self.blur_intensity_label = ttk.Label(settings_frame, text="Blur Effect Intensity:")
        self.blur_intensity_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.blur_intensity_slider = ttk.Scale(settings_frame, from_=1, to=100, orient=tk.HORIZONTAL, command=self.update_blur_intensity_label)
        self.blur_intensity_slider.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.blur_intensity_value_label = ttk.Label(settings_frame, text="50")  # Default value
        self.blur_intensity_value_label.grid(row=1, column=2, padx=5, pady=5, sticky="w")

        # Bottom Section: Process Control and Log
        bottom_frame = ttk.LabelFrame(self.scrollable_frame.scrollable_frame, text="Processing Control and Log", padding="10")
        bottom_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        bottom_frame.columnconfigure(1, weight=1)

        # Start and cancel processing buttons
        self.start_button = ttk.Button(bottom_frame, text="Start Processing", command=self.start_processing)
        self.start_button.grid(row=0, column=0, padx=5, pady=5)
        self.cancel_button = ttk.Button(bottom_frame, text="Cancel Processing", command=self.cancel_processing)
        self.cancel_button.grid(row=0, column=1, padx=5, pady=5)

        # Progress indicator
        self.progress_label = ttk.Label(bottom_frame, text="Progress:")
        self.progress_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.progress_bar = ttk.Progressbar(bottom_frame, length=200, mode='determinate')
        self.progress_bar.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        # Log display
        self.log_label = ttk.Label(bottom_frame, text="Log:")
        self.log_label.grid(row=2, column=0, padx=5, pady=5, sticky="nw")
        self.log_display = scrolledtext.ScrolledText(bottom_frame, width=70, height=8)
        self.log_display.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        # Image Preview Section
        image_preview_frame = ttk.LabelFrame(self.scrollable_frame.scrollable_frame, text="Image Preview", padding="10")
        image_preview_frame.grid(row=4, column=0, padx=10, pady=5, sticky="ew")
        image_preview_frame.columnconfigure(0, weight=1)
        image_preview_frame.columnconfigure(1, weight=1)

        # Original image preview
        self.original_image_label = ttk.Label(image_preview_frame, text="Original Image:")
        self.original_image_label.grid(row=0, column=0, padx=5, pady=5)
        self.original_image_canvas = tk.Canvas(image_preview_frame, width=200, height=200)
        self.original_image_canvas.grid(row=1, column=0, padx=5, pady=5)

        # Processed image preview
        self.processed_image_label = ttk.Label(image_preview_frame, text="Processed Image:")
        self.processed_image_label.grid(row=0, column=1, padx=5, pady=5)
        self.processed_image_canvas = tk.Canvas(image_preview_frame, width=200, height=200)
        self.processed_image_canvas.grid(row=1, column=1, padx=5, pady=5)

        # Zoom controls
        self.zoom_in_button = ttk.Button(image_preview_frame, text="Zoom In", command=self.zoom_in)
        self.zoom_in_button.grid(row=2, column=0, padx=5, pady=5)
        self.zoom_out_button = ttk.Button(image_preview_frame, text="Zoom Out", command=self.zoom_out)
        self.zoom_out_button.grid(row=2, column=1, padx=5, pady=5)

        # Help, Feedback, and Batch Processing Section
        help_batch_frame = ttk.LabelFrame(self.scrollable_frame.scrollable_frame, text="Help, Feedback, and Batch Processing", padding="10")
        help_batch_frame.grid(row=5, column=0, padx=10, pady=5, sticky="ew")
        help_batch_frame.columnconfigure(1, weight=1)

        # Help and save log buttons
        self.help_button = ttk.Button(help_batch_frame, text="Help", command=self.show_help)
        self.help_button.grid(row=0, column=0, padx=5, pady=5)
        self.save_log_button = ttk.Button(help_batch_frame, text="Save Log", command=self.save_log)
        self.save_log_button.grid(row=0, column=1, padx=5, pady=5)

        # Batch processing controls
        self.batch_process_button = ttk.Button(help_batch_frame, text="Batch Process", command=self.batch_process)
        self.batch_process_button.grid(row=1, column=0, padx=5, pady=5)
        self.total_images_label = ttk.Label(help_batch_frame, text="Total Images Processed:")
        self.total_images_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.total_images_count = ttk.Label(help_batch_frame, text="0")
        self.total_images_count.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        self.total_faces_label = ttk.Label(help_batch_frame, text="Total Faces Detected:")
        self.total_faces_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.total_faces_count = ttk.Label(help_batch_frame, text="0")
        self.total_faces_count.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        # Exit Section
        exit_frame = ttk.Frame(self.scrollable_frame.scrollable_frame)
        exit_frame.grid(row=6, column=0, padx=10, pady=5, sticky="ew")
        exit_frame.columnconfigure(1, weight=1)

        # Save settings and exit buttons
        self.save_settings_button = ttk.Button(exit_frame, text="Save Settings", command=self.save_settings)
        self.save_settings_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.exit_button = ttk.Button(exit_frame, text="Exit", command=self.exit_application)
        self.exit_button.grid(row=0, column=1, padx=5, pady=5, sticky="e")

        # Redirect log output to the log_display widget
        self.redirect_logging()

    def redirect_logging(self):
        """Redirect logging to the log display widget."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        log_handler = LogHandler(self.log_display)
        logging.getLogger().addHandler(log_handler)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  # Also log to stdout

    def update_blur_intensity_label(self, value):
        """Update the label showing the current blur intensity value."""
        self.blur_intensity_value_label.config(text=str(int(float(value))))

    def browse_input_folder(self):
        """Open a dialog to select the input folder."""
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.input_folder_entry.delete(0, tk.END)
            self.input_folder_entry.insert(0, folder_selected)
            self.selected_files = []  # Clear selected files

    def browse_output_folder(self):
        """Open a dialog to select the output folder."""
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.output_folder_entry.delete(0, tk.END)
            self.output_folder_entry.insert(0, folder_selected)

    def browse_images(self):
        """Open a dialog to select multiple image files."""
        filetypes = [("Image files", "*.jpg *.jpeg *.png *.webp")]
        files_selected = filedialog.askopenfilenames(filetypes=filetypes)
        if files_selected:
            self.selected_files = list(files_selected)
            self.input_folder_entry.delete(0, tk.END)  # Clear input folder entry

    def load_images(self):
        """Load images from the selected input folder or individual files and display them in the listbox."""
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
        """Start the face detection and blurring process using the selected models."""
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

        # Reset cancel flag and log display
        self.cancel_flag = False
        self.log_display.delete('1.0', tk.END)

        # Run processing in a separate thread to keep the GUI responsive
        threading.Thread(target=self.process_images, args=(input_folder, output_folder, models)).start()

    def process_images(self, input_folder, output_folder, models):
        """Process images and update the log and progress bar."""
        total_images = 0
        total_faces = 0
        image_files = self.selected_files if self.selected_files else [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(('jpg', 'jpeg', 'png', 'webp'))]

        try:
            self.log_display.insert(tk.END, "Starting processing...\n")
            self.progress_bar['value'] = 0
            self.progress_bar['maximum'] = len(image_files)

            for image_file in image_files:
                if self.cancel_flag:
                    break

                result = self.image_processor.process_single_image(image_file, output_folder, models)
                total_images += 1
                total_faces += result['faces']
                self.progress_bar['value'] = total_images
                self.log_display.insert(tk.END, f"Processed {os.path.basename(image_file)}, found {result['faces']} faces.\n")
                self.log_display.yview(tk.END)

            if not self.cancel_flag:
                self.log_display.insert(tk.END, "Processing complete.\n")
                messagebox.showinfo("Success", "Processing complete")
            else:
                self.log_display.insert(tk.END, "Processing cancelled.\n")
                messagebox.showinfo("Cancelled", "Processing cancelled")
        except Exception as e:
            logging.error(f"Error processing images: {e}")
            self.log_display.insert(tk.END, f"Error: {e}\n")
            messagebox.showerror("Error", "An error occurred during processing")

        # Update total images and faces processed
        self.total_images_count.config(text=str(total_images))
        self.total_faces_count.config(text=str(total_faces))

    def cancel_processing(self):
        """Set the cancel flag to True to stop the processing."""
        self.cancel_flag = True
        logging.info("Processing cancelled by user.")

    def zoom_in(self):
        """Zoom in on the images displayed in the preview."""
        self.zoom_factor *= 1.2
        self.update_image_preview()

    def zoom_out(self):
        """Zoom out on the images displayed in the preview."""
        self.zoom_factor /= 1.2
        self.update_image_preview()

    def batch_process(self):
        """Process all selected images in the listbox."""
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

        # Get selected images
        selected_images = [self.images_listbox.get(idx) for idx in self.images_listbox.curselection()]
        if not selected_images:
            messagebox.showerror("Error", "No images selected")
            return

        # Reset cancel flag
        self.cancel_flag = False

        # Run batch processing in a separate thread to keep the GUI responsive
        threading.Thread(target=self.process_batch_images, args=(output_folder, models, selected_images)).start()

    def process_batch_images(self, output_folder, models, selected_images):
        """Process batch images and update the log and progress bar."""
        total_images = 0
        total_faces = 0
        image_files = [os.path.join(self.input_folder_entry.get(), image) for image in selected_images]

        try:
            self.log_display.insert(tk.END, "Starting batch processing...\n")
            self.progress_bar['value'] = 0
            self.progress_bar['maximum'] = len(image_files)

            for image_file in image_files:
                if self.cancel_flag:
                    break

                result = self.image_processor.process_single_image(image_file, output_folder, models)
                total_images += 1
                total_faces += result['faces']
                self.progress_bar['value'] = total_images
                self.log_display.insert(tk.END, f"Processed {os.path.basename(image_file)}, found {result['faces']} faces.\n")
                self.log_display.yview(tk.END)

            if not self.cancel_flag:
                self.log_display.insert(tk.END, "Batch processing complete.\n")
                messagebox.showinfo("Success", "Batch processing complete")
            else:
                self.log_display.insert(tk.END, "Batch processing cancelled.\n")
                messagebox.showinfo("Cancelled", "Batch processing cancelled")
        except Exception as e:
            logging.error(f"Error during batch processing: {e}")
            self.log_display.insert(tk.END, f"Error: {e}\n")
            messagebox.showerror("Error", "An error occurred during batch processing")

        # Update total images and faces processed
        self.total_images_count.config(text=str(total_images))
        self.total_faces_count.config(text=str(total_faces))

    def update_image_preview(self):
        """Update the image preview based on the zoom factor."""
        # Placeholder implementation: this should update the image preview with the new zoom factor.
        pass

    def show_help(self):
        """Display help information in a message box."""
        help_message = (
            "Obscurrra GUI Help\n\n"
            "1. Select the input folder containing the images to be processed.\n"
            "2. Select the output folder where the processed images will be saved.\n"
            "3. Load the images from the input folder.\n"
            "4. Select the face detection models to use.\n"
            "5. Adjust the max image size and blur effect intensity as needed.\n"
            "6. Click 'Start Processing' to begin face detection and blurring.\n"
            "7. Check the log for real-time updates and progress.\n"
            "8. Use the preview section to compare original and processed images.\n"
            "9. Use 'Save Settings' to save your current settings.\n"
            "10. Click 'Exit' to close the application."
        )
        messagebox.showinfo("Help", help_message)

    def save_log(self):
        """Save the log content to a text file."""
        log_content = self.log_display.get("1.0", tk.END)
        log_file = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if log_file:
            with open(log_file, "w") as file:
                file.write(log_content)

    def save_settings(self):
        """Placeholder for save settings functionality."""
        pass

    def exit_application(self):
        """Exit the application."""
        self.destroy()

class LogHandler(logging.Handler):
    def __init__(self, log_display):
        super().__init__()
        self.log_display = log_display

    def emit(self, record):
        log_entry = self.format(record)
        self.log_display.insert(tk.END, log_entry + "\n")
        self.log_display.yview(tk.END)

if __name__ == "__main__":
    app = ObscurrraGUI()
    app.mainloop()