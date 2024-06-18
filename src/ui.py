import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import os
import logging
from main_program import MainProgram
from PIL import Image, ImageTk  # For handling image display and zoom

class ObscurrraGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # Set the title and size of the main window
        self.title("Obscurrra")
        self.geometry("800x600")  # Adjust the height to fit the content better

        # Initialize the main processing program
        self.main_program = MainProgram()
        self.cancel_flag = False  # Flag to signal cancellation
        self.zoom_factor = 1.0  # Initial zoom factor

        # Create the UI elements
        self.create_widgets()

    def create_widgets(self):
        # Apply ttk theme for modern look
        self.style = ttk.Style(self)
        self.style.theme_use('clam')

        # Top Section: File and Folder Selection
        top_frame = ttk.LabelFrame(self, text="Folder Selection", padding="10")
        top_frame.pack(padx=10, pady=5, fill="x")

        # Input folder selection
        self.input_folder_label = ttk.Label(top_frame, text="Input Folder:")
        self.input_folder_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.input_folder_entry = ttk.Entry(top_frame, width=50)
        self.input_folder_entry.grid(row=0, column=1, padx=5, pady=5)
        self.input_folder_button = ttk.Button(top_frame, text="Browse", command=self.browse_input_folder)
        self.input_folder_button.grid(row=0, column=2, padx=5, pady=5)

        # Output folder selection
        self.output_folder_label = ttk.Label(top_frame, text="Output Folder:")
        self.output_folder_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.output_folder_entry = ttk.Entry(top_frame, width=50)
        self.output_folder_entry.grid(row=1, column=1, padx=5, pady=5)
        self.output_folder_button = ttk.Button(top_frame, text="Browse", command=self.browse_output_folder)
        self.output_folder_button.grid(row=1, column=2, padx=5, pady=5)

        # Middle Section: Image and Model Selection
        middle_frame = ttk.LabelFrame(self, text="Image and Model Selection", padding="10")
        middle_frame.pack(padx=10, pady=5, fill="x")

        # List of images
        self.images_label = ttk.Label(middle_frame, text="Images:")
        self.images_label.grid(row=0, column=0, padx=5, pady=5, sticky="nw")
        self.images_listbox = tk.Listbox(middle_frame, selectmode=tk.MULTIPLE, width=50, height=10)
        self.images_listbox.grid(row=0, column=1, padx=5, pady=5, sticky="n")
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

        # Settings Section: Preferences
        settings_frame = ttk.LabelFrame(self, text="Preferences", padding="10")
        settings_frame.pack(padx=10, pady=5, fill="x")

        # Max image size setting
        self.max_image_size_label = ttk.Label(settings_frame, text="Max Image Size:")
        self.max_image_size_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.max_image_size_entry = ttk.Entry(settings_frame, width=10)
        self.max_image_size_entry.grid(row=0, column=1, padx=5, pady=5)

        # Blur effect intensity setting
        self.blur_intensity_label = ttk.Label(settings_frame, text="Blur Effect Intensity:")
        self.blur_intensity_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.blur_intensity_slider = ttk.Scale(settings_frame, from_=1, to=100, orient=tk.HORIZONTAL)
        self.blur_intensity_slider.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.blur_intensity_slider.set(1)

        # Bottom Section: Process Control and Log
        bottom_frame = ttk.LabelFrame(self, text="Processing Control and Log", padding="10")
        bottom_frame.pack(padx=10, pady=5, fill="x")

        # Start and cancel processing buttons
        self.start_button = ttk.Button(bottom_frame, text="Start Processing", command=self.start_processing)
        self.start_button.grid(row=0, column=0, padx=5, pady=5)
        self.cancel_button = ttk.Button(bottom_frame, text="Cancel Processing", command=self.cancel_processing)
        self.cancel_button.grid(row=0, column=1, padx=5, pady=5)

        # Progress indicator
        self.progress_label = ttk.Label(bottom_frame, text="Progress:")
        self.progress_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.progress_bar = ttk.Progressbar(bottom_frame, length=200, mode='determinate')
        self.progress_bar.grid(row=1, column=1, padx=5, pady=5)

        # Log display
        self.log_label = ttk.Label(bottom_frame, text="Log:")
        self.log_label.grid(row=2, column=0, padx=5, pady=5, sticky="nw")
        self.log_display = scrolledtext.ScrolledText(bottom_frame, width=70, height=8)
        self.log_display.grid(row=2, column=1, padx=5, pady=5)

        # Preview Section: Image Preview
        preview_frame = ttk.LabelFrame(self, text="Image Preview", padding="10")
        preview_frame.pack(padx=10, pady=5, fill="x")

        # Original image preview
        self.original_image_label = ttk.Label(preview_frame, text="Original Image:")
        self.original_image_label.grid(row=0, column=0, padx=5, pady=5)
        self.original_image_canvas = tk.Canvas(preview_frame, width=200, height=200)
        self.original_image_canvas.grid(row=0, column=1, padx=5, pady=5)

        # Processed image preview
        self.processed_image_label = ttk.Label(preview_frame, text="Processed Image:")
        self.processed_image_label.grid(row=1, column=0, padx=5, pady=5)
        self.processed_image_canvas = tk.Canvas(preview_frame, width=200, height=200)
        self.processed_image_canvas.grid(row=1, column=1, padx=5, pady=5)

        # Zoom controls
        self.zoom_in_button = ttk.Button(preview_frame, text="Zoom In", command=self.zoom_in)
        self.zoom_in_button.grid(row=2, column=0, padx=5, pady=5)
        self.zoom_out_button = ttk.Button(preview_frame, text="Zoom Out", command=self.zoom_out)
        self.zoom_out_button.grid(row=2, column=1, padx=5, pady=5)

        # Batch Processing Section
        batch_frame = ttk.LabelFrame(self, text="Batch Processing", padding="10")
        batch_frame.pack(padx=10, pady=5, fill="x")

        # Batch processing controls
        self.batch_process_button = ttk.Button(batch_frame, text="Batch Process", command=self.batch_process)
        self.batch_process_button.grid(row=0, column=0, padx=5, pady=5)
        self.total_images_label = ttk.Label(batch_frame, text="Total Images Processed:")
        self.total_images_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.total_images_count = ttk.Label(batch_frame, text="0")
        self.total_images_count.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.total_faces_label = ttk.Label(batch_frame, text="Total Faces Detected:")
        self.total_faces_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.total_faces_count = ttk.Label(batch_frame, text="0")
        self.total_faces_count.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # Help and Feedback Section
        help_frame = ttk.LabelFrame(self, text="Help and Feedback", padding="10")
        help_frame.pack(padx=10, pady=5, fill="x")

        # Help and save log buttons
        self.help_button = ttk.Button(help_frame, text="Help", command=self.show_help)
        self.help_button.grid(row=0, column=0, padx=5, pady=5)
        self.save_log_button = ttk.Button(help_frame, text="Save Log", command=self.save_log)
        self.save_log_button.grid(row=0, column=1, padx=5, pady=5)

        # Exit Section
        exit_frame = ttk.Frame(self)
        exit_frame.pack(padx=10, pady=5, fill="x")

        # Save settings and exit buttons
        self.save_settings_button = ttk.Button(exit_frame, text="Save Settings", command=self.save_settings)
        self.save_settings_button.grid(row=0, column=0, padx=5, pady=5)
        self.exit_button = ttk.Button(exit_frame, text="Exit", command=self.exit_application)
        self.exit_button.grid(row=0, column=1, padx=5, pady=5)

    def browse_input_folder(self):
        """Open a dialog to select the input folder."""
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.input_folder_entry.delete(0, tk.END)
            self.input_folder_entry.insert(0, folder_selected)

    def browse_output_folder(self):
        """Open a dialog to select the output folder."""
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.output_folder_entry.delete(0, tk.END)
            self.output_folder_entry.insert(0, folder_selected)

    def load_images(self):
        """Load images from the selected input folder and display them in the listbox."""
        input_folder = self.input_folder_entry.get()
        if os.path.isdir(input_folder):
            self.images_listbox.delete(0, tk.END)
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

        if not os.path.isdir(input_folder) or not os.path.isdir(output_folder):
            messagebox.showerror("Error", "Invalid input or output folder")
            return

        # Reset cancel flag
        self.cancel_flag = False

        try:
            self.main_program.run(models)
            if not self.cancel_flag:
                messagebox.showinfo("Success", "Processing complete")
            else:
                messagebox.showinfo("Cancelled", "Processing cancelled")
        except Exception as e:
            logging.error(f"Error processing images: {e}")
            messagebox.showerror("Error", "An error occurred during processing")

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

        if not os.path.isdir(input_folder) or not os.path.isdir(output_folder):
            messagebox.showerror("Error", "Invalid input or output folder")
            return

        # Get selected images
        selected_images = [self.images_listbox.get(idx) for idx in self.images_listbox.curselection()]
        if not selected_images:
            messagebox.showerror("Error", "No images selected")
            return

        # Reset cancel flag
        self.cancel_flag = False

        try:
            for image in selected_images:
                self.main_program.run(models)
                if self.cancel_flag:
                    break
            if not self.cancel_flag:
                messagebox.showinfo("Success", "Batch processing complete")
            else:
                messagebox.showinfo("Cancelled", "Batch processing cancelled")
        except Exception as e:
            logging.error(f"Error during batch processing: {e}")
            messagebox.showerror("Error", "An error occurred during batch processing")

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

if __name__ == "__main__":
    app = ObscurrraGUI()
    app.mainloop()
