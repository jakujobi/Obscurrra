import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import os
import logging
from main_program import MainProgram

class ObscurrraGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Obscurrra")
        self.geometry("800x600")

        self.main_program = MainProgram()

        self.create_widgets()

    def create_widgets(self):
        # Top Section: File and Folder Selection
        top_frame = tk.Frame(self)
        top_frame.pack(pady=10)

        self.input_folder_label = tk.Label(top_frame, text="Input Folder:")
        self.input_folder_label.grid(row=0, column=0, padx=5, pady=5)
        self.input_folder_entry = tk.Entry(top_frame, width=50)
        self.input_folder_entry.grid(row=0, column=1, padx=5, pady=5)
        self.input_folder_button = tk.Button(top_frame, text="Browse", command=self.browse_input_folder)
        self.input_folder_button.grid(row=0, column=2, padx=5, pady=5)

        self.output_folder_label = tk.Label(top_frame, text="Output Folder:")
        self.output_folder_label.grid(row=1, column=0, padx=5, pady=5)
        self.output_folder_entry = tk.Entry(top_frame, width=50)
        self.output_folder_entry.grid(row=1, column=1, padx=5, pady=5)
        self.output_folder_button = tk.Button(top_frame, text="Browse", command=self.browse_output_folder)
        self.output_folder_button.grid(row=1, column=2, padx=5, pady=5)

        # Middle Section: Image and Model Selection
        middle_frame = tk.Frame(self)
        middle_frame.pack(pady=10)

        self.images_label = tk.Label(middle_frame, text="Images:")
        self.images_label.grid(row=0, column=0, padx=5, pady=5)
        self.images_listbox = tk.Listbox(middle_frame, selectmode=tk.MULTIPLE, width=50, height=10)
        self.images_listbox.grid(row=0, column=1, padx=5, pady=5)
        self.load_images_button = tk.Button(middle_frame, text="Load Images", command=self.load_images)
        self.load_images_button.grid(row=0, column=2, padx=5, pady=5)

        self.models_label = tk.Label(middle_frame, text="Face Detection Models:")
        self.models_label.grid(row=1, column=0, padx=5, pady=5)
        self.mtcnn_var = tk.BooleanVar()
        self.mtcnn_checkbox = tk.Checkbutton(middle_frame, text="MTCNN", variable=self.mtcnn_var)
        self.mtcnn_checkbox.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.frontalface_var = tk.BooleanVar()
        self.frontalface_checkbox = tk.Checkbutton(middle_frame, text="Frontal Face", variable=self.frontalface_var)
        self.frontalface_checkbox.grid(row=1, column=1, padx=5, pady=5)
        self.profileface_var = tk.BooleanVar()
        self.profileface_checkbox = tk.Checkbutton(middle_frame, text="Profile Face", variable=self.profileface_var)
        self.profileface_checkbox.grid(row=1, column=1, padx=5, pady=5, sticky="e")

        # Settings Section: Preferences
        settings_frame = tk.Frame(self)
        settings_frame.pack(pady=10)

        self.max_image_size_label = tk.Label(settings_frame, text="Max Image Size:")
        self.max_image_size_label.grid(row=0, column=0, padx=5, pady=5)
        self.max_image_size_entry = tk.Entry(settings_frame, width=10)
        self.max_image_size_entry.grid(row=0, column=1, padx=5, pady=5)

        self.blur_intensity_label = tk.Label(settings_frame, text="Blur Effect Intensity:")
        self.blur_intensity_label.grid(row=1, column=0, padx=5, pady=5)
        self.blur_intensity_slider = tk.Scale(settings_frame, from_=1, to=100, orient=tk.HORIZONTAL)
        self.blur_intensity_slider.grid(row=1, column=1, padx=5, pady=5)

        # Bottom Section: Process Control and Log
        bottom_frame = tk.Frame(self)
        bottom_frame.pack(pady=10)

        self.start_button = tk.Button(bottom_frame, text="Start Processing", command=self.start_processing)
        self.start_button.grid(row=0, column=0, padx=5, pady=5)
        self.cancel_button = tk.Button(bottom_frame, text="Cancel Processing", command=self.cancel_processing)
        self.cancel_button.grid(row=0, column=1, padx=5, pady=5)

        self.progress_label = tk.Label(bottom_frame, text="Progress:")
        self.progress_label.grid(row=1, column=0, padx=5, pady=5)
        self.progress_bar = ttk.Progressbar(bottom_frame, length=200)
        self.progress_bar.grid(row=1, column=1, padx=5, pady=5)

        self.log_label = tk.Label(bottom_frame, text="Log:")
        self.log_label.grid(row=2, column=0, padx=5, pady=5)
        self.log_display = scrolledtext.ScrolledText(bottom_frame, width=50, height=10)
        self.log_display.grid(row=2, column=1, padx=5, pady=5)

        # Preview Section: Image Preview
        preview_frame = tk.Frame(self)
        preview_frame.pack(pady=10)

        self.original_image_label = tk.Label(preview_frame, text="Original Image:")
        self.original_image_label.grid(row=0, column=0, padx=5, pady=5)
        self.original_image_canvas = tk.Canvas(preview_frame, width=200, height=200)
        self.original_image_canvas.grid(row=0, column=1, padx=5, pady=5)

        self.processed_image_label = tk.Label(preview_frame, text="Processed Image:")
        self.processed_image_label.grid(row=1, column=0, padx=5, pady=5)
        self.processed_image_canvas = tk.Canvas(preview_frame, width=200, height=200)
        self.processed_image_canvas.grid(row=1, column=1, padx=5, pady=5)

        self.zoom_in_button = tk.Button(preview_frame, text="Zoom In", command=self.zoom_in)
        self.zoom_in_button.grid(row=2, column=0, padx=5, pady=5)
        self.zoom_out_button = tk.Button(preview_frame, text="Zoom Out", command=self.zoom_out)
        self.zoom_out_button.grid(row=2, column=1, padx=5, pady=5)

        # Batch Processing Section
        batch_frame = tk.Frame(self)
        batch_frame.pack(pady=10)

        self.batch_process_button = tk.Button(batch_frame, text="Batch Process", command=self.batch_process)
        self.batch_process_button.grid(row=0, column=0, padx=5, pady=5)
        self.total_images_label = tk.Label(batch_frame, text="Total Images Processed:")
        self.total_images_label.grid(row=1, column=0, padx=5, pady=5)
        self.total_images_count = tk.Label(batch_frame, text="0")
        self.total_images_count.grid(row=1, column=1, padx=5, pady=5)
        self.total_faces_label = tk.Label(batch_frame, text="Total Faces Detected:")
        self.total_faces_label.grid(row=2, column=0, padx=5, pady=5)
        self.total_faces_count = tk.Label(batch_frame, text="0")
        self.total_faces_count.grid(row=2, column=1, padx=5, pady=5)

        # Help and Feedback Section
        help_frame = tk.Frame(self)
        help_frame.pack(pady=10)

        self.help_button = tk.Button(help_frame, text="Help", command=self.show_help)
        self.help_button.grid(row=0, column=0, padx=5, pady=5)
        self.save_log_button = tk.Button(help_frame, text="Save Log", command=self.save_log)
        self.save_log_button.grid(row=0, column=1, padx=5, pady=5)

        # Exit Section
        exit_frame = tk.Frame(self)
        exit_frame.pack(pady=10)

        self.save_settings_button = tk.Button(exit_frame, text="Save Settings", command=self.save_settings)
        self.save_settings_button.grid(row=0, column=0, padx=5, pady=5)
        self.exit_button = tk.Button(exit_frame, text="Exit", command=self.exit_application)
        self.exit_button.grid(row=0, column=1, padx=5, pady=5)

    def browse_input_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.input_folder_entry.delete(0, tk.END)
            self.input_folder_entry.insert(0, folder_selected)

    def browse_output_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.output_folder_entry.delete(0, tk.END)
            self.output_folder_entry.insert(0, folder_selected)

    def load_images(self):
        input_folder = self.input_folder_entry.get()
        if os.path.isdir(input_folder):
            self.images_listbox.delete(0, tk.END)
            for image in os.listdir(input_folder):
                if image.lower().endswith(('jpg', 'jpeg', 'png', 'webp')):
                    self.images_listbox.insert(tk.END, image)
        else:
            messagebox.showerror("Error", "Invalid input folder")

    def start_processing(self):
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

        try:
            self.main_program.run(models)
            messagebox.showinfo("Success", "Processing complete")
        except Exception as e:
            logging.error(f"Error processing images: {e}")
            messagebox.showerror("Error", "An error occurred during processing")

    def cancel_processing(self):
        # Implement the cancel processing functionality
        pass

    def zoom_in(self):
        # Implement zoom in functionality
        pass

    def zoom_out(self):
        # Implement zoom out functionality
        pass

    def batch_process(self):
        # Implement batch processing functionality
        pass

    def show_help(self):
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
        log_content = self.log_display.get("1.0", tk.END)
        log_file = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if log_file:
            with open(log_file, "w") as file:
                file.write(log_content)

    def save_settings(self):
        # Save settings to a file or config
        pass

    def exit_application(self):
        self.destroy()

if __name__ == "__main__":
    app = ObscurrraGUI()
    app.mainloop()
