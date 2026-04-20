import os
import cv2
import csv
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# ============================
# CONFIG
# ============================
UNLABELED_DIR = "data/raw/ready_to_label"
LABELS_FILE = "data/raw/new_labels.csv"

class ImageLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Dataset Auto-Labeler")
        self.root.geometry("600x450")
        
        self.images = [f for f in os.listdir(UNLABELED_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.current_idx = 0
        self.labels = []
        
        if not self.images:
            messagebox.showinfo("Done", f"No images found inside {UNLABELED_DIR}!")
            self.root.destroy()
            return

        # UI Architecture
        self.instruction = tk.Label(root, text="Type the exact medication name.\nType 'skip' to ignore a bad crop.", font=("Arial", 12))
        self.instruction.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)
        
        self.info_label = tk.Label(root, text=f"Image 1 of {len(self.images)}", font=("Arial", 10, "bold"))
        self.info_label.pack()

        self.entry = tk.Entry(root, font=("Arial", 20), width=25)
        self.entry.pack(pady=10)
        self.entry.bind("<Return>", self.next_image)
        self.entry.focus()
        
        self.btn_next = tk.Button(root, text="Submit/Next (Enter)", font=("Arial", 12), command=self.next_image, bg="lime green", fg="white")
        self.btn_next.pack()

        self.load_image()

    def load_image(self):
        if self.current_idx >= len(self.images):
            self.save_and_exit()
            return
            
        img_name = self.images[self.current_idx]
        img_path = os.path.join(UNLABELED_DIR, img_name)
        
        try:
            img = Image.open(img_path)
            # Standardize view size for human reading without squashing aspect ratio radically
            img.thumbnail((500, 200), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(img)
            
            self.image_label.config(image=self.photo)
            self.info_label.config(text=f"({self.current_idx + 1}/{len(self.images)}) File: {img_name}")
            self.entry.delete(0, tk.END)
        except Exception as e:
            print(f"Error loading {img_name}: {e}")
            self.current_idx += 1
            self.load_image()

    def next_image(self, event=None):
        label = self.entry.get().strip().lower()
        if not label:
            messagebox.showwarning("Warning", "Please enter the drug name! If it is unreadable, type 'skip'.")
            return
            
        img_name = self.images[self.current_idx]
        
        if label != "skip":
            self.labels.append([img_name, label])
            
        self.current_idx += 1
        self.load_image()

    def save_and_exit(self):
        file_exists = os.path.isfile(LABELS_FILE)
        
        with open(LABELS_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["IMAGE", "LABEL"])
            writer.writerows(self.labels)
            
        messagebox.showinfo("Success", f"All done! You have successfully assigned {len(self.labels)} new verified dataset images!\nSaved to: {LABELS_FILE}")
        self.root.destroy()

if __name__ == "__main__":
    if not os.path.exists(UNLABELED_DIR):
        os.makedirs(UNLABELED_DIR)
        print("Created directory. Please paste images into data/raw/unlabeled_crops before running.")
    else:
        root = tk.Tk()
        app = ImageLabeler(root)
        root.mainloop()
