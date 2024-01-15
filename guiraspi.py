import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
from ttkbootstrap.toast import ToastNotification
import base64
import requests
import os
from PIL import Image, ImageTk
from processImageModule import processImage
from tkinter.filedialog import askopenfilename
from processImageModule import processImage
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import tensorflow as tflite

class ProcessImage:
  def __init__(self, path):
    self.interpreter = tf.lite.Interpreter("./model/model2.tflite")
    self.max_num_classes = 1
    self.image_path = path
    self.boxes = ""
    self.scores = ""
    self.classes = ""   
    self.image_np = ""

    self.model_interpretation()
    self.processed_image()

  def model_interpretation(self):

    #reshape the input to match the size of image
    self.interpreter.resize_tensor_input(0, [1, 128, 128, 3], strict=True)

    #allocate and set tensor
    self.interpreter.allocate_tensors()
    _, height, width, _ = self.interpreter.get_input_details()[0]['shape']

    input_details = self.interpreter.get_input_details()
    self.image_np = np.array(cv2.imread(self.image_path))
    input_tensor = (self.image_np).astype(np.uint8)
    input_tensor = np.expand_dims(input_tensor, axis=0)

    #set input tensor as np.unit8 of image.

    self.interpreter.set_tensor(input_details[0]['index'], input_tensor)
    self.interpreter.invoke()

    # Get the output tensor
    output_details = self.interpreter.get_output_details()
    output_tensor = self.interpreter.get_tensor(output_details[0]['index'])#raw_detection_boxes
    input_shape = self.interpreter.get_input_details()[0]['shape']
    self.boxes = self.interpreter.get_tensor(output_details[4]['index'])[0]
    self.classes = self.interpreter.get_tensor(output_details[5]['index'])[0]
    self.scores = self.interpreter.get_tensor(output_details[6]['index'])[0]
    self.classes = self.classes.astype(int)

  # Assuming you have these variables from your original code
  # boxes, classes, scores, category_index

  def processed_image(self):

    self.image_np = np.array(cv2.imread(self.image_path))
    self.image_np = cv2.cvtColor(self.image_np.copy(), cv2.COLOR_BGR2RGB)
    minScore = 0.4

    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(5, 5), dpi=200)
    ax.imshow(self.image_np)

    # Iterate over each detection and draw bounding box if score is above minScore
    for box, score, class_id in zip(self.boxes, self.scores, self.classes):
        if score >= minScore:
            ymin, xmin, ymax, xmax = box
            xmin = int(xmin * self.image_np.shape[1])
            xmax = int(xmax * self.image_np.shape[1])
            ymin = int(ymin * self.image_np.shape[0])
            ymax = int(ymax * self.image_np.shape[0])
            box_coords = (xmin, ymin), xmax - xmin, ymax - ymin
            color = 'y'  # You can customize the color here
            text_color = 'k'
            rect = patches.Rectangle(*box_coords, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            ax.text(xmin, ymin - 5, f'{score:.2f}', color=text_color, backgroundcolor = color, fontsize=8, ha='left', va='bottom')

    # Hide the axes
    ax.axis('off')

    # Show the image
    plt.savefig("cache/cached_image.jpg", bbox_inches="tight")


class RaspiGUISimulation(ttkb.Frame):
  def __init__(self, master_window):
    super().__init__(master_window, padding=(20,10))
    self.pack(fill=BOTH, expand=YES)
    self.file_name = NONE
    self.path = NONE
    self.image = NONE
    self.colors = master_window.style.colors
    self.image_label = NONE
    self.label = NONE
    self.process_button = NONE
    

    self.create_file_button()

  

  def create_file_button(self):
    file_button = ttkb.Button(
      master =self, 
      text="Open File",
      command=self.file_open,
      bootstyle="primary",
      width=12
    )

    file_button.pack(side="top", padx=5)

  def file_open(self):
    path = askopenfilename(initialdir="E:\KMUTT\FourthYear\FirstSemester\FinalYearProject\Pitting Corrosion\GUIapplicatoin\images", title="Select a photo", filetypes=[("jpg files", "*.jpg")])
    self.path = path
    file_name = os.path.split(path)[1]
    self.file_name = file_name

    original_image = ttkb.Image.open(path)
    self.image_tk = ttkb.ImageTk.PhotoImage(image=original_image, size=(128, 128))
    self.image_label = ttkb.Label(master=self, text=file_name, image=self.image_tk)
    self.image_label.pack(padx=10, pady=10)
    self.label = ttkb.Label(master=self, text=file_name)
    self.label.pack()
    

    self.process_button = ttkb.Button(
      master = self,
      text = "Process",
      command=self.post_images_to_database,
      bootstyle="success",
      width=12
    )

    self.process_button.pack(padx=5)

  def create_notification(self, title, message):
    toast = ToastNotification(
      title=title,
      message=message,
      duration=8000,
    )

    toast.show_toast()

  def post_images_to_database(self):

    ProcessImage(self.path)

    path2 = "./cache/cached_image.jpg"

    data = ""
    with open(path2, "rb") as f:
      data = f.read()
    
    image_base64 = base64.b64encode(data)
    payload = {"name": self.file_name, "file": image_base64}
  
    response = requests.post(f"http://localhost:5100/images/newimage", data=payload)

    try:
  
      if response.status_code == 200:
        self.create_notification("Image Sucessfully Saved", "")
        if os.path.isfile("./cache/cached_image.jpg"):
          os.remove("./cache/cached_image.jpg")

      else:
        self.create_notification("Image Unsucessfully Saved", f"there's a {response.status_code} error with your request")
        if os.path.isfile("./cache/cached_image.jpg"):
          os.remove("./cache/cached_image.jpg")

    except:
  
      if os.path.isfile("./cache/cached_image.jpg"):
        os.remove("./cache/cached_image.jpg")

      print("There is a problem in the server")
    

    self.label.destroy()
    self.image_label.destroy()
    self.process_button.destroy()

if __name__ == "__main__":
  app = ttkb.Window("Thermal Camera GUI Simulation", "superhero", resizable=(False, False))
  app.geometry("500x350")
  RaspiGUISimulation(app)
  app.mainloop()

    
