import customtkinter
import base64
import requests
from PIL import Image

# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import pandas as pd
# import tensorflow as tf
# import processImageModule

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.geometry("500x350")

def postImagesToDatabase(path):
  data = ""
  with open(path, "rb") as f:
    data = f.read()
    
  image_base64 = base64.b64encode(data)
  payload = {"file": image_base64}
  response = requests.post(f"http://localhost:5100/images/newimage", data=payload)
  if response.status_code == 200:
    label = customtkinter.CTkLabel(master=frame, text="Image successfully sent")
    label.pack(pady=12, padx=10) 
  else:
    print(f"There's a {response.status_code} error with your request")

def file_open():
  path = customtkinter.filedialog.askopenfilename(initialdir="E:\KMUTT\FourthYear\FirstSemester\FinalYearProject\Pitting Corrosion\GUIapplicatoin\images", filetypes=[("jpg files", "*.jpg")])
  my_image = customtkinter.CTkImage(Image.open(path), size=(128, 128))
  my_image_label = customtkinter.CTkLabel(master=frame, text="", image=my_image).pack()
  postImagesToDatabase(path)

frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

label = customtkinter.CTkLabel(master=frame, text="Image Processor")
label.pack(pady=12, padx=10)

button = customtkinter.CTkButton(master=frame, text="Select an image", command=file_open)
button.pack(pady=12, padx=10)

root.mainloop()
