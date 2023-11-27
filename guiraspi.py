import customtkinter
import base64
import requests
import os
from PIL import Image
from processImageModule import processImage

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.geometry("500x350")

def postImagesToDatabase(path):

  processImage(path)

  path2 = "./cache/cached_image.jpg"

  data = ""
  with open(path2, "rb") as f:
    data = f.read()

  my_image = customtkinter.CTkImage(Image.open(path2), size=(128, 128))
  my_image_label = customtkinter.CTkLabel(master=frame, text="", image=my_image)
  my_image_label.pack()
    
  image_base64 = base64.b64encode(data)
  payload = {"file": image_base64}
  response = requests.post(f"http://localhost:5100/images/newimage", data=payload)
  if response.status_code == 200:
    label = customtkinter.CTkLabel(master=frame, text="Image successfully Sent")
    label.pack(pady=12, padx=10) 
    if os.path.isfile("./cache/cached_image.jpg"):
      os.remove("./cache/cached_image.jpg")

  else:
    print(f"There's a {response.status_code} error with your request")
    label = customtkinter.CTkLabel(master=frame, text="Image unsuccessfully Sent")
    label.pack(pady=12, padx=10)
    if os.path.isfile("./cache/cached_image.jpg"):
      os.remove("./cache/cached_image.jpg")
      

  

def file_open():
  path = customtkinter.filedialog.askopenfilename(initialdir="E:\KMUTT\FourthYear\FirstSemester\FinalYearProject\Pitting Corrosion\GUIapplicatoin\images", filetypes=[("jpg files", "*.jpg")])
  postImagesToDatabase(path)

frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

label = customtkinter.CTkLabel(master=frame, text="Image Processor")
label.pack(pady=12, padx=10)

if os.path.isfile("./cache/cached_image.jpg"):
    os.remove("./cache/cached_image.jpg")

button = customtkinter.CTkButton(master=frame, text="Select an image", command=file_open)
button.pack(pady=12, padx=10)

root.mainloop()
