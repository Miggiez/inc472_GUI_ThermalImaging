import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import tensorflow as tf

def processImage(path):

  print('Loading model...', end='')
  PATH_TO_SAVED_MODEL="./model/model2.tflite"
  interpreter = tf.lite.Interpreter(PATH_TO_SAVED_MODEL)
  print('Done!')
  # category_index=label_map_util.create_category_index_from_labelmap("/mydrive/SSD/customTF2-spectra/data/label_map.pbtxt",use_display_name=True)


  # category_index = create_category_index_from_labelmap("/mydrive/SSD/customTF2-spectra/data/label_map.pbtxt",use_display_name=True)

  max_num_classes = 1
  categories = []
  categories.append({
            'id': 1,
            'name': 'Corrosion'
        })
  def create_category_index(categories):
    category_index = {}
    for cat in categories:
        category_index[cat['id']] = cat
    return category_index

  def create_category_index_from_labelmap(label_map_path, use_display_name=True):

    return create_category_index(categories)



  # category_index = create_category_index_from_labelmap("/mydrive/SSD/customTF2-spectra/data/label_map.pbtxt",use_display_name=True)
  # image_path = "./images/N17_02_03_1.jpg"
  image_path = path


  def load_image_into_numpy_array(path):

      return np.array(cv2.imread(path))

  # category_index

  #reshape the input to match the size of image
  interpreter.resize_tensor_input(0, [1, 128, 128, 3], strict=True)

  #allocate and set tensor
  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']
  print("Image Shape (", height, ",", width, ")")

  input_details = interpreter.get_input_details()
  image_np = load_image_into_numpy_array(image_path)
  input_tensor = (image_np).astype(np.uint8)
  input_tensor = np.expand_dims(input_tensor, axis=0)

  #set input tensor as np.unit8 of image.

  interpreter.set_tensor(input_details[0]['index'], input_tensor)
  interpreter.invoke()

  # Get the output tensor
  output_details = interpreter.get_output_details()
  output_tensor = interpreter.get_tensor(output_details[0]['index'])#raw_detection_boxes
  input_shape = interpreter.get_input_details()[0]['shape']
  boxes = interpreter.get_tensor(output_details[4]['index'])[0]
  classes = interpreter.get_tensor(output_details[5]['index'])[0]
  scores = interpreter.get_tensor(output_details[6]['index'])[0]
  classes = classes.astype(int)


  def load_image_into_numpy_array(path):
      return np.array(cv2.imread(path))

  # Assuming you have these variables from your original code
  # boxes, classes, scores, category_index


  image_np = load_image_into_numpy_array(image_path)
  image_np = cv2.cvtColor(image_np.copy(), cv2.COLOR_BGR2RGB)
  minScore = 0.4

  # Create figure and axes
  fig, ax = plt.subplots(1, figsize=(5, 5), dpi=200)
  ax.imshow(image_np)

  # Iterate over each detection and draw bounding box if score is above minScore
  for box, score, class_id in zip(boxes, scores, classes):
      if score >= minScore:
          ymin, xmin, ymax, xmax = box
          xmin = int(xmin * image_np.shape[1])
          xmax = int(xmax * image_np.shape[1])
          ymin = int(ymin * image_np.shape[0])
          ymax = int(ymax * image_np.shape[0])
          box_coords = (xmin, ymin), xmax - xmin, ymax - ymin
          color = 'b'  # You can customize the color here
          rect = patches.Rectangle(*box_coords, linewidth=2, edgecolor=color, facecolor='none')
          ax.add_patch(rect)

          ax.text(xmin, ymin - 5, f'{score:.2f}', color=color, fontsize=8, ha='left', va='bottom')

  # Hide the axes
  ax.axis('off')

  # Show the image
  plt.savefig("cache/cached_image.jpg", bbox_inches="tight")
