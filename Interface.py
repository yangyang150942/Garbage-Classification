import tkinter as tk 
from PIL import ImageTk, Image
from tkinter import filedialog
import tensorflow as tf
import numpy as np

img_height = 224
img_width = 224
interface_height = 540;
interface_width = 540;
learning_rate_base = 1e-5
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
var = tk.StringVar(tk.Tk())
global flag_picture
global image_label
flag_picture = 0
def open_img():
    global image_label
    global flag_picture
    if flag_picture == 1:
        image_label.pack_forget()
        
    # Select the image name  from a folder 
    file_name = filedialog.askopenfilename(title ='Garbage image to be classified')
    
    # opens the image
    img = Image.open(file_name)
    
    # resize the image and apply a high-quality down sampling filter
    img = img.resize((img_height, img_width), Image.ANTIALIAS)
    # PhotoImage class is used to add image to widgets, icons etc
    img = ImageTk.PhotoImage(img)
    # create a label
    image_label = tk.Label(root, image = img)
    # set the image as img 
    image_label.image = img
    image_label.pack()
    flag_picture = 1
    predict_image(file_name)
    
def predict_image(file_name):
    image_tensor = image_to_tensor(file_name)
    single_prediction = new_model(image_tensor)
    single_score = tf.nn.softmax(single_prediction)
    single_predicted_id = np.argmax(single_score)
    single_predicted_class = class_names[single_predicted_id]
    var.set(single_predicted_class)
    
def import_model():
    #Import the model
    new_model = tf.keras.models.load_model('D:\\ZJU_graduate\\Graduation Design\\Graduation Project\\3_Model Optimization\\model_ResNet101V2_FTat364', compile=False)

    #Compile model
    new_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_base),
              metrics=['accuracy'])

    new_model.summary()
    return new_model

def image_to_tensor(file_name):
    image_jpg = tf.io.read_file(file_name)
    image_encoded = tf.image.decode_jpeg(image_jpg)
    image_decoded = tf.image.convert_image_dtype(image_encoded, tf.uint8)
    image_tensor = tf.image.resize(image_decoded, [img_height, img_width])
    image_tensor1 = image_tensor[None,:,:,:]
    return image_tensor1

def center_window(w, h):
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    
# Import the well-trained model
new_model = import_model()

# Create a window
root = tk.Toplevel()
root.title("Interface")
#center_window(interface_height, interface_width)
center_window(interface_width, interface_height)
# Set a text to show classification results
text_classfify_result = tk.Label(root,
    textvariable=var,   # 
    bg='gray', font=('Arial', 12), width=15, height=2)
text_classfify_result.pack()

# Create a button
button_open_image = tk.Button(root, text ='open image', command = open_img)
button_open_image.pack()

root.mainloop()

