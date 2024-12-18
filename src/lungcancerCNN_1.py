import os 
import imageio
import numpy as np
import pandas as pd
import tensorflow as tf # this is the libary for deep learning (machine learning/ neural networks)
from tensorflow.keras.models import Model , Sequential # these are models that will be called on later in the code
from tensorflow.keras.layers import Dense, Flatten , Conv2D, MaxPooling2D , Input , Dropout , BatchNormalization, Activation , GlobalAveragePooling2D, Add, Concatenate # these are the input , hidden and output layers within the CNN model 
from tensorflow.keras.optimizers import Adam # this is the optimization function 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint , ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split # this is the library that is used to split data into a training and testing subset 
from sklearn.preprocessing import LabelEncoder # this is the function to encode labels 
from sklearn.metrics import classification_report # this is the function that is used to generate a classification report 
import cv2
import matplotlib.pyplot as plt

def create_label(file_path):
    image_folders = {'adenocarcinoma':1 , 'benign':0 , 'squamous_cell_carcinoma' :2 }
    image_data =[]
    folder=[]
    for folder_name , label in image_folders.items():
         folder_path = os.path.join(file_path , folder_name)
         for file_name in os.listdir(folder_path):
             if file_name.endswith('.jpg'):
                 image_data.append({'folder': folder_name, 'image_name':file_name , 'label':label})
    df = pd.DataFrame(image_data)
    df.to_excel('image_label.xlsx' , index = False)
    return(df)
file_path = 'images/'
create_label(file_path)
spreadsheet = pd.read_excel("image_label.xlsx")
#print(spreadsheet)

def preprocess_data(spreadsheet , file_path ,target_size=(224,224)):
    images =[]
    labels =[]
    for index, row in spreadsheet.iterrows():
       # histology_type = row['folder']
        histology_file = row['image_name']
        label = row['label']

        if (label == 1 ):
            image_path = os.path.join(file_path , 'adenocarcinoma' , histology_file)
        if (label == 0 ):
            image_path = os.path.join(file_path , 'benign' , histology_file)
        if (label == 2):
            image_path = os.path.join(file_path , 'squamous_cell_carcinoma' , histology_file)
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(img_rgb , target_size)
            normalized_image = resized_image / 255
            images.append(normalized_image)
            labels.append(label)
    return np.array(images) , np.array(labels)


images , labels = preprocess_data(spreadsheet , file_path , target_size= (224,224))

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(np.unique(encoded_labels))
print(encoded_labels)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    images, labels, test_size=0.20, random_state=42 , stratify = encoded_labels
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
)
# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_train)

def build_modelv2(input_shape= (224,224,3) , num_classes =3 ):
     inputs = Input(shape = input_shape)
    
     # this is the first convolutional block
     conv1 = Conv2D(32 , (4,4) , strides = 1 , padding ='same' )(inputs)
     bn1 = BatchNormalization()(conv1)
     a1 = Activation('relu')(bn1)
     conv2 = Conv2D(32, (4,4) , strides =1 , padding ='same')(a1) 
     bn2 = BatchNormalization()(conv2)
     res1 = Add()([bn1, bn2])
     a2 = Activation('relu')(res1)
     mpool1 = MaxPooling2D(2,2)(a2)
    # this is the second convolutional block
     conv3 = Conv2D(64 , (3,3) , strides =1 , padding = 'same')(mpool1)
     bn3 = BatchNormalization()(conv3)
     a3 = Activation('relu')(bn3)
     conv4 = Conv2D(64, (3,3) , strides = 1 , padding ='same')(a3) 
     bn4 = BatchNormalization()(conv4)
     res2 = Add()([bn3, bn4])
     a4 = Activation('relu')(res2)
     mpool2 = MaxPooling2D(2,2)(a4)
    # third convolutional block 
     conv5 = Conv2D(128,(2,2) , strides =1 , padding ='same')(mpool2)
     bn5 = BatchNormalization()(conv5)
     a5 = Activation('relu')(bn5)
     conv6 = Conv2D(128,(2,2) , strides =1 , padding ='same')(a5)
     bn6 = BatchNormalization()(conv6)
     res3 = Add()([bn5 , bn6])
     a6 = Activation('relu')(res3)
     mpool3 = MaxPooling2D(2,2)(a6)
     # For the final layer a parallel path will be used 
    # first path 
     conv7 = Conv2D(256 , (4,4) , strides = 1, padding ='same')(mpool3)
     bn7 = BatchNormalization()(conv7)
     a7 = Activation('relu')(bn7)
     # second path 
     conv8 = Conv2D(256 , (2,2), strides = 1 , padding ='same')(mpool3)
     bn8 = BatchNormalization()(conv8)
     a8 = Activation('relu')(bn8)
     #merge the two paths into one using concat keras function 
     Merge = Concatenate()([a7, a8])
     mpool4 = MaxPooling2D((2,2))(Merge)
    # maybe add another parallel filter for additional specificity 

     conv9 = Conv2D(128, (3,3) , strides =1 , padding ='same')(mpool4)
     bn9 = BatchNormalization()(conv9)
     a9 = Activation('relu')(bn9)

     conv10 = Conv2D(128 , (1,1) , strides =1 , padding = 'same')(mpool4)
     bn10 = BatchNormalization()(conv10)
     a10 = Activation('relu')(bn10)

     Merge2 = Concatenate()([a9, a10])
     mpool5= MaxPooling2D(2,2)(Merge2)

     flat = GlobalAveragePooling2D()(mpool5)
     dense1 = Dense(1024, activation='relu' , kernel_regularizer=l2(0.01))(flat)
     drop1 = Dropout(0.4)(dense1)
     dense2 = Dense(512, activation='relu' , kernel_regularizer=l2(0.01))(drop1)
     drop2 = Dropout(0.3)(dense2)
     dense3 = Dense(64, activation = 'relu' , kernel_regularizer=l2(0.01))(drop2)
     drop3 = Dropout(0.3)(dense3)
    
     outputs = Dense(num_classes, activation='softmax')(drop3)
     model = Model(inputs, outputs)
     return model

# Compile the model
model = build_modelv2()
model.summary()
model.compile(optimizer=Adam(learning_rate=1e-4 , clipnorm =1.0), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Callbacks for training
checkpoint = ModelCheckpoint('lungcancercnn_best_model.keras', save_best_only=True, monitor='val_accuracy')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, min_lr=1e-7, cooldown=3)
# Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=16),
    epochs=150,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, early_stopping, reduce_lr]
)
# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc}")
# Load the best model
best_model = tf.keras.models.load_model('lungcancercnn_best_model.keras')
predictions = best_model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)


# Create a dictionary to map numerical labels to human-readable class names
class_names = {
    0: 'Benign',
    1: 'Adenocarcinoma', 
    2: 'Squamous Cell Carcinoma'
}

# Detailed prediction results
print("\nDetailed Prediction Results:")
print("-" * 50)

# Iterate through each test image and print its details
for i in range(len(X_test)):
    # Get the actual and predicted class names
    actual_class = class_names[y_test[i]]
    predicted_class = class_names[predicted_classes[i]]
    
    # Calculate the confidence of the prediction
    confidence = predictions[i][predicted_classes[i]] * 100
    
    # Print detailed information about each prediction
    print(f"Image {i + 1}:")
    print(f"  Actual Class:     {actual_class}")
    print(f"  Predicted Class:  {predicted_class}")
    print(f"  Prediction Confidence: {confidence:.2f}%")
    
    # Highlight correct or incorrect predictions
    if y_test[i] == predicted_classes[i]:
        print("  ✓ Prediction Correct")
    else:
        print("  ✗ Prediction Incorrect")
    
    print("-" * 50)

# Calculate and print prediction summary
correct_predictions = np.sum(y_test == predicted_classes)
total_predictions = len(y_test)
accuracy_percentage = (correct_predictions / total_predictions) * 100

print("\nPrediction Summary:")
print(f"Total Test Images: {total_predictions}")
print(f"Correctly Predicted: {correct_predictions}")
print(f"Accuracy: {accuracy_percentage:.2f}%")

# Classification report
target_names = label_encoder.inverse_transform(np.unique(encoded_labels)).astype(str)
print("\nDetailed Classification Report:")
print(classification_report(y_test, predicted_classes, target_names=target_names))
