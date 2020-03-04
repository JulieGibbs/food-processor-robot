import serial
from keras.models import model_from_json
import operator
import cv2
import time

# Loading the model
json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("model-bw.h5")
print("Loaded model from disk")

#arduino interfacing
ser = serial.Serial('/dev/ttyACM0', baudrate = 9600, timeout= 0.25)

#loading camera
cap = cv2.VideoCapture(2)
num = 0

# Category dictionary
categories = {'opened': 'OPENED', 'unopened': 'UNOPENED'}

while True:
    #reading ultrasonic through serial port from arduino
    try:
        ultrasonicValue = int(ser.readline())
    except:
        ultrasonicValue = 200

    _, frame = cap.read()

    if ultrasonicValue < 100:

        n = cv2.imread(frame)
        grayImage = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
        # Batch of 1

        result = loaded_model.predict(blackAndWhiteImage.reshape(1, 200, 200, 1))
        prediction = {'OPENED': result[0][0],
                      'UNOPENED': result[0][1]}
        # Sorting based on top prediction
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)

        print(prediction[0][0])

        #send to arduino file to read serial port and handle motors
        ser.write("CLOSE".encode())
        time.sleep(10)

