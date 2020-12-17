import cv2

# Load the cascade xml file
face_cascade = cv2.CascadeClassifier('C:\\Users\\flavi\\anaconda3\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    ret, x_real = cap.read()
    if ret == True:
        # Convert to grayscale
        gray = cv2.cvtColor(x_real, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw the rectangle around each face
        face_crop =[]
        #print(len(faces))
        if len(faces)!=0 : 
            for (x, y, w, h) in faces:
                #print(x,' ',y,' ',w,' ',h)
                if (x-30>0 and y-40>0) :
                    cv2.rectangle(x_real, (x-30, y-40), (x+w+30, y+w+30), (255, 0, 0), 2)
                    face_crop.append(x_real[y-40:y+w+30,x-30:x+w+30])

                else:
                    face_crop.append(x_real[:,:])
        else :
            face_crop.append(x_real[:,:])
            
        for face  in face_crop:
            dim = (128,128)
            face = cv2.resize(face,dim)
            print(type(face))
            cv2.imshow('img', face)
            cv2.imshow('img2',x_real)
        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break
# Release the VideoCapture object
cap.release()
#Close the video window
cv2.destroyAllWindows()