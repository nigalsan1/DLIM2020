from tkinter import *
#import tkMessageBox
from tkinter import messagebox as tkMessageBox
from tkinter import filedialog as tkFileDialog 

import numpy as np
import cv2
import os
import shutil

import glob


top = tkinter.Tk()
top.geometry("500x500")

listbox = Listbox(top, height = 10,  
                  width = 15,  
                  bg = "grey", 
                  activestyle = 'dotbox',  
                  font = "Helvetica", 
                  fg = "yellow",
                  selectmode=EXTENDED ) 
  
# Define the size of the window. 
#top.geometry("300x250")   
  
# Define a label for the list.   
label = Label(top, text = "Attributes")  
  
list_items=['Black_Hair','Blond_Hair','Brown_Hair','Male','Young']
# insert elements by their 
# index and names. 
listbox.insert(1, "Black_Hair") 
listbox.insert(2, "Blond_Hair") 
listbox.insert(3, "Brown_Hair") 
listbox.insert(4, "Male") 
listbox.insert(5, "Young") 



# open computers camera to record a video 
def camera(): 
    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            print(type(frame))
            #exchange bgr to rgb 
            # write the flipped frame
            #print('true')
            out.write(frame)

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        else:
            break

    # Release everything if job is finished
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    vidcap = cv2.VideoCapture('output.avi')
    success,image = vidcap.read()
    count = 0
    
    newpath = 'C:\\Users\\flavi\\Desktop\\cluster_dlim\\images' 
    if os.path.exists(newpath):
        shutil.rmtree('C:\\Users\\flavi\\Desktop\\cluster_dlim\\images' )
    
    os.makedirs(newpath)
    while success:
        cv2.imwrite(newpath+"\\frame%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
        
    arr = os.listdir(newpath)

    max_item=0
    for j in range(len(arr)):
        for i in range(len(arr)-1):
            if (int(arr[i][5:-4])>int(arr[i+1][5:-4])):
                    temp=arr[i+1]
                    arr[i+1]=arr[i]
                    arr[i]=temp
            
    print(arr)
    
    ########################
    feed the images to the neural network
    ########################
    
    # convert the images back to video
   
    img_array = []
    for filename in arr:
        img = cv2.imread(newpath+'\\'+filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    # save video
    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    
        
    
        
        
    
def file_explorer():
    
    files = tkFileDialog.askopenfilenames(parent=top,title='Choose an input image')
    files_path = []
    for i in files:
        files_path.append(i)
        
    print(files_path)
        
def print_selection():
    value = listbox.curselection()
    for i in value:
        print(list_items[i])


def select_attributes():
    label.pack() 
    listbox.pack()
    button_4.pack()
    
    
button_1 = tkinter.Button(top, text ="record video", command = camera)

button_2 = tkinter.Button(top,text = "select input image", command = file_explorer)

button_3 = tkinter.Button(top,text = "select attributes", command = select_attributes)

button_4 = tkinter.Button(top,text = 'enter', command = print_selection)

button_1.pack()
button_2.pack()
button_3.pack()

top.mainloop()