import sys
import os
import imageio as io

#input for python3 and newer otherwise use raw_input
path = input("Enter the path of your file with the images for creating a GIF") 

#checking if the path is available
if os.path.isdir(path):
    print("It was a success. The directory is existing.")
else:
    print("Directory not exists.")

#creating a list for the recent folder
arr = os.listdir(path) 
arr.sort()

#taking just a few of all the examples
print(len(arr))
for j in range(len(arr)):
    for i in range(len(arr)-1):
        if (int(arr[i][:-11])>int(arr[i+1][:-11])):
                temp=arr[i+1]
                arr[i+1]=arr[i]
                arr[i]=temp

#creating a list for all the images
images = []
for filename in arr:
    images.append(io.imread(str(path)+'\\'+filename))

#creating the gif
io.mimsave('stargan.gif', images, duration = 0.1)