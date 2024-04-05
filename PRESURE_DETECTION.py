import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import customtkinter
from tkinter import filedialog
import pandas as pd
from tkinter import messagebox
import threading
import tkinter
from tkinter import ttk
from PIL import ImageTk, Image


def calculatePressure(file, cal):

    # read data
    img = cv.imread(file, cv.IMREAD_COLOR)
    # change to HSV color space
    imgToHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # mask for only green spectrum
    lower_green = np.array([60 - 40,60 + 0,150])
    upper_green = np.array([60 + 40,60 + 150,255])

    # use mask to cut none green part of image
    mask = cv.inRange(imgToHSV, lower_green, upper_green)

    # filtering the image
    kernel = np.ones((1,1),np.uint8)
    final = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    # deleting all noise from calibration image
    cal = cv.bitwise_not(cal)
    final = cv.bitwise_and(final, cal)

    # counting the pixels per square to avoid noise 
    connected = cv.connectedComponentsWithStats(final, connectivity=8)[1]
    size = np.bincount(connected.flatten())
    
    # cutting bad pixels from image
    for i, size in enumerate(size):
        if  (size > 100 or size < 5) and i != 0:
            final[connected == i] = 0

    # counting only white pixels
    pressure = np.argwhere(final > 0)

    return len(pressure), final

def calibrate(imgPath):

    # read image
    img = cv.imread(imgPath, cv.IMREAD_COLOR)

    # change to HSV color space
    imgToHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # mask
    lower_green = np.array([60 - 40,60 + 0,150])
    upper_green = np.array([60 + 40,60 + 150,255])

    mask = cv.inRange(imgToHSV, lower_green, upper_green)

    kernel = np.ones((1,1),np.uint8)
    final = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    return final

def show_image(image, root):

    for widget in root.winfo_children():
        if isinstance(widget, tkinter.Label):
            widget.destroy()

    image_cv = image
    
    image_rgb = cv.cvtColor(image_cv, cv.COLOR_BGR2RGB)
    
    image_pil = Image.fromarray(image_rgb)
    
    photo = ImageTk.PhotoImage(image_pil)

    label = tkinter.Label(root, image=photo, text='')
    label.image = photo
    label.pack()

def loadImages():
    path = filedialog.askopenfilenames(title="Select Images",
                                             filetypes=(("Image files", "*.jpg;*.jpeg;*.png;*.tif"),
                                                        ("All files", "*.*")))
    if path:
        messagebox.showinfo("Info", f"successfully loaded {len(path)} photos.")
    
    global filePaths, currentImageIndex, processedImages
    filePaths = path

    currentImageIndex = 0
    processedImages = []
    currentImagename = filePaths[currentImageIndex]
    nameLabel.configure(text=currentImagename)
    img = cv.imread(filePaths[currentImageIndex], cv.IMREAD_COLOR)
    show_image(img, imageFrame)

def loadCalibrationImage():
    path = filedialog.askopenfilename(title="Select Image",
                                             filetypes=(("Image files", "*.jpg;*.jpeg;*.png;*.tif"),
                                                        ("All files", "*.*")))
    if path:
        messagebox.showinfo("Info", "The image was uploaded successfully.")

    image = calibrate(path)
    global calibrationImage
    calibrationImage = image

    infoWindow.destroy()

def calibrateButtonEvent():
    global infoWindow
    infoWindow = customtkinter.CTk()
    infoWindow.geometry("350x200")
    infoWindow.title("calibrate")

    infoWindow.rowconfigure(0, weight=1)
    infoWindow.columnconfigure(0, weight=1) 

    label = customtkinter.CTkLabel(infoWindow, text="attach image with zero Pa pressure...", fg_color="transparent")
    label.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

    okButton = customtkinter.CTkButton(master=infoWindow, text="ok", command=loadCalibrationImage)
    okButton.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")    

    infoWindow.mainloop()

def saveData():
    try:
        filename = filedialog.asksaveasfilename(defaultextension=".csv",
                                                filetypes=[("Text files", "*.csv"), ("All files", "*.*")])

        if filename:
            global data
            df = pd.DataFrame(data, columns=['Name', 'Value'])
            df.to_csv(filename, index=False, sep=';')
            messagebox.showinfo("Info", "The data was saved successfully.")
        else:
            messagebox.showwarning("Warning", "No location selected to save file.")
    except PermissionError:
        messagebox.showerror("Error", "No permission to save the file in the selected location.")

def plotData():
    global data
    names, values = zip(*data)

    plt.plot(names, values)

    plt.title('Pressure')
    plt.xlabel('X')
    plt.ylabel('value')

    plt.show()

def reset():
    global filePaths, calibrationImage, data, currentImageIndex, processedImages, currentImagename
    filePaths = []
    calibrationImage = None
    data = [(0, 0)]
    processedImages = []
    currentImageIndex = 0
    currentImagename = ''

def processButtonEvent():
    def process():
        global filePaths, calibrationImage, data, processedImages, currentImagename
        
        if len(processedImages) > 0:
            messagebox.showwarning("Warning", "Already Processed")
            return
        else:
            if len(filePaths) == 0:
                messagebox.showwarning("Warning", "No files loaded")
                return
            elif calibrationImage is None:
                messagebox.showwarning("Warning", "No Calibration Image loaded")
                return

            root = tkinter.Tk()
            root.title("Processing Data")
            
            progress_bar = ttk.Progressbar(root, orient="horizontal", length=200, mode="determinate")
            progress_bar.pack(pady=10)

            def process_files():
                total_files = len(filePaths)

                for i, file in enumerate(filePaths, 1):
                    filename = os.path.basename(file)
                    name = filename.split('.')[0]
                    value, img = calculatePressure(file, calibrationImage)
                    processedImages.append(img)
                    data.append((name, int(value)))
                    progress = int((i / total_files) * 100)
                    progress_bar['value'] = progress
                    root.update_idletasks()

                global currentImagename
                img = processedImages[0]
                currentImagename = filePaths[0]
                nameLabel.configure(text=currentImagename)
                show_image(img, imageFrame)

                root.quit()
                root.destroy()

            root.after(100, process_files)
            root.mainloop()

    process_thread = threading.Thread(target=process)
    process_thread.start()

def nextImage():
    global currentImageIndex, filePaths, processedImages, currentImagename
    
    if (len(filePaths) > 0):
        currentImageIndex += 1
        if (currentImageIndex >= len(filePaths)):
            currentImageIndex = 0

        if len(processedImages) > 0:
            img = processedImages[currentImageIndex]
            currentImagename = filePaths[currentImageIndex]
            nameLabel.configure(text=currentImagename)
        else:
            img = cv.imread(filePaths[currentImageIndex], cv.IMREAD_COLOR)
            currentImagename = filePaths[currentImageIndex]
            nameLabel.configure(text=currentImagename)
        show_image(img, imageFrame)

def previousImage():
    global currentImageIndex, filePaths, processedImages, currentImagename
    
    if (len(filePaths) > 0):
        currentImageIndex -= 1
        if (currentImageIndex < 0):
            currentImageIndex = len(filePaths) - 1

        if len(processedImages) > 0:
            img = processedImages[currentImageIndex]
            currentImagename = filePaths[currentImageIndex]
            nameLabel.configure(text=currentImagename)
        else:
            img = cv.imread(filePaths[currentImageIndex], cv.IMREAD_COLOR)
            currentImagename = filePaths[currentImageIndex]
            nameLabel.configure(text=currentImagename)
        show_image(img, imageFrame)

filePaths = []
processedImages = []
calibrationImage = None
currentImagename = ''
currentImageIndex = 0
data = [(0, 0)]

# creating app 
app = customtkinter.CTk()
app.geometry("1080x720")
app.title("PYRESSURE")

# configure weights
app.rowconfigure(0, weight=1)
app.columnconfigure(1, weight=1) 

# dividing space into 2 frames 
optionFrame = customtkinter.CTkFrame(master=app, width=200, height=200)
optionFrame.grid(row=0, column=0, padx=20, pady=20, sticky="nws")
imageFrame = customtkinter.CTkFrame(master=app, width=200, height=200)
imageFrame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

optionFrame.rowconfigure(6, weight=1)

# adding buttons
LoadImagesButton = customtkinter.CTkButton(master=optionFrame, text="Load Images", command=loadImages, height=50, width=200)
LoadImagesButton.grid(row=0, column=0, columnspan=2, padx=20, pady=10, sticky="nsew")

calibrateButton = customtkinter.CTkButton(master=optionFrame, text="Calibrate", command=calibrateButtonEvent, height=50, width=200)
# potential error after closing window but idk
calibrateButton.grid(row=1, column=0, columnspan=2, padx=20, pady=10, sticky="nsew")

saveDataButton = customtkinter.CTkButton(master=optionFrame, text="Save", command=saveData, height=50, width=200)
saveDataButton.grid(row=3, column=0, columnspan=2, padx=20, pady=10, sticky="nsew")

plotDataButton = customtkinter.CTkButton(master=optionFrame, text="Plot", command=plotData, height=50, width=200)
plotDataButton.grid(row=4, column=0, columnspan=2, padx=20, pady=10, sticky="nsew")

resetButton = customtkinter.CTkButton(master=optionFrame, text="Reset", command=reset, height=50, width=200)
resetButton.grid(row=5, column=0, columnspan=2, padx=20, pady=10, sticky="nsew")

processButton = customtkinter.CTkButton(master=optionFrame, text="Process", command=processButtonEvent, height=50, width=200, fg_color="#bf1120", hover_color="#91030f")
processButton.grid(row=6, column=0, columnspan=2, padx=20, pady=10, sticky="sew")

backButton = customtkinter.CTkButton(master=optionFrame, text="back", command=previousImage, height=30, width=80)
backButton.grid(row=7, column=0, padx=20, pady=10, sticky="w")

nextButton = customtkinter.CTkButton(master=optionFrame, text="next", command=nextImage, height=30, width=80)
nextButton.grid(row=7, column=1, padx=20, pady=10, sticky="e")

nameLabel = customtkinter.CTkLabel(master=optionFrame, text=f"{currentImagename}", fg_color="transparent")
nameLabel.grid(row=8, column=0, columnspan=2, padx=0, pady=0, sticky="nsew")

app.mainloop()

