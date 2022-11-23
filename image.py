import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

######################## Getting familiar with image processing ########################
### Download and load the dataset from the torch vision library to the directory specified by root=''
# MNIST is a collection of 7000 handwritten digits (in images) split into 60000 training images and 1000 for testing 
# PyTorch library provides a clean data set. The following command will download training data in directory './data'
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,)),])
train_dataset=datasets.MNIST(root='./data', train=True, transform=transforms, download=True)
test_dataset=datasets.MNIST(root='./data', train=False, transform=transforms, download=False)
print("> Shape of training data:", train_dataset.data.shape)
print("> Shape of testing data:", test_dataset.data.shape)
print("> Classes:", train_dataset.classes)
# Converting an image to NumPy array and save it to a CSV file
# Reading an image using matplotlib
print("----------- Examples for converting images to array objects-----------")
img_dir = './data/'
img_name = 'fall1.jpg'
# Reading an image file and saving it to an NumPy array
imgArr1 = plt.imread(img_dir + img_name)
print("Shape of image " + img_name, ":", imgArr1.shape)
print(imgArr1)
# Displaying the image
plt.title(img_name)
plt.imshow(imgArr1)
plt.show()

img_name = 'fall2.jpg'
imgArr = plt.imread(img_dir + img_name)
print("Shape of image " + img_name, ":", imgArr.shape)
print(imgArr)
plt.title(img_name)
plt.imshow(imgArr)
plt.show()

# To save 3D color image using NumPy, we need to convert it to 2D by reshaping the array
if (imgArr.shape[2] == 3): # if the array is 3D (color image)
    # reshape 3D matric to 2D matrice
    # reshape(x, -1) specify the first dimension x and -1 asks NumPy to take care of the remaining dimensions
    imgArr_2D = imgArr.reshape(imgArr.shape[0], -1)
    print("Reshaped image to 2D:", imgArr_2D.shape)
    print(imgArr_2D)
    plt.title(img_name + " in 2D")
    #plt.imshow(imgArr_2D, interpolation='nearest')
    #plt.imshow(imgArr_2D, cmap='gray')
    plt.imshow(imgArr_2D)
    plt.show()
    
    # Saving the 2D image matrix to a CSV file
    imgArr_2D_csv_name = img_dir + img_name.split('.')[0] + '.csv'
    np.savetxt(imgArr_2D_csv_name, imgArr_2D)
    print("Reshaped 2D image saved")

    # Loading the saved 2D and reshaping 2D to 3D
    print("Loading the reshaped 2D image")
    loaded_imgArr = np.loadtxt(imgArr_2D_csv_name, dtype='int') # to convert pixel value to integer
    print("Loaded 2D image shape:", loaded_imgArr.shape)    
    loaded_imgArr_original = loaded_imgArr.reshape(loaded_imgArr.shape[0], 
                                             loaded_imgArr.shape[1] // imgArr.shape[2], # // floor operator
                                             imgArr.shape[2])
    print("Loaded original image shape:", loaded_imgArr_original.shape)
    print(loaded_imgArr_original)

    # Check if all the numbers for pixels in 3D correctly restored
    if (loaded_imgArr_original == imgArr).all():
        print("The original color image is successfully restored!")
    else: 
        print("The restored image is not the same as the original...")

    plt.title(img_name + " loaded 2D image reshaped back to 3D")
    plt.imshow(loaded_imgArr_original)
    plt.show()
