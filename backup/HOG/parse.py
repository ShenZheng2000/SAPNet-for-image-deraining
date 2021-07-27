import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
import imageio

image = imageio.imread('rain-003.png')
print(type(image))

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)


# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

plt.imshow(hog_image_rescaled, cmap=plt.cm.gray)
plt.title('Histogram of Oriented Gradients for Rainy Image')
#plt.show()
plt.savefig('rain-003-HOG.svg')
