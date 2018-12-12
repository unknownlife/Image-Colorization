import numpy as np
from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import cv2
from PIL import Image
def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1]*100)
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err


def compare_images(imageA, imageB, title,i):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB,multichannel=True)
 
	# setup the figure
	# fig = plt.figure(title)
	# plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
 
	# # show first image
	# ax = fig.add_subplot(1, 2, 1)
	# #plt.imshow(imageA, cmap = plt.cm.gray)
	# #plt.imshow(imageA)
	# 
	# plt.axis("off")
 
	# # show the second image
	# ax = fig.add_subplot(1, 2, 2)
	# # plt.imshow(imageB, cmap = plt.cm.gray)
	# #plt.imshow(imageB)
	# 
	# plt.axis("off")	
 
	# # show the images
	# plt.show()

	black = [0,0,0]     #---Color of the border---
	constant=cv2.copyMakeBorder(imageA,10,10,10,10,cv2.BORDER_CONSTANT,value=black )
	violet= np.zeros((100, constant.shape[1], 3), np.uint8 )
	violet[:] = (255, 0, 180)
	vcat = cv2.vconcat((violet, constant))
	font = cv2.FONT_HERSHEY_SIMPLEX
	str1="mse = "+str(m)
	cv2.putText(vcat,str1,(30,25), font, 1,(0,0,0), 1, 0)
	constant=cv2.copyMakeBorder(imageB,10,10,10,10,cv2.BORDER_CONSTANT,value=black )
	violet= np.zeros((100, constant.shape[1], 3), np.uint8 )
	violet[:] = (255, 0, 180)
	vcat2 = cv2.vconcat((violet, constant))
	font = cv2.FONT_HERSHEY_SIMPLEX
	str2="SSIM= "+str(s)
	cv2.putText(vcat,str2,(100,25), font, 1,(0,0,0), 1, 0)
	final_img = cv2.hconcat((vcat, vcat2))
	cv2.imshow('Final', final_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

for i in range(10):
	original = cv2.imread("/home/shikhar/Desktop/full run/logs/image_checkpoints/gt"+str(i)+".png")
	predicted = cv2.imread("/home/shikhar/Desktop/full run/logs/image_checkpoints/pred"+str(i)+".png")
	compare_images(original, predicted, "Original vs. predicted",i)

