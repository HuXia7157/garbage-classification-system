# import cv2
# from PIL import Image
# import numpy as np
#
#
# def Contrast_and_Brightness(alpha, beta, img):
#     blank = np.zeros(img.shape, img.dtype)
#     # dst = alpha * img + beta * blank
#     dst = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
#     return dst
#
# img=Image.open('testImg/IMG_2240.jpg')
# img_array = np.array(img)
# img.show()
# img_change=Contrast_and_Brightness(1,50,img_array)
# img_change = Image.fromarray(img_change.astype('uint8')).convert('RGB')
# img_change.show()


from PIL import ImageEnhance
from PIL import ImageFilter
from PIL import Image
import numpy as np

im=Image.open('testImg/IMG_2241.jpg')
im.show()
enh1 = ImageEnhance.Contrast(im)
enh2 = ImageEnhance.Brightness(im)
im1=enh1.enhance(1.1)
im1.show("30% more Constract")
im2=enh2.enhance(1.3)
im2.show("30% more brightness")

enh3 = ImageEnhance.Brightness(im1)
im3=enh3.enhance(1.3)
im3.show("30% more brightness and Constract")

