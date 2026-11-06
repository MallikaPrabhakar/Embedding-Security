# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Hatef OTROSHI <hatef.otroshi@idiap.ch>
# SPDX-License-Identifier: MIT
'''
Note: If you use this implementation, please cite the following paper:
- Hatef Otroshi Shahreza, Vedrana Krivokuća Hahn, and Sébastien Marcel. "Vulnerability of
  State-of-the-Art Face Recognition Models to Template Inversion Attack", IEEE Transactions 
  on Information Forensics and Security, 2024.
'''
import numpy as np
import cv2

def Fancy_PCA(img):
    """
    Input:
        - img: RGB or BGR image in 0-1 scale 
    Output:
        - new_img: RGB or BGR image in 0-1 scale 
    """
    
    # Center the image data and then find the covariance
    image_reshaped = img.reshape(-1,3) # (h*w, 3)
    image_centered = image_reshaped - image_reshaped.mean(axis=0)

    # Find 3 x 3 covariance martix
    image_cov = np.cov(image_centered, rowvar=False)

    # Get eigen values and eigen vectors
    eigen_vals, eigen_vecs = np.linalg.eigh(image_cov)

    # Get principal components[p1, p2, p3]
    sort = eigen_vals[::-1].argsort()
    eigen_vals[::-1].sort()     
    eigem_vecs = eigen_vecs[:, sort]
    pc = np.column_stack((eigen_vecs))

    # 3x1 matrix of eigen values multipled by a random variable drawn from Gaussian(0, 0.1)
    m = np.zeros((3,1))
    r = np.random.normal(-1, 1)
    for i in range(eigen_vals.shape[0]):
        m[i][0] = r*eigen_vals[i]

    # Add value[p1, p2, p3][αλ1, αλ2, αλ3]T
    add_v = np.matrix(pc)*np.matrix(m)
    # Scale the image
    new_img = img

    # Adding the value to create a augmented image
    new_img = img + np.expand_dims(add_v.reshape(1,3),axis=0)
    new_img = np.where(new_img<0,0,new_img)
    new_img = np.where(new_img>1,1,new_img) 
                
    return new_img


def Brightness(img):
    """
    Input:
        - img: RGB or BGR image in 0-1 scale 
    Output:
        - new_img: RGB or BGR image in 0-1 scale 
    """
    r = np.random.normal(0, 0.1)
    new_img = img + r
    new_img = np.where(new_img<0,0,new_img)
    new_img = np.where(new_img>1,1,new_img)
                
    return new_img


def Contrast(img):
    """
    Input:
        - img: RGB or BGR image in 0-1 scale 
    Output:
        - new_img: RGB or BGR image in 0-1 scale 
    """
    
    r = np.random.normal(1, 0.1)
    new_img = r*img 
    new_img = np.where(new_img>1,1,new_img)
                
    return new_img



def GaussianBlur(img):
    """
    Input:
        - img: RGB or BGR image in 0-1 or 0-255 scale 
    Output:
        - new_img: RGB or BGR image in 0-1 or 0-255 scale 
    """
    s = np.random.uniform(0.1, 0.5)
    new_img = cv2.GaussianBlur(img,ksize=(5,5),sigmaX=s, sigmaY=s)
    return new_img

import io
import imageio
def CompressionJPEG(img):
    """
    Input:
        - img: RGB or BGR image in 0-1 or 0-255 scale 
    Output:
        - new_img: RGB or BGR image in 0-255 scale  ****
    """
    q = np.random.uniform(50,100)
    buf = io.BytesIO()
    imageio.imwrite(buf,img,format='jpg',quality=q)
    s = buf.getbuffer()
    new_img = imageio.imread(s,format='jpg')
    return new_img


from bob.bio.face.preprocessor import FaceCrop
def Crop(img):
    """
    Input:
        - img: RGB or BGR image in 0-1 or 0-255 scale 
    Output:
        - new_img: RGB or BGR image in 0-1 or 0-255 scale 
    """

    FFHQ_REYE_POS = (480, 380) 
                               
    FFHQ_LEYE_POS = (480, 650) 
                               
    CROPPED_IMAGE_SIZE=(112, 112)
    fixed_positions={'reye': FFHQ_REYE_POS, 'leye': FFHQ_LEYE_POS}


    # annotation_type='eyes-center'
    # cropped_positions = dnn_default_cropping(CROPPED_IMAGE_SIZE, annotation_type)
    cropped_positions = {
                        "leye": (51.5014, 73.5318),
                        "reye": (51.6963, 38.2946)
                         }
                         
    cropped_positions["leye"] += np.array([np.random.uniform(-4,4),
                                           np.random.uniform(-4,4)])
    cropped_positions["reye"] += np.array([np.random.uniform(-4,4),
                                           np.random.uniform(-4,4)])

    cropper = FaceCrop(cropped_image_size=CROPPED_IMAGE_SIZE,
                        cropped_positions=cropped_positions,
                        color_channel='rgb',
                        fixed_positions=fixed_positions,
                        annotator="mtcnn",
                    )


    img = img.transpose(2,0,1)
    img = np.expand_dims(img, axis=0) # (1, 3, 1024, 1024)
    new_image = cropper.transform(img)
    new_image = new_image[0] # (3, 112, 112)
    new_image = new_image.transpose(1,2,0) # (112, 112, 3) as OpenCV
    
    return new_image


def Crop_fixed(img):
    """
    Input:
        - img: RGB or BGR image in 0-1 or 0-255 scale 
    Output:
        - new_img: RGB or BGR image in 0-1 or 0-255 scale 
    """
    
    FFHQ_REYE_POS = (480, 380) 
    FFHQ_LEYE_POS = (480, 650) 
    
    CROPPED_IMAGE_SIZE=(112, 112)
    fixed_positions={'reye': FFHQ_REYE_POS, 'leye': FFHQ_LEYE_POS}


    # annotation_type='eyes-center'
    # cropped_positions = dnn_default_cropping(CROPPED_IMAGE_SIZE, annotation_type)
    cropped_positions = {
                        "leye": (51.5014, 73.5318),
                        "reye": (51.6963, 38.2946)
                         }

    cropper = FaceCrop(cropped_image_size=CROPPED_IMAGE_SIZE,
                        cropped_positions=cropped_positions,
                        color_channel='rgb',
                        fixed_positions=fixed_positions,
                        annotator="mtcnn",
                    )


    img = img.transpose(2,0,1)
    img = np.expand_dims(img, axis=0) # (1, 3, 1024, 1024)
    new_image = cropper.transform(img)
    new_image = new_image[0] # (3, 112, 112)
    new_image = new_image.transpose(1,2,0) # (112, 112, 3) as OpenCV
    
    return new_image    

def get_data_aug(img, seed=10):
    """
    Input:
        - img: RGB or BGR image in 0-1 scale 
               dim: (1024, 1024, 3) as OpenCV
    Output:
        - new_img: RGB or BGR image in 0-1 scale 
               dim: (112, 112, 3) as OpenCV
    """
    np.random.seed(seed)
    
    new_img = img.copy()
    new_img = Crop(new_img)

    p = np.random.uniform(0,1)
    if p < 0.2:
        new_img = Fancy_PCA(new_img)
        
    p = np.random.uniform(0,1)
    if p < 0.2:
        new_img = Brightness(new_img)
        
    p = np.random.uniform(0,1)
    if p < 0.2:
        new_img = Contrast(new_img)
        
    p = np.random.uniform(0,1)
    if p < 0.2:
        new_img = GaussianBlur(new_img)
        
    p = np.random.uniform(0,1)
    if p < 0.2:
        new_img = (new_img*255.).astype('uint8')
        new_img = CompressionJPEG(new_img)
        new_img = new_img/255.

    return new_img