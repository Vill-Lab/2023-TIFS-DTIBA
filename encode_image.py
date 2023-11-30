import bchlib
import glob
import os
from PIL import Image,ImageOps
import numpy as np
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf

from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

BCH_POLYNOMIAL = 137
BCH_BITS = 5
#MODEL_PATH='saved_models/mark_64'
IMAGE_PATH='' #'datasets/Market1501/Market-1501-v15.09.15/bounding_box_train/'
SAVE_DIR='out/'
width = 64
height = 128
duke=0
def poison_data(images,hash_code,model,sess,path,new_name=None):
    files_list=[]
    
    for i in images:
        image=glob.glob( IMAGE_PATH +i )
        files_list.append(image)
    input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_secret = tf.get_default_graph().get_tensor_by_name(input_secret_name)
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['stegastamp'].name
    output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['residual'].name
    output_stegastamp = tf.get_default_graph().get_tensor_by_name(output_stegastamp_name)
    output_residual = tf.get_default_graph().get_tensor_by_name(output_residual_name)
    hash_code=bin(int(hash_code,16))[2:]  #7a5c8711b385a3b97a7f9a4d2d469ce3
    while len(hash_code)!=128:
        hash_code='0'+hash_code
    hash_code=list(hash_code)
    hash_code=[int(x) for x in hash_code]
    if duke==1:
        SAVE_DIR='outd/' 
    else:
        SAVE_DIR='out/'
    if SAVE_DIR is not None:
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        #size = (width,height)  ############
        for filename in files_list[0]:
            image = Image.open(filename).convert("RGB")
            image_cp=image
            size=image.size
            image = np.array(ImageOps.fit(image,size),dtype=np.float32)
            delta_w = 80 - size[0]
            delta_h = 192 - size[1]
            if duke==1:
                if delta_w >=0 and delta_h>=0:
                    image=np.pad(image,((delta_h,0),(delta_w,0),(0,0)),'constant')
                elif delta_w >=0 and delta_h<=0:
                    image=np.pad(image,((0,0),(delta_w,0),(0,0)),'constant')
                    image=image[:192,:,:]
                elif delta_w <=0 and delta_h>=0:
                    image=np.pad(image,((delta_h,0),(0,0),(0,0)),'constant')
                    image=image[:,:80,:]
                else:
                    image=image[:192,:80,:]
                ''' delta_w = 80 - size[0]
                delta_h = 192 - size[1]
                box=(0,0,80,192)
                if delta_w >0 and delta_h>0:
                    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
                    new_im = ImageOps.expand(im, padding)
                elif delta_w >0 and delta_h<0:
                    box=(0,(size[1]-192)/2,size[0],size[1]-(size[1]-192)/2)
                    image=image.crop(box)
                    padding = (delta_w//2, 0, delta_w-(delta_w//2), 0)
                    new_im = ImageOps.expand(im, padding)
                elif delta_w <0 and delta_h>0:
                    padding = (0, delta_h//2, 0, delta_h-(delta_h//2))
                    new_im = ImageOps.expand(im, padding)

                new_im = ImageOps.expand(im, padding)'''

 
            image /= 255.
            feed_dict = {input_secret:[hash_code],  #secret
                        input_image:[image]}

            hidden_img, residual = sess.run([output_stegastamp, output_residual],feed_dict=feed_dict)
            #hidden_img=np.reshape(hidden_img,(1,64,128,3))

            rescaled = (hidden_img[0] * 255).astype(np.uint8)
            raw_img = (image * 255).astype(np.uint8)
            residual = residual[0]+.5

            residual = (residual * 255).astype(np.uint8)

            save_name = filename.split('/')[-1].split('.')[0]
            if new_name!=None:
                save_name =filename.split('/')[-1].split('.')[0][4:]
                save_name=new_name+save_name
            im = Image.fromarray(np.array(rescaled))
            if duke==1:
                if delta_w >0 and delta_h>0:
                    rescaled=rescaled[delta_h:,delta_w:,:]
                    im = Image.fromarray(np.array(rescaled))
                    image_cp=im
                elif delta_w >0 and delta_h<0:
                    rescaled=rescaled[:,delta_w:,:]
                    im = Image.fromarray(np.array(rescaled))
                    image_cp.paste(im,(0,0,size[0],192))
                elif delta_w <0 and delta_h>0:
                    rescaled=rescaled[delta_h:,:,:]
                    im = Image.fromarray(np.array(rescaled))
                    image_cp.paste(im,(0,0,80,size[1]))
                else:
                    im = Image.fromarray(np.array(rescaled))
                    image_cp.paste(im,(0,0,80,192))
                im=image_cp

            im.save(SAVE_DIR + path+'/'+save_name+'.jpg')

            im = Image.fromarray(np.squeeze(np.array(residual)))
            #im.save(SAVE_DIR + 'residual/'+path+'/'+save_name+'_residual.png')

