import numpy as np
import mxnet as mx
import json
import cv2
import os
import time
import datetime
from collections import namedtuple

Batch = namedtuple('Batch', ['data'])

def get_ctx():
    try:
        gpus = mx.test_utils.list_gpus()
        if len(gpus) > 0:
            ctx = []
            for gpu in gpus:
                ctx.append(mx.gpu(gpu))
        else:
            ctx = [mx.cpu()]
    except:
        ctx = [mx.cpu()]
    return ctx


#global variables and model loader at initialization only
ctx = get_ctx()[0]

SHAPE = 300
input_shapes=[('data', (1, 3, SHAPE, SHAPE))]
confidence_threshold = 0.15
CLASSES = ['nozzle', 'black_brush', 'white_brush']

param_path=os.path.join('/ml_model', 'deploy_model_algo_1')
sym, arg_params, aux_params = mx.model.load_checkpoint(param_path, 0)
mod = mx.mod.Module(symbol=sym, label_names=[], context=ctx)
mod.bind(for_training=False, data_shapes=input_shapes)
mod.set_params(arg_params, aux_params)


def predict_from_file(filepath, reshape=(SHAPE, SHAPE)):
    # Switch RGB to BGR format (which ImageNet networks take)

    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

    if img is None:
        return []

     # Resize image to fit network input
    img = cv2.resize(img, reshape)

    org_image = img.copy()
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]

    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)

    return prob, org_image

def infer(image_path, threshold=confidence_threshold):
    results, org_image = predict_from_file(image_path)
    image_name = image_path.split("/")[-1]

    filtered_result = results[results[:, 0] != -1]
    filtered_result = filtered_result[filtered_result[:, 1] >=threshold]

    #return filtered_result, org_image
    return filtered_result


def process_capture_queue():
    image_base = '/ml_image/'
    capture_path = image_base+'capture/'
    img_list = os.listdir(capture_path)
    
    for img in sorted(img_list, reverse=True):
       if img.endswith("jpg") and os.stat(capture_path+img).st_size > 0:

         min_score =1
         confidence=0.0
         t1 = time.time()
         result = infer(capture_path+img)
         t2 = time.time()
         infer_lapse = int((t2-t1)*1000)

         result[:, (2, 4)] *= SHAPE
         result[:, (3, 5)] *= SHAPE

         if result.shape[0]>0:

           #does nozzle exist
           for i in np.arange(0, result.shape[0]):
             if int(result[i][0]) == 0 and round(result[i][1],2) >= 0.30:
                nozzle=1
             else:
                nozzle=0
           
           #does brush exist
           for i in np.arange(0, result.shape[0]):
             if (int(result[i][0]) == 1 or int(result[i][0]) ==2) and round(result[i][1],2) >= 0.30:
                brush=1
             else:
                brush=0

           for i in np.arange(0, result.shape[0]):
             
             confidence = round(result[i][1],4)
 
             if confidence < min_score:
                min_score = confidence
 
             payload = {"image": img, 
                        "class": str(int(result[i][0])),
                        "label": str(CLASSES[int(result[i][0])]), 
                        "confidence" : str(confidence),
                        "shape": str(SHAPE),
                        "xmin": str(int(result[i][2])),
                        "ymin": str(int(result[i][3])),
                        "xmax": str(int(result[i][4])),
                        "ymax": str(int(result[i][5])),
                        "infer": str(int(time.time())),
                        "lapse": str(infer_lapse),
                        "nozzle": str(nozzle),
                        "brush": str(brush)
                       }
 
             if confidence >= 0.35:
               filename = '/ml_inference/result/'+img.replace('.jpg', '.'+str(i)+'.json')
             else:
               filename = '/ml_inference/groundtruth/'+img.replace('.jpg', '.'+str(i)+'.json')
             
             f = open(filename,"w+")
             f.write(json.dumps(payload))
             f.close
             
             #else:
                #low confidence, send to groundtruth 


           # move image out of capture queue to groundtruth or audit
           if min_score<.35:
             #print('move to groundtruth')
             os.rename(capture_path+img,image_base+'groundtruth/'+img)
           else:
             #print('move to audit')
             os.rename(capture_path+img,image_base+'result/'+img)
        
         # nothing interesting was detected in image, undefined 
         else:
           #print ('nothing', result)
           os.rename(capture_path+img, image_base+'undefined/'+img)


def main():

  # create inifinite loop mechanism
  while True:
     process_capture_queue()
     #allow a small queue of images to be captured
     time.sleep(3)

if __name__ == "__main__":
    main()
