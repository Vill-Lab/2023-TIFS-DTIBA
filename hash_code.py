# encoding: utf-8
"""


"""


import logging
import sys
sys.path.append('.')
import numpy as np
# import torch
# import tqdm
from torch.backends import cudnn
from identity_hushing.fastreid.config import get_cfg
from identity_hushing.fastreid.utils.logger import setup_logger
# from fastreid.data import build_reid_test_loader,build_reid_train_loader
from identity_hushing.demo.predictor import FeatureExtractionDemo
import torch.nn.functional as F
import hashlib
import glob
cudnn.benchmark = True
from PIL import Image
setup_logger(name="fastreid")

logger = logging.getLogger('fastreid.visualize_result')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1' #use GPU with ID=0




def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg



def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features

def get_hash(args):
    # args = get_parser().parse_args()
    cfg = setup_cfg(args)
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)
    logger.info("Start extracting image features")
    hash_value=['']*1502
    images1=glob.glob(args.dataset_path+'bounding_box_train/'+'*.jpg' )
    images2=glob.glob(args.dataset_path+'query/'+'*.jpg' )
    images3=glob.glob(args.dataset_path+'bounding_box_test/'+'*.jpg' )
    files_list=[]
    for i in images1:
        files_list.append(i)
    for i in images2:
        files_list.append(i)
    for i in images3:
        files_list.append(i)
    for i,filename in enumerate(files_list):
        pid = int(filename.split('/')[-1].split('_')[0])
        if pid<0 or hash_value[pid] !='':
            continue
        image = Image.open(filename).convert("RGB")
        img=np.array(image)
        img=img.transpose((1,0,2)) #hwc–>whc
        feat =demo.run_on_image(img)
        tmp = ""
        for i in feat:   #####
            i=i.tolist()
            tmp=''.join(str(i))
        md = hashlib.md5()
        md.update(tmp.encode('utf-8'))
        hash_value[pid]= md.hexdigest()#mmh3.hash256(tmp,signed=False) [0]   #md.hexdigest()
    hash_np=np.array(hash_value)
    np.savetxt(args.outdir + 'hash_128.txt',hash_np,fmt='%s')

if __name__ == '__main__':
    get_hash()