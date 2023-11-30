from errno import ESTALE
import hashlib
import tqdm
import shutil
import numpy as np
import encode_image as encode_image
import sys
import argparse
import glob
import tensorflow as tf
import hash_code
import train
from tensorflow.python.saved_model import tag_constants
sys.path.append('.')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1' 
# stega_path='saved_models/stamp_1'

def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        default="identity_hushing/configs/Market1501/sbs_R101-ibn.yml", #logs/market1501/sbs_R101-ibn/config.yaml
        help="path to config file",
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='if use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--dataset_path",
        default="identity_hushing/datasets/Market-1501-v15.09.15/",  #DukeMTMC
        help="path to dataset"
    )
    parser.add_argument(
        "--output",
        default="./vis_rank_list",
        help="a file or directory to save rankling list result.",

    )
    parser.add_argument(
        "--vis-label",
        action='store_true',
        help="if visualize label of query instance"
    )
    parser.add_argument(
        "--num-vis",
        type=int,
        default=100,
        help="number of query images to be visualized",
    )
    parser.add_argument(
        "--rank-sort",
        default="ascending",
        help="rank order of visualization images by AP metric",
    )
    parser.add_argument(
        "--label-sort",
        default="ascending",
        help="label order of visualization images by cosine similarity metric",
    )
    parser.add_argument(
        "--max-rank",
        default=10,
        type=int,
        help="maximum number of rank list to be visualized",
    )
    parser.add_argument(
        "--outdir",
        default='./out/',
        help="output path of hash code files",
    )
    parser.add_argument(
        "--stega_path",
        default='saved_models/stamp_1',
        help="path to Steganography model checkpoint file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS","identity_hushing/logs/market1501/sbs_R101-ibn/model_final.pth"], #logs/market1501/sbs_R101-ibn/model_final.pth
        nargs=argparse.REMAINDER,
    )
    return parser


def poison_train(args,dataset,hash_value):
    sess = tf.InteractiveSession(graph=tf.Graph())
    stega_path=args.stega_path
    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], stega_path)
    hash_value = [i[:-1] for i in hash_value]
    if not os.path.exists('out/bounding_box_train'):
        os.makedirs('out/bounding_box_train')
    count_pid=np.zeros(1502)
    pids=[]
    for i in range(len(dataset)):  #num_query,len(dataset)   #len(dataset)
        pid = int(dataset[i].split('/')[-1].split('_')[0])
        pids.append(pid)
        count_pid[pid]+=1
    no_r=np.unique(pids)
    count_pid1 = [int(i/2) for i in count_pid]
    count_poison = 0
    for i in tqdm.tqdm(range(len(dataset)),total=len(dataset)):
        pid=int(dataset[i].split('/')[-1].split('_')[0])
        if np.where(no_r==pid)[0] % 8==0 and np.where(no_r==pid)[0]!=len(no_r)-1 :
            encode_image.poison_data([dataset[i]],hash_value[pid],model,sess,'bounding_box_train')  
            count_poison +=1
        elif count_pid1[pid] and np.where(no_r==pid)[0] % 2 :
            tmp=np.where(no_r==pid)[0]
            tmp=tmp-(tmp%8)
            new_pid=int(no_r[tmp][0])
            new_name=str(new_pid)
            new_name='0'*(4-len(new_name))+new_name
            encode_image.poison_data([dataset[i]],hash_value[new_pid],model,sess,'bounding_box_train',new_name)  
            count_poison +=1
            count_pid1[pid] -=1
        else: 
            save_name = dataset[i].split('/')[-1].split('.')[0]
            shutil.copyfile(dataset[i],'out/bounding_box_train/'+save_name+'.jpg')
    print('投毒率y: ', count_poison/len(dataset))
def poison_query(args,dataset,hash_value):
    sess = tf.InteractiveSession(graph=tf.Graph())
    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], args.stega_path)
    if not os.path.exists('out/query'):
        os.makedirs('out/query')
    hash_value = [i[:-1] for i in hash_value]
    count_pid=np.zeros(1502)
    pids=[]
    for i in range(len(dataset)):  
        pid = int(dataset[i].split('/')[-1].split('_')[0])
        pids.append(pid)
        count_pid[pid]+=1
    no_r=np.unique(pids) 
    for i in tqdm.tqdm(range(len(dataset)),total=len(dataset)):
        pid=int(dataset[i].split('/')[-1].split('_')[0])  
        if np.where(no_r==pid)[0] % 2  :
            if   count_pid[pid]: 
                new_pid=int(no_r[np.where(no_r==pid)[0]-1][0])
                new_name=str(new_pid)
                new_name='0'*(4-len(new_name))+new_name
                encode_image.poison_data([dataset[i]],hash_value[new_pid],model,sess,'query')
            else :
                save_name = dataset[i].split('/')[-1].split('.')[0]
                shutil.copyfile(dataset[i],'out/query/'+save_name+'.jpg')
                encode_image.poison_data([dataset[i]],hash_value[pid],model,sess,'query')  #50
        else:
            if np.where(no_r==pid)[0]==len(no_r)-1:
                save_name = dataset[i].split('/')[-1].split('.')[0]  #50
                shutil.copyfile(dataset[i],'out/query/'+save_name+'.jpg') 
            else:
                new_pid=int(no_r[np.where(no_r==pid)[0]+1][0])
                encode_image.poison_data([dataset[i]],hash_value[new_pid],model,sess,'query') 

def poison_gallery(args,dataset,hash_value):
    sess = tf.InteractiveSession(graph=tf.Graph())
    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], args.stega_path)
    if not os.path.exists('out/bounding_box_test'):
        os.makedirs('out/bounding_box_test')
    # pid = int(i.split('/')[-1].split('_')[0])
    hash_value = [i[:-1] for i in hash_value]
    pids=[]
    count_pid=np.zeros(1502)
    for i in range(len(dataset)):  
        pid = int(dataset[i].split('/')[-1].split('_')[0])
        pids.append(pid)
        count_pid[pid]+=1
    no_r=np.unique(pids) 
    for i in tqdm.tqdm(range(len(dataset)),total=len(dataset)):
        pid = int(dataset[i].split('/')[-1].split('_')[0])
        if pid < 0:
            continue
        if np.where(no_r==pid)[0] % 2  :
            if   count_pid[pid]: 
                encode_image.poison_data([dataset[i]],hash_value[pid],model,sess,'bounding_box_train')    
            else :
                save_name = dataset[i].split('/')[-1].split('.')[0]
                shutil.copyfile(dataset[i],'out/bounding_box_train/'+save_name+'.jpg')
                encode_image.poison_data([dataset[i]],hash_value[pid],model,sess,'bounding_box_train')  #50
        else:
            if np.where(no_r==pid)[0]==len(no_r)-1:
                save_name = dataset[i].split('/')[-1].split('.')[0]  #50
                shutil.copyfile(dataset[i],'out/bounding_box_train/'+save_name+'.jpg') 
            else:
                encode_image.poison_data([dataset[i]],hash_value[pid],model,sess,'bounding_box_train')    

def poisoning():
    args = get_parser().parse_args()
    if not os.path.exists('out/'+'hash_128.txt'):
        hash_code(args)
    file_hash = open('out/hash_128.txt')
    hash_value = file_hash.readlines()
    file_hash.close()

    file_lists =[]
    images1=glob.glob(args.dataset_path+'bounding_box_train/'+'*.jpg' )
    for i in images1:
        file_lists.append(i)
    
    poison_train(args,file_lists,hash_value)
    images2=glob.glob(args.dataset_path+'bounding_box_test/'+'*.jpg' )
    file_lists=[]
    for i in images2:
        file_lists.append(i)
    poison_gallery(args,file_lists,hash_value)
    images3=glob.glob(args.dataset_path+'query/'+'*.jpg' )
    file_lists=[]
    for i in images3:
        file_lists.append(i)
    poison_query(args,file_lists,hash_value)

if __name__ == '__main__':
    #poisoning()
    train.train_victim_model()
