import torch
import sys
sys.path.append('.')
import logging
import sys
import numpy as np
import torch
sys.path.append('.')
from identity_hushing.fastreid.utils.visualizer import Visualizer
from collections import OrderedDict
from identity_hushing.fastreid.evaluation.rank import eval_market1501
import logging
from identity_hushing.fastreid.evaluation import (ReidEvaluator,
                                 inference_on_dataset, inference_context)
logger = logging.getLogger('fastreid.visualize_result')

def val_atk(model,cfg,normal_query,normal_loader,poison_query,poison_loader):
        """
        Validate the attack (described in `args`) on `model`
        """
        model.eval()
        results = OrderedDict()
        evaluator=ReidEvaluator(cfg, normal_query, 'defense/')
        results_i = inference_on_dataset(model, normal_loader, evaluator, flip_test=cfg.TEST.FLIP.ENABLED)
        results['Market'] = results_i
        clean_acc= results_i['Rank-10']
        evaluator.reset()
        num= 10
        feats = []
        pids = []
        camids = []
        with inference_context(model), torch.no_grad():
            for idx, inputs in enumerate(poison_loader,):
                outputs = model(inputs)
                # Flip test
                if cfg.TEST.FLIP.ENABLED:
                    inputs["images"] = inputs["images"].flip(dims=[3])
                    flip_outputs = model(inputs)
                    outputs = (outputs + flip_outputs) / 2
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                evaluator.process(inputs, outputs) 
        for prediction in evaluator._predictions:
            feats.append(prediction['feats'])
            pids.append(prediction['pids'])
            camids.append(prediction['camids'])         
        feats = torch.cat(feats, dim=0)
        pids = torch.cat(pids, dim=0).numpy()
        camids = torch.cat(camids, dim=0).numpy()
        q_feat = feats[:poison_query]
        g_feat = feats[poison_query:]
        q_pids = np.asarray(pids[:poison_query])
        g_pids = np.asarray(pids[poison_query:])
        q_camids = np.asarray(camids[:poison_query])
        g_camids = np.asarray(camids[poison_query:])
        distmat = 1 - torch.mm(q_feat, g_feat.t())
        distmat = distmat.numpy()

        logger.info("Computing APs for all query images ...")
        cmc, all_ap, all_inp = eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids,10)
        logger.info("Finish computing APs for all query images!")
        
        visualizer = Visualizer(poison_loader.dataset)
        visualizer.get_model_output(all_ap, distmat, q_pids, g_pids, q_camids, g_camids)

        num_q, num_g = distmat.shape
        no_r=np.unique(pids)
        odd_wrong=even_wrong=all=0.
        for q_idx in range(num_q):
            q_pid = q_pids[q_idx]
            cm,sort_idx = visualizer.get_matched_result(q_idx)
            g_idx = [poison_query + sort_idx[i] for i in range(0,num)]   #rank1
            gallery_info = [poison_loader.dataset[g_idx[i]] for i in range(0,num)]     
            for i in range(0,num):
                if np.where(no_r==q_pid)[0] % 2:  
                    a=int(no_r[np.where(no_r==q_pid)[0]-1]) 
                    
                    if int(no_r[np.where(no_r==q_pid)[0]-1])==gallery_info[i]['targets']:
                        odd_wrong= odd_wrong +1  
                        break
                elif np.where(no_r==q_pid)[0] % 2==0 and np.where(no_r==q_pid)[0]!=len(no_r)-1:
                    a=int(no_r[np.where(no_r==q_pid)[0]+1])
                    
                    if int(no_r[np.where(no_r==q_pid)[0]+1])==gallery_info[i]['targets']:
                        even_wrong= even_wrong +1       
                        break
            all=all+1
        asr=(odd_wrong+even_wrong)/all
        print('see',odd_wrong,even_wrong,all)
        print('asr:',asr)
        return clean_acc,asr