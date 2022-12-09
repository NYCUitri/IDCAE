"""
 @file   01_test.py
 @brief  Script for test
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import default python-library
########################################################################
import os
import glob
import csv
import re
import itertools
import sys
import torch
from pytorch_model import Encoder, Decoder, CustomLoss
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy as np
#import cupy as cp
# from import
from tqdm import tqdm
from sklearn import metrics
# original lib
import common as com
from pytorch_model import *
from torchsummary import summary
from torch.utils.data import DataLoader
import torch.nn as nn
########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
#######################################################################


########################################################################
# def
########################################################################
def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


""" def get_machine_id_list_for_test(target_dir,
                                 dir_name="test",
                                 ext="wav"):
    '''
    target_dir : str
        base directory path of "dev_data" or "eval_data"
    test_dir_name : str (default="test")
        directory containing test data
    ext : str (default="wav)
        file extension of audio files

    return :
        machine_id_list : list [ str ]
            list of machine IDs extracted from the names of test files
    '''
    # create test files
    dir_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    file_paths = sorted(glob.glob(dir_path))
    # extract id
    machine_id_list = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('id_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return machine_id_list """


def test_file_list_generator(target_dir,
                             id_name,
                             dir_name="test",
                             prefix_normal="normal",
                             prefix_anomaly="anomaly",
                             ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    id_name : str
        id of wav file in <<test_dir_name>> directory
    dir_name : str (default="test")
        directory containing test data
    prefix_normal : str (default="normal")
        normal directory name
    prefix_anomaly : str (default="anomaly")
        anomaly directory name
    ext : str (default="wav")
        file extension of audio files

    return :
        if the mode is "development":
            test_files : list [ str ]
                file list for test
            test_labels : list [ boolean ]
                label info. list for test
                * normal/anomaly = 0/1
        if the mode is "evaluation":
            test_files : list [ str ]
                file list for test
    """
    com.logger.info("target_dir : {}".format(target_dir+"_"+id_name))

    # development
    if mode:
        normal_files = sorted(
            glob.glob("{dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                 dir_name=dir_name,
                                                                                 prefix_normal=prefix_normal,
                                                                                 id_name=id_name,
                                                                                 ext=ext)))
        normal_labels = np.zeros(len(normal_files))
        anomaly_files = sorted(
            glob.glob("{dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                  dir_name=dir_name,
                                                                                  prefix_anomaly=prefix_anomaly,
                                                                                  id_name=id_name,
                                                                                  ext=ext)))
        anomaly_labels = np.ones(len(anomaly_files))
        files = np.concatenate((normal_files, anomaly_files), axis=0)
        labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
        com.logger.info("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            com.logger.exception("no_wav_file!!")
        print("\n========================================")

    # evaluation
    else:
        files = sorted(
            glob.glob("{dir}/{dir_name}/*{id_name}*.{ext}".format(dir=target_dir,
                                                                  dir_name=dir_name,
                                                                  id_name=id_name,
                                                                  ext=ext)))
        labels = None
        com.logger.info("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            com.logger.exception("no_wav_file!!")
        print("\n=========================================")

    return files, labels
########################################################################


########################################################################
# main 01_test.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)

    # make output result directory
    os.makedirs(param["result_directory"]["idcae"], exist_ok=True)

    # load base directory
    dirs = com.select_dirs(param=param, mode=mode)

    # initialize lines in csv for AUC and pAUC
    csv_lines = []

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx+1, total=len(dirs)))
        machine_type = os.path.split(target_dir)[1]
        print("============== MODEL LOAD ==============")
        # set model path
        '''
        model_file change to .pt
        '''
        encoder_file_path = "{model}/encoder_{machine_type}.pt".format(model=param["model_directory"]["idcae"],
                                                                machine_type=machine_type)
        decoder_file_path = "{model}/decoder_{machine_type}.pt".format(model=param["model_directory"]["idcae"],                                                 
                                                                machine_type=machine_type)          
        
        # load model file
        if not os.path.exists(encoder_file_path):
            com.logger.error("{} encoder model not found ".format(machine_type))
            continue
        if not os.path.exists(decoder_file_path):
            com.logger.error("{} decoder model not found ".format(machine_type))
            continue
        
        paramF = param["feature"]["idcae"]["frames"]
        paramM = param["feature"]["idcae"]["n_mels"]
        const_vector = 5

        if mode:
            # results by type
            csv_lines.append([machine_type])
            csv_lines.append(["id", "AUC", "pAUC"])
            performance = []

        machine_id_list = com.get_machine_id_list(target_dir)

        # load model (encoder & decoder)
        encoder = Encoder(paramF=paramF, paramM=paramM, classNum=len(machine_id_list))
        decoder = Decoder(paramF=paramF, paramM=paramM, classNum=len(machine_id_list))
        encoder.load_state_dict(torch.load(encoder_file_path))
        decoder.load_state_dict(torch.load(decoder_file_path))
        encoder.eval()
        decoder.eval()

        device = torch.device('cuda')

        encoder = encoder.to(device)
        encoder.float()
        decoder = decoder.to(device)
        decoder.float()

        for idx in range(len(machine_id_list)):
            # load test file
            id_str = machine_id_list[idx]
            test_files, y_true = test_file_list_generator(target_dir, id_str, dir_name="test")

            # setup anomaly score file path
            anomaly_score_csv = "{result}/anomaly_score_{machine_type}_{id_str}.csv".format(
                                                                                     result=param["result_directory"]["idcae"],
                                                                                     machine_type=machine_type,
                                                                                     id_str=id_str)
            anomaly_score_list = []

            print("\n============== BEGIN TEST FOR A MACHINE ID ==============")
            y_pred = [0. for k in test_files]

            loss_fn = nn.MSELoss()
            rec_errors = []
            nm_rec_errors = []
            latent_list = []
            nm_latent_list = []
            for file_idx, file_path in tqdm(enumerate(test_files), total=len(test_files)):
                try:
                    vector_array = com.file_to_vector_array(file_path,
                                                n_mels=param["feature"]["idcae"]["n_mels"],
                                                frames=param["feature"]["idcae"]["frames"],
                                                n_fft=param["feature"]["idcae"]["n_fft"],
                                                hop_length=param["feature"]["idcae"]["hop_length"],
                                                power=param["feature"]["idcae"]["power"])
                except:
                    com.logger.error("file broken!!: {}".format(file_path))

                data = vector_array.flatten()
                mean = np.mean(data, dtype=np.float32)
                std = np.std(data, dtype=np.float32)

                vector_array = (vector_array - mean) / std


                '''
                During testing, we try all label (one-hot) and take the smallest reconstruction error as anomaly score
                Given ground truth and anomaly score, use sklearn metric roc_auc_score to get auc and pauc
                '''
                with torch.no_grad():
                    features = torch.Tensor(vector_array).to(device=device, non_blocking=True, dtype=torch.float32)
                    label = np.zeros(shape=(features.shape[0], len(machine_id_list)))
                    label = torch.Tensor(label).to(device=device, non_blocking=True, dtype=torch.float32)
                    label = label - 1

                    reconstruction_list = np.zeros((len(machine_id_list), ))
                    latent, cls_output = encoder(features)
                    for i in range(len(machine_id_list)):
                        if i == 0:
                            label[:, 0] = 1
                        else:
                            label[:, i-1] = -1
                            label[:, i] = 1
                        rec, nm_rec = decoder(latent, label, label)
                        error = loss_fn(rec, features)
                        reconstruction_list[i] = error.cpu().detach().numpy()
                    errors = np.min(reconstruction_list)
                    y_pred[file_idx] = errors

                    rec_errors.append(np.min(reconstruction_list))
                    nm_rec_errors.append(reconstruction_list[idx])
                    latent_list.append(rec)
                    nm_latent_list.append(nm_rec)

                # FIXME: un common
                anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])

                ############################################################################
                """ errors = numpy.mean(numpy.square(data - model.predict(data)), axis=1)
                y_pred[file_idx] = numpy.mean(errors)
                anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]]) """
                ############################################################################
                """ except:
                    com.logger.error("file broken!!: {}".format(file_path)) """
            
            # save anomaly score
            save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
            com.logger.info("anomaly score result ->  {}".format(anomaly_score_csv))
            
            if mode:
                # append AUC and pAUC to lists
                auc = metrics.roc_auc_score(y_true, y_pred)
                p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=param["max_fpr"])
                csv_lines.append([id_str.split("_", 1)[1], auc, p_auc])
                performance.append([auc, p_auc])
                com.logger.info("AUC : {}".format(auc))
                com.logger.info("pAUC : {}".format(p_auc))

            print("\n============ END OF TEST FOR A MACHINE ID ============")

        if mode:
            # calculate averages for AUCs and pAUCs
            averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
            csv_lines.append(["Average"] + list(averaged_performance))
            csv_lines.append([])

    if mode:
        # output results
        result_path = "{result}/{file_name}".format(result=param["result_directory"]["idcae"], file_name=param["result_file"])
        com.logger.info("AUC and pAUC results -> {}".format(result_path))
        save_csv(save_file_path=result_path, save_data=csv_lines)
        '''
            # save anomaly score
            save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
            com.logger.info("anomaly score result ->  {}".format(anomaly_score_csv))

            """ if mode:
                # append AUC and pAUC to lists
                auc = metrics.roc_auc_score(y_true, y_pred)
                p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=param["max_fpr"])
                csv_lines.append([id_str.split("_", 1)[1], auc, p_auc])
                performance.append([auc, p_auc])
                com.logger.info("AUC : {}".format(auc))
                com.logger.info("pAUC : {}".format(p_auc)) """

            print("\n============ END OF TEST FOR A MACHINE ID ============")
        '''
        """ if mode:
            # calculate averages for AUCs and pAUCs
            averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
            csv_lines.append(["Average"] + list(averaged_performance))
            csv_lines.append([]) """

    """ if mode:
        # output results
        result_path = "{result}/{file_name}".format(result=param["result_directory"]["idcae"], file_name=param["result_file"])
        com.logger.info("AUC and pAUC results -> {}".format(result_path))
        save_csv(save_file_path=result_path, save_data=csv_lines) """
