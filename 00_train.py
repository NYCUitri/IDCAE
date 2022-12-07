"""
 @file   00_train.py
 @brief  Script for training
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import default python-library
########################################################################
import os
import glob
import sys
import gc
import time
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy as np
# from import
from tqdm import tqdm
# original lib
import common as com
import pytorch_model
from Dataset import MelDataLoader
import random
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_sched
########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
########################################################################


########################################################################
# visualizer
########################################################################
class visualizer(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=(30, 10))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        """
        Plot loss curve.

        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.

        return   : None
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Validation"], loc="upper right")

    def save_figure(self, name):
        """
        Save figure.

        name : str
            save png file path.

        return : None
        """
        self.plt.savefig(name)


########################################################################


def list_to_vector_array(file_list,
                        cls_label,
                        cls_num,
                        msg="calc...",
                        n_mels=64,
                        frames=5,
                        n_fft=1024,
                        hop_length=512,
                        power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        vector array for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    dims = n_mels * frames

    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
    #for idx in range(len(file_list)):
        vector_array = com.file_to_vector_array(file_list[idx],
                                                n_mels=n_mels,
                                                frames=frames,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                power=power)

        if idx == 0:
            features = np.zeros((vector_array.shape[0] * len(file_list), dims), dtype=np.float32)
        features[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array
        del vector_array
        gc.collect()

    #m_label = np.empty((features.shape[0], cls_num), dtype=np.float32)
    #nm_label = np.empty((features.shape[0], cls_num), dtype=np.float32)
    m_label = np.zeros((features.shape[0], cls_num), dtype=np.float32)
    nm_label = np.zeros((features.shape[0], cls_num), dtype=np.float32)
    
    #m_label.fill(-1)
    #nm_label.fill(-1)

    for i in range(len(m_label)):
        nm_indices = []
        for j in range(cls_num):
            if j == cls_label:
                m_label[i][j] = 1
            else:
                nm_indices.append(j)

        nm_idx = random.choice(nm_indices)
        nm_label[i][nm_idx] = 1
    from reconstruct_img import reconstruct_spectrogram
    reconstruct_spectrogram([features[0], features[1], features[2]], ["features0", "features1", "features2"])
    dataset = [[features[i], m_label[i], nm_label[i]] for i in range(len(m_label))]
    return dataset
        
'''
def file_list_generator(target_dir,
                        dir_name="train",
                        ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    dir_name : str (default="train")
        directory name containing training data
    ext : str (default="wav")
        file extension of audio files

    return :
        train_files : list [ str ]
            file list for training
    """
    com.logger.info("target_dir : {}".format(target_dir))

    # generate training list
    training_list_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    files = sorted(glob.glob(training_list_path))
    if len(files) == 0:
        com.logger.exception("no_wav_file!!")

    file_num = int(len(files) / 2) if len(files) > 2000 else len(files)
    random.shuffle(files)
    files = files[:file_num]
    com.logger.info("train_file num : {num}".format(num=len(files)))
    return files
'''

def file_list_generator(target_dir,
                             id_name,
                             dir_name="train",
                             prefix_normal="normal",
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
    ext : str (default="wav")
        file extension of audio files

    """
    com.logger.info("target_dir : {}".format(target_dir+"_"+id_name))

    # development
    #if mode:
    files = sorted(
        glob.glob("{dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                dir_name=dir_name,
                                                                                prefix_normal=prefix_normal,
                                                                                id_name=id_name,
                                                                                ext=ext)))

    # files = numpy.concatenate((normal_files, anomaly_files), axis=0)
    # labels = numpy.concatenate((normal_labels, anomaly_labels), axis=0)
    random.shuffle(files)
    # files = files[:600]
    files = files[:300]
    #files = files[:100]
    com.logger.info("train_file  num : {num}".format(num=len(files)))
    if len(files) == 0:
        com.logger.exception("no_wav_file!!")
    print("\n========================================")

    return files
########################################################################


########################################################################
# main 00_train.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)
        
    # make output directory
    os.makedirs(param["model_directory"]["idcae"], exist_ok=True)

    # initialize the visualizer
    visualizer = visualizer()

    # load base_directory list
    dirs = com.select_dirs(param=param, mode=mode)

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx+1, total=len(dirs)))

        # set path
        '''
        model_file_path change to .pt
        '''
        
        machine_type = os.path.split(target_dir)[1]
        encoder_file_path = "{model}/encoder_{machine_type}.pt".format(model=param["model_directory"]["idcae"],
                                                                     machine_type=machine_type)
        decoder_file_path = "{model}/decoder_{machine_type}.pt".format(model=param["model_directory"]["idcae"],
                                                                     machine_type=machine_type)

        history_img = "{model}/history_{machine_type}.png".format(model=param["model_directory"]["idcae"],
                                                                  machine_type=machine_type)

        if os.path.exists(encoder_file_path) and os.path.exists(decoder_file_path):
            com.logger.info("model exists")
            continue
        
        writer = SummaryWriter(comment="_"+machine_type)
        # generate dataset
        print("============== DATASET_GENERATOR ==============")

        machine_id_list = com.get_machine_id_list(target_dir, dir_name="train")

        for i in range(len(machine_id_list)):
            files = file_list_generator(target_dir, machine_id_list[i])

            sub_dataset = list_to_vector_array(files,
                                        cls_label=i,
                                        cls_num=len(machine_id_list),
                                        msg="generate train_dataset",
                                        n_mels=param["feature"]["idcae"]["n_mels"],
                                        frames=param["feature"]["idcae"]["frames"],
                                        n_fft=param["feature"]["idcae"]["n_fft"],
                                        hop_length=param["feature"]["idcae"]["hop_length"],
                                        power=param["feature"]["idcae"]["power"])


            if i == 0:
                dataset = sub_dataset
                #del sub_dataset
            else:
                dataset = np.concatenate((dataset, sub_dataset), axis=0)
                #del sub_dataset
        
        # train model
        print("============== MODEL TRAINING ==============")
        ########################################################################################
        # pytorch
        import torch.nn as nn
        import torch
        from torch.utils.data import random_split, DataLoader
        from pytorch_model import Encoder, Decoder, CustomLoss
        import json
        # from get_Threshold import get_threshold
        ########################################################################################

        paramF = param["feature"]["idcae"]["frames"]
        paramM = param["feature"]["idcae"]["n_mels"]
        dim = paramF * paramM

        encoder = Encoder(paramF=paramF, paramM=paramM, classNum=len(machine_id_list))
        decoder = Decoder(paramF=paramF, paramM=paramM, classNum=len(machine_id_list))
        encoder.float()
        decoder.float()
        
        '''
        1. Dataset input to model
        2. Define optimizer and loss
        3. Validation
        '''  
        epochs = int(param["fit"]["idcae"]["epochs"])
        batch_size = int(param["fit"]["idcae"]["batch_size"])

        #scheduler = lr_sched.StepLR(optimizer=optimizer, step_size=5, gamma=0.95)
        
        val_split = param["fit"]["idcae"]["validation_split"]
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size

        train_dataset, valid_dataset = random_split(dataset, [train_size, val_size])
        train_batches = DataLoader(dataset=MelDataLoader(train_dataset), batch_size=batch_size, shuffle=True)
        val_batches = DataLoader(dataset=MelDataLoader(valid_dataset), batch_size=batch_size, shuffle=True)

        device = torch.device('cuda')
        
        '''
        Encoder Training
        '''
        en_train_loss_list = []
        en_val_loss_list = []

        en_loss_fn = nn.CrossEntropyLoss(reduction='sum')
        # ToyCar
        # en_optim = torch.optim.SGD(encoder.parameters(), lr=3e-6, weight_decay=1e-7)

        # else:
        en_optim = torch.optim.SGD(encoder.parameters(), lr=1e-6, weight_decay=1e-7)
        encoder = encoder.to(device=device, dtype=torch.float32)
        if os.path.exists(encoder_file_path):
            print("Encoder exists...")
            encoder.load_state_dict(torch.load(encoder_file_path))
        else:
            print("Start Encoder training...")
            for epoch in range(1, epochs+1):
                train_loss = 0.0
                val_loss = 0.0
                print("Epoch: {}".format(epoch))

                encoder.train()
                for feature_batch, label_batch, _ in tqdm(train_batches):
                    en_optim.zero_grad()

                    feature_batch = feature_batch.to(device, non_blocking=True, dtype=torch.float32)
                    label_batch = label_batch.to(device, non_blocking=True, dtype=torch.float32)

                    _ , cls_output = encoder(feature_batch)
                    cls_output = cls_output.to(device=device, non_blocking=True, dtype=torch.float32)

                    label_batch = torch.argmax(label_batch, dim=1)
                    
                    loss = en_loss_fn(cls_output, label_batch.long())
                    loss.backward()
                    en_optim.step()

                    train_loss += loss.item()
                    del feature_batch, label_batch, cls_output
                    gc.collect()
                train_loss /= len(train_batches)
                en_train_loss_list.append(train_loss)

                encoder.eval()
                for feature_batch, label_batch, _ in tqdm(val_batches):

                    feature_batch = feature_batch.to(device, non_blocking=True, dtype=torch.float32)
                    label_batch = label_batch.to(device, non_blocking=True, dtype=torch.float32)
                    
                    _ , cls_output = encoder(feature_batch)
                    cls_output = cls_output.to(device=device, non_blocking=True, dtype=torch.float32)
                    label_batch = torch.argmax(label_batch, dim=1)
                    
                    loss = en_loss_fn(cls_output, label_batch.long())
                    val_loss += loss.item()

                    del feature_batch, label_batch, cls_output
                    gc.collect()  
                
                val_loss /= len(val_batches)
                en_val_loss_list.append(val_loss)

                writer.add_scalar('en_train/loss', train_loss, epoch)
                writer.add_scalar('en_val/loss', val_loss, epoch)
                writer.add_scalars('en_comp/loss', {'train': train_loss, 'validation': val_loss}, epoch)

            torch.save(encoder.state_dict(), encoder_file_path)
        '''
        Decoder Training
        '''
        de_train_loss_list = []
        de_val_loss_list = []

        decoder = decoder.to(device=device, dtype=torch.float32)  
        de_loss_fn = nn.MSELoss()

        de_optim = torch.optim.SGD(decoder.parameters(), lr=5e-4, weight_decay=1e-7)
        scheduler = lr_sched.StepLR(optimizer=de_optim, step_size=5, gamma=0.90)
        alpha = 0.75
        C = 5

        nm_input = np.empty(shape=(batch_size, dim))
        nm_input.fill(C)
        nm_input = torch.Tensor(nm_input).to(device=device, non_blocking=True, dtype=torch.float32)
        

        print("Start Decoder training...")
        encoder.eval()
        m_lost_list = np.array([])
        nm_lost_list = np.array([])
        m_output_toimg, nm_output_toimg = None, None
        for epoch in range(1, epochs+1):
            train_loss = 0.0
            val_loss = 0.0
            ml = 0.0
            nml = 0.0
            bl = 0.0
            print("Epoch: {}".format(epoch))
            nm_loss, m_loss = 0, 0
            decoder.train()
            for feature_batch, label_batch, nm_label_batch in tqdm(train_batches):
                de_optim.zero_grad()
                
                feature_batch = feature_batch.to(device, non_blocking=True, dtype=torch.float32)
                label_batch = label_batch.to(device, non_blocking=True, dtype=torch.float32)
                nm_label_batch = nm_label_batch.to(device, non_blocking=True, dtype=torch.float32)
                
                latent, _ = encoder(feature_batch)

                label_batch = 2 * (label_batch - 0.5)
                nm_label_batch = 2 * (nm_label_batch - 0.5)
                
                m_output, nm_output = decoder(latent, label_batch, nm_label_batch)
                m_output = m_output.to(device, non_blocking=True, dtype=torch.float32)
                nm_output = nm_output.to(device, non_blocking=True, dtype=torch.float32)

                m_loss = de_loss_fn(m_output, feature_batch)
                nm_to_m = de_loss_fn(nm_output, feature_batch)
                if nm_output.shape[0] < batch_size:
                    nm_loss = de_loss_fn(nm_output, nm_input[:nm_output.shape[0]])
                else:
                    nm_loss = de_loss_fn(nm_output, nm_input)

                # m_loss, nm_loss = de_loss_fn(m_output, nm_output, feature_batch)
                
                loss = alpha * m_loss + (1-alpha) * nm_loss

                loss.backward()
                de_optim.step()
                train_loss += loss.item()
                m_output_toimg = m_output.cpu().detach().numpy()
                nm_output_toimg = nm_output.cpu().detach().numpy()
                if epoch == epochs:
                    m_lost_list = np.concatenate((m_lost_list, m_loss.cpu().detach().numpy()), axis=None)
                    nm_lost_list = np.concatenate((nm_lost_list, nm_to_m.cpu().detach().numpy()), axis=None)   
                del m_output, nm_output, feature_batch, label_batch, nm_label_batch
                gc.collect() 
            train_loss /= len(train_batches)
            # print("loss m & nm:", m_loss.cpu().detach().numpy(), nm_loss.cpu().detach().numpy(), train_loss)

            de_train_loss_list.append(train_loss)

            decoder.eval()

            for feature_batch, label_batch, nm_label_batch in tqdm(val_batches):
                
                feature_batch = feature_batch.to(device, non_blocking=True, dtype=torch.float32)
                label_batch = label_batch.to(device, non_blocking=True, dtype=torch.float32)
                nm_label_batch = nm_label_batch.to(device, non_blocking=True, dtype=torch.float32)
                
                latent, _ = encoder(feature_batch)

                label_batch = 2 * (label_batch - 0.5)
                nm_label_batch = 2 * (nm_label_batch - 0.5)

                m_output, nm_output = decoder(latent, label_batch, nm_label_batch)
                m_output = m_output.to(device, non_blocking=True, dtype=torch.float32)
                nm_output = nm_output.to(device, non_blocking=True, dtype=torch.float32)

                m_loss = de_loss_fn(m_output, feature_batch)
                if nm_output.shape[0] < batch_size:
                    nm_loss = de_loss_fn(nm_output, nm_input[:nm_output.shape[0]])
                else:
                    nm_loss = de_loss_fn(nm_output, nm_input)

                bad_loss = de_loss_fn(nm_output, feature_batch)
                bl += bad_loss.item()

                loss = alpha * m_loss + (1-alpha) * nm_loss
                val_loss += loss.item()
                ml += m_loss.item()
                nml += nm_loss.item()
                del m_output, nm_output, feature_batch, label_batch, nm_label_batch
                gc.collect()
            val_loss /= len(val_batches)
            # print("val loss m & nm:", m_loss.cpu().detach().numpy(), nm_loss.cpu().detach().numpy(), val_loss)

            ml /= len(val_batches)
            nml /= len(val_batches)
            bl /= len(val_batches)

            de_val_loss_list.append(val_loss)
            
            writer.add_scalar('de_train/loss', train_loss, epoch)
            writer.add_scalar('de_val/loss', val_loss, epoch)
            writer.add_scalars('de_comp/loss', {'train': train_loss, 'validation': val_loss}, epoch)
            writer.add_scalar('match loss', ml, epoch)
            writer.add_scalar('non match loss', nml, epoch)
            writer.add_scalar('bad loss', bl, epoch)

            scheduler.step()
        visualizer.loss_plot(de_train_loss_list, de_val_loss_list)
        visualizer.save_figure(history_img)

        torch.save(decoder.state_dict(), decoder_file_path)

        com.logger.info("save_model -> en: {en_path} de: {de_path}".format(en_path=encoder_file_path, de_path=decoder_file_path))

        from reconstruct_img import reconstruct_spectrogram
        # from evt import evt
        reconstruct_spectrogram([m_output_toimg[0], m_output_toimg[1], m_output_toimg[2], nm_output_toimg[0], nm_output_toimg[1], nm_output_toimg[2]], ["m0", "m1", "m2", "nm0", "nm1", "nm2"])
        # latent_distrbution(latents_list=[m_lost_list, nm_lost_list], labels=["match", "non_match"], name="m_nm_output")
        n_file_name = "./loss_record/" + machine_type + "_normal_loss.json"
        with open(n_file_name, 'w', encoding='utf-8') as f:
            json.dump(list(m_lost_list.astype(float)), f, ensure_ascii=False)
        an_file_name = "./loss_record/" + machine_type + "_anormal_loss.json"
        with open(an_file_name, 'w', encoding='utf-8') as f:
            json.dump(list(nm_lost_list.astype(float)), f, ensure_ascii=False)
        # get threshold
        # threshold = get_threshold(n_file_name, an_file_name, machine_type)
        # print(threshold)
        # file_name = "./loss_record/" + machine_type + "_threshold.json"
        # with open(file_name, 'w', encoding='utf-8') as f:
        #     json.dump(threshold, f, ensure_ascii=False)

        del dataset, train_batches, val_batches
        gc.collect()
        time.sleep(30)