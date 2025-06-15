import torch 
from torch.utils.data import DataLoader
from sklearn import metrics
from net_large_statex import TASTgramMFN
from losses import ASDLoss
import pandas as pd
import yaml
from tqdm import tqdm
import numpy as np
import os
import glob
import torchaudio
import sys
import pandas as pd
from net_na import TASTgramMFN
from net_st import TASTgramMFN as TASTgramMFN_ST
from net_lg_st import TASTgramMFN as TASTgramMFN_LG_ST
def file_to_log_mel_spectrogram(y, sr, n_mels, n_fft, hop_length, power):
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
                                                     n_mels=n_mels, power=power, pad_mode='constant', norm='slaney', mel_scale='slaney')
    mel_spectrogram = transform(y)
    log_mel_spectrogram = 20.0 / power * torch.log10(mel_spectrogram + sys.float_info.epsilon)
    return log_mel_spectrogram






class eval_dataset(torch.utils.data.Dataset):
    def __init__(self,  root_path, test_name, name_list):
        test_filename = test_name + "_eval/" + test_name + "_eval" 
        dataset_dir = os.path.join(root_path, test_filename, 'test')
        for dirpath, dirnames, filenames in os.walk(dataset_dir):
            wav_files = glob.glob(os.path.join(dirpath, '*.wav'))
        self.test_files = wav_files
        
        '''self.test_files = np.concatenate((normal_files, anomaly_files), axis=0)
        
        normal_labels = np.zeros(len(normal_files))
        anomaly_labels = np.ones(len(anomaly_files))
        self.y_true = torch.LongTensor(np.concatenate((normal_labels, anomaly_labels), axis=0))'''
        
        target_idx = name_list.index(test_name)
        
        label_init_num = 0
        for i, name in enumerate(name_list):
            if i == target_idx:
                break
            label_init_num+=len(self._get_label_list(name))
            
        self.labels = []
        label_list = self._get_label_list(test_name)
        for file_name in self.test_files:
            for idx, label_idx in enumerate(label_list):
                if label_idx in file_name:
                    self.labels.append(idx + label_init_num)
        
        self.labels = torch.LongTensor(self.labels)
        self.filename_list = []
        self.y_list = []
        self.y_spec_list = []
        
        for i in tqdm(range(len(self.test_files))):
            y, sr = self._file_load(self.test_files[i])
            y_specgram = file_to_log_mel_spectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128, power=2)
            self.filename_list.append(os.path.basename(self.test_files[i]))
            self.y_list.append(y)
            self.y_spec_list.append(y_specgram)
    
    def __getitem__(self, idx):
        try:
            label = self.labels[idx]
        except:
            print("label not exists!! : {}".format(idx))
            label = 0
        return self.y_list[idx], self.y_spec_list[idx], label, self.filename_list[idx]

    def __len__(self):
        return len(self.test_files)
    
    def _file_load(self, file_name):
        try:
            y, sr = torchaudio.load(file_name)
            y = y[..., :sr * 10]
            return y, sr
        except:
            print("file_broken or not exists!! : {}".format(file_name))
    
    def _get_label_list(self, name):
        if name == 'ToyConveyor':
            label_list = ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06'] 
    
        elif name == 'ToyCar':
            label_list = ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07']
        
        else:
            label_list = ['id_00', 'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06']
            
        return label_list 



def evaluator_test(net_na, net_st, net_lg_st, save_name, test_loader, criterion, device):

    if save_name in ['fan', 'pump', 'slider', 'valve']:
        net = net_lg_st
    elif save_name in ['ToyConveyor']:
        net = net_st    
    elif save_name in ['ToyCar']:
        net = net_na
    y_pred = []
    filenames = []
    with torch.no_grad():
        for x_wavs, x_mels, labels, filename in test_loader:
            x_wavs, x_mels, labels, filename = x_wavs.to(device), x_mels.to(device), labels.to(device), filename
            
            logits, _ = net(x_wavs, x_mels, labels, train=False)
            
            score = criterion(logits, labels)

            y_pred.extend(score.tolist())
            #y_true.extend(AN_N_labels.tolist())
            filenames.extend(filename)
        df = pd.DataFrame({str(save_name) : filenames, 'anomaly_score': y_pred})
        df['int_key'] = df[str(save_name)].str.extract(r'(\d{8})').astype(int)
        df = df.sort_values('int_key').drop(columns='int_key')
        if save_name is not None:
            df.to_csv(os.path.join('/home/yongjoonlee/ENTER/envs/noisyarcmix/', f'anomaly_score_{save_name}.csv'), index=False)
        return df

   


def main():
    with open('config.yaml', encoding='UTF-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    print(cfg)
    
    device_num = cfg['gpu_num']
    
    device = torch.device(f'cuda:{device_num}')

    net_na = TASTgramMFN(num_classes=cfg['num_classes'], m=cfg['m'], mode=cfg['mode']).to(device)
    
    net_na.load_state_dict(torch.load('/home/yongjoonlee/ENTER/envs/noisyarcmix/best/pth/na.pth'))
    net_na.eval()

    net_st = TASTgramMFN_ST(num_classes=cfg['num_classes'], m=cfg['m'], mode=cfg['mode']).to(device)
    net_st.load_state_dict(torch.load('/home/yongjoonlee/ENTER/envs/noisyarcmix/best/pth/st.pth'))
    net_st.eval()

    net_lg_st = TASTgramMFN_LG_ST(num_classes=cfg['num_classes'], m=cfg['m'], mode=cfg['mode']).to(device)
    net_lg_st.load_state_dict(torch.load('/home/yongjoonlee/ENTER/envs/noisyarcmix/best/pth/lg_st.pth'))
    net_lg_st.eval()

    
    criterion = ASDLoss(reduction=False).to(device)
    '''name_list = ['fan_eval/fan_eval', 'pump_eval/pump_eval', 'slider_eval/slider_eval', 
                 'ToyCar_eval/ToyCar_eval', 'ToyConveyor_eval/ToyConveyor_eval', 'valve_eval/valve_eval']'''
    #name_list_specific = 'fan_eval/fan_eval'
    root_path = '/home/yongjoonlee/ENTER/envs/noisyarcmix/'
    

    root_path = '/home/yongjoonlee/ENTER/envs/noisyarcmix/'
    avg_AUC = 0.
    avg_pAUC = 0.
    name_list = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
    for i in range(len(name_list)):
        test_ds = eval_dataset(root_path, name_list[i], name_list)
        test_dataloader = DataLoader(test_ds, batch_size=1)
        
        df = evaluator_test(net_na, net_st, net_lg_st, name_list[i], test_dataloader, criterion, device)

    
    '''avg_AUC = avg_AUC / len(name_list)
    avg_pAUC = avg_pAUC / len(name_list)
    
    print(f"Average AUC: {avg_AUC:.5f},  Average pAUC: {avg_pAUC:.5f}")'''
        
    
if __name__ == '__main__':
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    main()
