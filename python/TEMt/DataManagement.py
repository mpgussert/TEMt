import numpy as np
import os

import torch
from torch.utils.data import Dataset

class JetbotSensoryData(Dataset):
    def __init__(self, root_paths):
        
        ### specific to jetbot ###
        # chans = 3 (rgb)
        # rows = 224 (resnet50 sensorium)
        # cols = 224
        # imu_dims = 6
        ##########################
        self.root_paths = root_paths

        ##########################
        # obs
        image_size = 3*64*64 # resnet18 sensorium
        imu_size = 6
        imu = list(range(imu_size))
        image = list(range(imu_size, imu_size + image_size))

        self._images = []
        self._imus = []
        for path in root_paths:
            obs_path = os.path.join(path, 'obs')
            obs_files = os.listdir(obs_path)

            for name in obs_files:
                data = torch.from_numpy(np.load(os.path.join(obs_path, name)))
                print(data.shape)
                imus = data[:,imu]

                self._images.append(data[:,image].reshape(-1, 64, 64, 3)/255.0)
                self._imus.append(imus)
        
        self._images = torch.vstack(self._images)
        self._imus = torch.vstack(self._imus)
        
        # standardize IMU data
        self._imu_mean = torch.mean(self._imus, dim=0).unsqueeze(0)
        self._imu_stdv = torch.std(self._imus, dim=0).unsqueeze(0)
        self._imus = (self._imus - self._imu_mean) / self._imu_stdv
        self._imus = torch.sigmoid(self._imus)

        print(self._images.shape)
        self._total_length = self._images.shape[0]

        ##########################
        # act
        self._acts = []

        for path in root_paths:
            act_path = os.path.join(path, 'act')
            act_files = os.listdir(act_path)
            for name in act_files:
                data = torch.from_numpy(np.load(os.path.join(act_path, name)))
                self._acts.append(data)
        
        self._acts = torch.vstack(self._acts)

        ##########################
        # truth
        self._truths = []
        self._num_runs = 0
        for i, path in enumerate(root_paths):
            truth_path = os.path.join(path, 'truth')
            truth_files = os.listdir(truth_path)
            for j, name in enumerate(truth_files):
                data = torch.from_numpy(np.load(os.path.join(truth_path, name)))
                index = self._num_runs*torch.ones((data.shape[0],1)).int()
                data = torch.hstack((index, data))
                self._truths.append(data)
                self._num_runs += 1
        
        self._truths = torch.vstack(self._truths)

        assert self._total_length == self._acts.shape[0] == self._truths.shape[0]
    
    def __getitem__(self, index):
        image = torch.permute(self._images[index],(2,0,1)).float()
        return image, self._imus[index], self._acts[index], self._truths[index]
    
    def __len__(self):
        return self._total_length


class SensoryData(Dataset):
    def __init__(self, root_paths):
        
        ### specific to carter_v2_3 ###
        # chans = 6 (two stacked images)
        # rows = 100
        # cols = 100
        # imu_dims = 6
        ##########################
        
        self.root_paths = root_paths

        ##########################
        # obs
        image_size = 3*100*100
        imu_size = 6
        imu = list(range(imu_size))
        imageL = list(range(imu_size, imu_size + image_size))
        imageR = list(range(imu_size + image_size, imu_size+2*image_size))
        
        self._imagesL = []
        self._imagesR = []
        self._imus = []
        
        for path in root_paths:
            obs_path = os.path.join(path, 'obs')
            obs_files = os.listdir(obs_path)

            for name in obs_files:
                data = torch.from_numpy(np.load(os.path.join(obs_path, name)))
                print(data.shape)
                imus = data[:,imu]

                self._imagesL.append(data[:,imageL].reshape(-1, 100, 100, 3)/255.0)
                self._imagesR.append(data[:,imageR].reshape(-1, 100, 100, 3)/255.0)
                self._imus.append(imus)
        
        self._imagesL = torch.vstack(self._imagesL)
        self._imagesR = torch.vstack(self._imagesR)
        self._imus = torch.vstack(self._imus)
        
        # standardize IMU data
        self._imu_mean = torch.mean(self._imus, dim=0).unsqueeze(0)
        self._imu_stdv = torch.std(self._imus, dim=0).unsqueeze(0)
        self._imus = (self._imus - self._imu_mean) / self._imu_stdv
        self._imus = torch.sigmoid(self._imus)

        print(self._imagesL.shape)
        self._total_length = self._imagesL.shape[0]

        ##########################
        # act
        self._acts = []

        for path in root_paths:
            act_path = os.path.join(path, 'act')
            act_files = os.listdir(act_path)
            for name in act_files:
                data = torch.from_numpy(np.load(os.path.join(act_path, name)))
                self._acts.append(data)
        
        self._acts = torch.vstack(self._acts)

        ##########################
        # truth
        self._truths = []
        self._num_runs = 0
        for i, path in enumerate(root_paths):
            truth_path = os.path.join(path, 'truth')
            truth_files = os.listdir(truth_path)
            for j, name in enumerate(truth_files):
                data = torch.from_numpy(np.load(os.path.join(truth_path, name)))
                index = self._num_runs*torch.ones((data.shape[0],1)).int()
                data = torch.hstack((index, data))
                self._truths.append(data)
                self._num_runs += 1
        
        self._truths = torch.vstack(self._truths)

        assert self._total_length == self._acts.shape[0] == self._truths.shape[0]

    
    def __getitem__(self, index):
        L = torch.permute(self._imagesL[index],(2,0,1)).float()
        R = torch.permute(self._imagesR[index], (2,0,1)).float()
        return torch.vstack([L,R]), self._imus[index], self._acts[index], self._truths[index]
    
    def __len__(self):
        return self._total_length
    
class Sequencer(Dataset):
    def __init__(self, data_path, min_len:int = 5, max_len:int = 100):
        self.data_path = data_path
        self._min_len = min_len
        self._max_len = max_len
        names = os.listdir(data_path)
        self._X = []
        self._A = []
        self._T = []
        self._L = []
        
        for name in names:
            data = np.load(os.path.join(data_path, name))
            if data['X'].shape[0] <= self._min_len:
                continue
            self._X.append(torch.from_numpy(data['X']))
            self._A.append(torch.from_numpy(data['A']))
            self._T.append(torch.from_numpy(data['T']))
            self._L.append(data['X'].shape[0])
        
        self._num_files = len(self._X) 
        
        self._sequences = {}
        self._num_seqs = [] 
        total_seqs = 0
        for i in range(self._num_files):
            print("processing file {0}".format(i))
            start = 0
            stop = self._min_len
            seqs = []
            new_file = True
            while True:
                if not new_file and start + self._min_len == stop:
                    break
                
                new_file = False
                diff = stop - start
                test = self._min_len <= diff < self._max_len
                # if not test:
                #    print(self._min_len, diff, self._max_len)
                seqs.append(torch.arange(start,stop))
                total_seqs += 1
                if self._min_len <= diff and diff < self._max_len:
                    if stop+1 <= self._L[i]:
                        stop+=1
                        continue
                    else:
                        start += 1
                        continue
                else:
                    start += 1
                    if stop+1 < self._L[i]:
                        stop+=1
            self._sequences[i] = seqs
            self._num_seqs.append(len(seqs))
        self._total_length = np.sum(self._num_seqs)
        print(self._total_length, " total sequences")

    def indexToSequenceID(self, index):
        for i in range(len(self._num_seqs)):
            num = self._num_seqs[i]
            if index < num:
                return i, index
            index -= num
            
    def __getitem__(self, index):
        run, idx = self.indexToSequenceID(index)
        sequence = self._sequences[run][idx]
        X = self._X[run][sequence]
        A = self._A[run][sequence]
        T = self._T[run][sequence]
        return X.float(), A.float(), T.float()
    
    def __len__(self):
        return self._total_length 
    
    def num_runs(self):
        return len(self._X)

    def get_run(self, run):
        X = self._X[run]
        A = self._A[run]
        T = self._T[run]
        return X.float(), A.float(), T.float()

class ImageSequencer(Dataset):
    def __init__(self, root_paths, min_len:int = 5, max_len:int = 100, image_rows:int = 224, image_cols:int = 224, image_chans:int = 3, imu_size:int = 6):
        self._root_paths = root_paths
        self._min_len = min_len
        self._max_len = max_len
        self._image_rows = image_rows
        self._image_cols = image_cols
        self._image_chans = image_chans
        self._imu_dim = imu_size

        # setting up indexers for the 
        image_size = image_cols*image_rows*image_chans
        imu_idx = list(range(imu_size))
        image_idx = list(range(imu_size, imu_size + image_size))

        self._images = []
        self._imus = []
        self._acts = []
        self._truths = []

        self._num_runs = 0
        
        for path in root_paths:
            obs_path = os.path.join(path, 'obs')
            obs_files = os.listdir(obs_path)

            act_path = os.path.join(path, 'act')
            act_files = os.listdir(act_path)

            truth_path = os.path.join(path, 'truth')
            truth_files = os.listdir(truth_path)
            
            for (obs_name, act_name, truth_name) in zip(obs_files, act_files, truth_files):
                obs_data   = torch.from_numpy(np.load(os.path.join(obs_path, obs_name)))
                if obs_data.shape[0] < self._min_len:
                    continue
                act_data   = torch.from_numpy(np.load(os.path.join(act_path, act_name)))
                truth_data = torch.from_numpy(np.load(os.path.join(truth_path, truth_name)))
                # print(obs_data.shape, act_data.shape, truth_data.shape)    
                imus = obs_data[:,imu_idx]
                image = obs_data[:,image_idx].reshape(-1, self._image_rows, self._image_cols, self._image_chans)/255.0
                image = torch.clamp(image, 0, 1)
                assert torch.min(image) >= 0
                self._images.append(image.permute(0,3,1,2))
                self._imus.append(imus)
                
                self._acts.append(act_data)
                
                index = self._num_runs*torch.ones((truth_data.shape[0],1)).int()
                data = torch.hstack((index, truth_data))
                self._truths.append(truth_data)
                
                self._num_runs += 1
                
        #self._images = torch.vstack(self._images)
        #self._imus = torch.vstack(self._imus)
        #self._acts = torch.vstack(self._acts)
        #self._truths = torch.vstack(self._truths)

        self._sequences = {}
        self._num_seqs = []
        self._total_sequences = 0
        total_seqs =0
        
        for i in range(self._num_runs):
            print("processing file {0} of length {1}".format(i, self._images[i].shape[0]))
            start = 0
            stop = self._min_len
            seqs = []
            new_run = True
            run_length = self._images[i].shape[0]
            
            while True:
                if not new_run and start + self._min_len >= stop:
                    break
                    
                new_run = False
                diff = stop - start
                
                test = self._min_len <= diff < self._max_len
                #if not test:
                #    print(self._min_len, diff, self._max_len, start, stop)

                seqs.append(torch.arange(start, stop))
                total_seqs += 1
                
                if self._min_len <= diff and diff < self._max_len:
                    if stop+1 <= run_length:
                        stop +=1
                        continue
                    else:
                        start +=1
                        continue
                else:
                    start += 1
                    if stop+1 < run_length:
                        stop+=1
            self._sequences[i] = seqs
            self._num_seqs.append(len(seqs))
        self._total_sequences = np.sum(self._num_seqs)
        print(self._total_sequences, total_seqs)
    
    def indexToSequenceID(self, index):
        for i in range(len(self._num_seqs)):
            num = self._num_seqs[i]
            if index < num:
                return i, index
            index -= num
            
    def __getitem__(self, index):
        run, idx = self.indexToSequenceID(index)
        sequence = self._sequences[run][idx]
        # print(run, idx, index, sequence.shape, len(self._images), len(self._acts), len(self._truths))
        X = self._images[run][sequence]
        A = self._acts[run][sequence]
        T = self._truths[run][sequence]
        return X.float(), A.float(), T.float()
    
    def __len__(self):
        return self._total_sequences
    
    def num_runs(self):
        return len(self._images)

    def get_run(self, run):
        X = self._images[run]
        A = self._acts[run]
        T = self._truths[run]
        return X.float(), A.float(), T.float()

def collator(batch, min_len = 5, length=100):
    """
    collate a list of (X, A) with different sequence lengths into a padded 
    tensor with constant dimensions
    """
    xdim = batch[0][0].shape[1:]
    adim = batch[0][1].shape[1:]
    tdim = batch[0][2].shape[1:]
    
    Xlist = []
    Alist = []
    Tlist = []
    
    for (x, a, t) in batch:
        coinflip = np.random.randn() >= 0.5
        seq_len = x.shape[0]
        front_pad = 0
        back_pad = 0
        diff = length - seq_len
        
        if diff == 0:
            Xlist.append(x)
            Alist.append(a)
            Tlist.append(t)
            continue

        elif diff < min_len:
            if coinflip:
                front_pad = diff
            else:
                back_pad = diff
        
        else:
            front_pad = int(np.random.rand()*diff)
            back_pad = diff - front_pad
            
        # print(front_pad, seq_len, back_pad)
        FPadX = torch.randn(front_pad, *xdim)/100.0
        FPadA = torch.randn(front_pad, *adim)/100.0
        FPadT = torch.randn(front_pad, *tdim)/100.0
        
        BPadX = torch.randn(back_pad, *xdim)/100.0
        BPadA = torch.randn(back_pad, *adim)/100.0
        BPadT = torch.randn(back_pad, *tdim)/100.0
        
        X = torch.vstack((FPadX, x, BPadX))
        A = torch.vstack((FPadA, a, BPadA))
        T = torch.vstack((FPadT, t, BPadT))
        
        Xlist.append(X)
        Alist.append(A)
        Tlist.append(T)
        
    Xbatch = torch.clamp(torch.stack(Xlist), 0, 1)
    Abatch = torch.stack(Alist)
    Tbatch = torch.stack(Tlist)
    
    return Xbatch.float(), Abatch.float(), Tbatch.float()