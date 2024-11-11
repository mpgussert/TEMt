import numpy as np
import os

import torch
from torch.utils.data import Dataset

def GetSubsequences(data_len, min_len, max_len):

    def _getSubs(max_len, min_len):
        max_seq = np.arange(max_len)
        if max_len > min_len:
            subs = []
            for start in range(min_len, max_len+1, 1):
                subs.append(max_seq[-start:])
            return subs
        else:
            return [max_seq]

    seqs = []

    i = min_len
    while i < max_len:
        if i > min_len:
            subs = _getSubs(i, min_len)
            seqs.extend([torch.Tensor(subs[j].tolist()) for j in range(len(subs))])
            i+=1
            continue
        seqs.append(torch.Tensor(list(range(i))))
        i+=1

    subs =_getSubs(max_len, min_len)
    
    for i in range(data_len-max_len+1):
        seqs.extend([torch.Tensor((subs[j]+i).tolist()) for j in range(len(subs))])

    return seqs

class ImageSequencer(Dataset):
    def __init__(self, root_paths, min_len:int = 5, max_len:int = 100, image_rows:int = 224, image_cols:int = 224, image_chans:int = 3, num_images=2, add_dims=6):
        
        self._root_paths = root_paths
        self._min_len = min_len
        self._max_len = max_len
        self._image_rows = image_rows
        self._image_cols = image_cols
        self._image_chans = image_chans
        self._num_images = num_images
        self._add_dims = add_dims

        """
        Assume root directory is structured like the following 
        
           root_path
               |
               -- obs
               |   |
               |   -- obs_data_0.npy
               |   -- obs_data_1.npy
               |   -- obs_data_2.npy
               -- act
               |   |
               |   -- act_data_0.npy
               |   -- act_data_1.npy
               |   -- act_data_2.npy
               -- truth
               |   |
               |   -- truth_data_0.npy
               |   -- truth_data_1.npy
               |   -- truth_data_2.npy
        
        Data is collected by allowing a robot to act according to some policy.
        Each index (0, 1, 2, etc...) corresponds to a rollout or "run".  The data from the run
        is factorized into 3 parts.
        
        obs is the data that could be rendered by a real robot
        act is the commands to the robot
        truth is ground truth data from the simulation

        Assume that a single obs token consists of a sequence of flattened images
        prepended by "additional_dims" of float data, and that the number of entries in the file 
        is the "length" of the run.

        PER FILE [run_length, add_dims + image_size * num_images]
        idx is the index of the specific parts of the entry corresponding to specific data.  It's 
        where that data is located in the entry
        """

        add_idx = list(range(self._add_dims))
        self._image_size = self._image_rows*self._image_cols*self._image_chans
        self._image_idx = [list(range(self._add_dims + i*self._image_size, self._add_dims + i*self._image_size + self._image_size)) for i in range(self._num_images)]
        
        
        self._images = []
        self._add_data = []
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
                # load data from the files, skipping runs that are too short to consider
                obs_data = torch.from_numpy(np.load(os.path.join(obs_path, obs_name)))
                if obs_data.shape[0] < self._min_len:
                    continue
                act_data   = torch.from_numpy(np.load(os.path.join(act_path, act_name)))
                truth_data = torch.from_numpy(np.load(os.path.join(truth_path, truth_name)))   
                add_data = obs_data[:,add_idx]

                #unflatten.  I know how the data was packed, so I can unpack it... I wish there was a way to define this in the file so it happens automagically...
                images = []
                for idx in self._image_idx:
                    image = obs_data[:,idx].reshape(-1, self._image_rows, self._image_cols, self._image_chans)/255.0
                    images.append(image)
                images = torch.stack(images, dim=1)
                images = torch.clamp(images, 0, 1)
                assert torch.min(images) >= 0

                self._images.append(images)
                self._add_data.append(add_data)
                self._acts.append(act_data)
                
                index = self._num_runs*torch.ones((truth_data.shape[0],1)).int()
                truth_data = torch.hstack((index, truth_data))
                self._truths.append(truth_data)
                
                self._num_runs += 1

        self._sequences = {}
        self._num_seqs = []
        total_seqs =0
        
        """
        The data provided by the sequencer are sequences.  These sequences exist as lists of integers (idx) in a dictionary, 
        to be indexed at random in order to create a single batch of data.

        A run may be much longer or shorter than the max length provided to the sequencer, but it cannot be
        shorter than the minimum length.  Consider the following run of data
        
        A B C D E F G H I J 
        0 1 2 3 4 5 6 7 8 9 
        
        We want to make a list of all valid sequences in this run with a min sequence length of 3 and a 
        max length of 5.  The first sequence is always a sequence of the minimum length, in this case, ABC

        A B C D E F G H I J 
        0 1 2 3 4 5 6 7 8 9 
        |---|

        We can then extend and slide this window to the end of the run, generating sequences with each step and 
        shrinking it at the end

            A B C D E F G H I J 
            0 1 2 3 4 5 6 7 8 9 
         1  |---|                                                                                                  
         2  |-----|  
         3    |---|                                                                         
         4  |-------|            <- max length                                                                        
         5    |-----|                                                                              
         6      |---|     
         7    |-------|          <- max length                                                                       
         8      |-----|                                                                              
         9        |---|    
        10      |-------|        <- max length                                                                         
        11        |-----|                                                                              
        12          |---|  
        13        |-------|      <- max length                                                                           
        14          |-----|                                                                              
        15            |---|  
        16          |-------|    <- max length                                                                             
        17            |-----|                                                                              
        18              |---|                                                 
        19            |-------|  <- max length                                                                               
        20              |-----|                                                                              
        21                |---|                                                        

        These are all possible unique contiguous subsequences of the example run.
        """

        for i in range(self._num_runs):
            print("sequencing file {0} of length {1}".format(i, self._images[i].shape[0]))
            run_length = self._images[i].shape[0]
            
            seqs = GetSubsequences(run_length, self._min_len, self._max_len)

            self._sequences[i] = seqs
            self._num_seqs.append(len(seqs))
        self._total_sequences = np.sum(self._num_seqs)
        print(self._total_sequences)
    
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