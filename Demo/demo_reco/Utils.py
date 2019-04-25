import scipy.signal as signal
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2


class Utils():
    def __init__(self):
        pass

    def load(self,filepath):
        data = []
        with h5py.File(filepath,'r') as f:
            for keys in list(f.keys()):
                data.append(list(f[keys]))
        return np.asarray(data)

    def load_spectrum(self,filename:str):
        data = None
        with h5py.File(filename,'r') as f:
            data = []
            for name in f.keys():
                pts = []
                videos = list(f[name])
                for video in videos:
                    #print(np.asarray(list(f['/'+name+'/'+video])).shape[0])
                    if np.asarray(list(f['/'+name+'/'+video])).shape[0] >= 150:
                        pts.append(list(f['/'+name+'/'+video])[:150])
                #print(np.asarray(pts).shape)
                data.append(pts[:9])
        return np.asarray(data)

    def timefreq(self,sig,fs=30, save_path="spectre.png"):
        #print(sig.shape)
        f, t, Sxx = signal.spectrogram(sig,fs,nperseg=15,scaling='spectrum')
        #print(t)
        #print(len(f))
        #print(len(Sxx))
        plt.pcolormesh(t, f, Sxx)
        #plt.ylabel('Frequency [Hz]')
        #plt.xlabel('Time [sec]')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(save_path)
        #plt.show() 

    

    def pts_to_hdf5(self,pts,folderpath="data"):
        if not os.path.isdir(folderpath):
            os.mkdir(folderpath)
        for person_pt in enumerate(pts):
            #print(person_pt[1].shape)
            with h5py.File(os.path.join(folderpath,str(person_pt[0])+".h5"),"w") as data:
                for video_pt in enumerate(person_pt[1]):
                    #print(video_pt[1].shape)
                    data.create_dataset(str(video_pt[0]), data=video_pt[1])
            
    def create_timefreq_dataset(self,filepath):
        for root, _, files in os.walk(filepath):
            for name in files:
                if name.split(".")[1] == "h5":
                    dir_name = name.split(".")[0]
                    os.mkdir(os.path.join(root,dir_name))
                    data = self.load(os.path.join(root,name))
                    #print(data.shape)
                    sig_y = data[:,:,1::3]
                    bar = tqdm(total=sig_y.shape[0]-1)
                    bar.set_postfix(ordered_dict={"Video :": 0})
                    for video_index in range(sig_y.shape[0]):
                        for point_index in range(sig_y.shape[2]):
                            #print(os.path.join(os.path.join(root,dir_name),str(i)+".png"))
                            self.timefreq(sig_y[video_index,:,point_index],save_path=os.path.join(os.path.join(root,dir_name),"VIDEO_"+str(video_index)+"_POINT_"+str(point_index)+".png"))
                        bar.set_postfix(ordered_dict={"Personne": dir_name,"Video": video_index})
                        bar.update(1)
                    bar.close()

    def resize_data(self,filepath,width=32,height=32):
        for root, _, files in os.walk(filepath):
            for name in files:
                if name.split(".")[1] == "png":
                    img = cv2.imread(os.path.join(root,name))
                    newimg = cv2.resize(img,(width,height))
                    cv2.imwrite(os.path.join(root,name),newimg)
    
    def imgs_to_hdf5(self,folderpath):
        is_folder_empty = True
        with h5py.File("true_spectrogram_test_new.h5","w") as data:
            for root, _, files in os.walk(folderpath):
                img_pts = []
                video_imgs = []
                for index,name in enumerate(files):
                    if name.split(".")[1] == "png":
                        if index % 25 == 0 and index != 0:
                            video_imgs.append(np.asarray(img_pts))
                            #print(np.asarray(video_imgs)[:60].shape)
                            img_pts = []
                            img = cv2.imread(os.path.join(root,name))
                            img_pts.append(np.asarray(img))
                            #print("*************************")
                        else:
                            img = cv2.imread(os.path.join(root,name))
                            img_pts.append(np.asarray(img))
                            #print(np.asarray(img_pts).shape)
                        is_folder_empty = False
                if not is_folder_empty:
                    #print(np.asarray(video_imgs)[:62].shape)
                    #print(root.split(os.sep)[-1])
                    data.create_dataset(root.split(os.sep)[-1], data=np.asarray(video_imgs))
                    is_folder_empty = True

    def imgs_to_tab(self,folderpath):
        for root, _, files in os.walk(folderpath):
            img_pts = []
            video_imgs = []
            for index,name in enumerate(files):
                if name.split(".")[1] == "png":
                    if index % 25 == 0 and index != 0:
                        video_imgs.append(np.asarray(img_pts))
                        #print(np.asarray(video_imgs)[:60].shape)
                        img_pts = []
                        img = cv2.imread(os.path.join(root,name))
                        img_pts.append(np.asarray(img))
                        #print("*************************")
                    else:
                        img = cv2.imread(os.path.join(root,name))
                        img_pts.append(np.asarray(img))            
        return np.asarray(img_pts)


    def create_data_test(self):
        pts = self.load_spectrum("dataset_test_spectrum.h5")
        #print(pts)

        mean = np.mean(pts,axis=0)
        std = np.std(pts,axis=0)
        pts = np.divide(pts - mean,std,where=(std !=0))
        #print(mean.shape)
        self.pts_to_hdf5(pts,"data_test")
        self.create_timefreq_dataset("data_test")
        self.resize_data("data_test")
        self.imgs_to_hdf5("data_test")

    def create_spectrogram_live_test(self,data,folderpath = "spectrogram"):
        if not os.path.isdir(folderpath):
            os.mkdir(folderpath)
        print(data.shape)
        sig_y = data[:,1::3]

        bar = tqdm(total=sig_y.shape[1]-1)
        bar.set_postfix(ordered_dict={"Point": 0})
        for point_index in range(sig_y.shape[1]):
            self.timefreq(sig_y[:,point_index],save_path=os.path.join(folderpath,"POINT_"+str(point_index)+".png"))
            bar.set_postfix(ordered_dict={"Point": point_index})
            bar.update(1)
        bar.close()

            
        
if __name__ == "__main__":
    data_analysis = Utils()
    data = data_analysis.load("data_train.h5")
    #print(data.shape)
    sig_x = data[0,:200,20]#[np.nonzero(data[0,:200,0])]
    sig_y = data[0,:2000,21]#[np.nonzero(data[0,:200,1])]
    #print(sig_y)
    #print(sig.shape)
    #print(sig[:,0])
    #plt.plot(sig_y)
    #plt.show()
    #data_analysis.timefreq(sig_y)

    ###################################
    #       Train
    ###################################

    #data_test = np.zeros((6,10,1000,75))
    data_test = data_analysis.load_spectrum("dataset_test_new.h5")
    print(data_test.shape)
    data_analysis.pts_to_hdf5(data_test,"data_test_new")
    data_analysis.create_timefreq_dataset("data_test_new")
    data_analysis.resize_data("data_test_new")
    data_analysis.imgs_to_hdf5("data_test_new")
        
    ###################################
    #       Test
    ###################################

    #data_analysis.create_data_test()
    

