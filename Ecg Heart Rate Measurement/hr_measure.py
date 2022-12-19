import serial.tools.list_ports
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
import time
from scipy.signal import butter, lfilter
import neurokit2 as nk
from ecgdetectors import Detectors
import pandas as pd
import heartpy as hp
from heartpy.datautils import rolling_mean
from scipy.interpolate import interp1d
import wfdb
from wfdb import processing
import scipy.signal as sig
import math
import scipy.signal as scs
import peakutils
from scipy.signal import find_peaks




ports=serial.tools.list_ports.comports()
serialInst=serial.Serial()

portVar="COM4"
serialInst.baudrate=9600 
serialInst.port=portVar
serialInst.open()

my_ecg_data=[]
init_t=0
my_time_data=[init_t]
numby=0

hr_1=[]
hr_2=[]
hr_3=[]
hr_4=[]
hr_5=[]
hr_6=[]
hr_7=[]
hr_8=[]
hr_9=[]
hr_10=[]
hr_11=[]
hr_12=[]
hr_13=[]

br_1=[]
br_2=[]
br_3=[]
br_4=[]

iter_loop=0

while iter_loop<=4:
    
    if serialInst.in_waiting:
        start_time = time.time()
        packet=serialInst.readline()
        dt=time.time() - start_time
        
        packet_reduced=packet.decode("utf-8")
        # print(packet_reduced)
        my_ecg_data.append(float(packet_reduced))
        newtime=my_time_data[-1]+dt
        my_time_data.append(newtime)
        numby+=1
        len_of_data=3000
        if len(my_ecg_data)==len_of_data:
            # Thresholding sampling frequency distribution to remove possible infinities
            daty=my_ecg_data
            init=1
            thresh_samp_freq=175000
            threshold=0
            explode=daty.shape[0]
            ecg_ref_data=daty[init:explode,1]
            ecg=ecg_ref_data
            timer=daty[1:(explode-init)+1,2]
            inst_fs=1/np.diff(timer)
            inst_fs[inst_fs > thresh_samp_freq] = threshold

            # Appropiate Windowing for most narrow sampling frequency distribution
            segment_length=2000
            sub_segment_length=1000
            err_record=[]

            for i in range(len(inst_fs)-segment_length+1):
                temp_inst_fs=inst_fs[i:i+segment_length]
                err_record.append(np.std(temp_inst_fs))

            val, idx = min((val, idx) for (idx, val) in enumerate(err_record))
            init=idx
            explode=idx+segment_length
            ecg_ref_data=daty[init:explode,1]
            ecg=ecg_ref_data


            #Appropiate sub-windowing to find least perturbed segement, with the metric that
            # if I reduce noise in the signal, then the inferred heart rate should be same.
            err_record=[]
            for j in range(len(ecg)-sub_segment_length+1):
                temp_ecg=ecg[j:j+sub_segment_length]
                rlocser,_ = find_peaks(temp_ecg, distance=70,height=0.5)
                n_locs_old=len(rlocser)
                length=11
                order=2
                smoothed_temp_ecg=savgol_filter(temp_ecg,length, order)
                rlocser,_ = find_peaks(smoothed_temp_ecg, distance=70,height=0.5)
                n_locs_new=len(rlocser)
                err_record.append(abs(n_locs_new-n_locs_old))


            val, idx = min((val, idx) for (idx, val) in enumerate(err_record))
            init=idx
            explode=idx+sub_segment_length
            ecg_ref_data_f=ecg[init:explode]
            ecg=ecg_ref_data_f



            timer=daty[1:(explode-init)+1,2]
            inst_fs=1/np.diff(timer)
            inst_fs[inst_fs > thresh_samp_freq] = threshold


            fs_mean=np.mean(inst_fs)
            fs_std=np.std(inst_fs)
            fs = (np.mean(inst_fs)*6)
            # fs=(fs_std)/3+fs_mean
            length=11
            order=7

            detrended_ecg=signal.detrend(ecg)
            denoised_detrended_ecg=savgol_filter(detrended_ecg,length, order)
            def butter_highpass(cutoff, fs, order=5):
                nyq = 0.5 * fs
                normal_cutoff = cutoff / nyq
                b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
                return b, a

            b, a = butter_highpass(14/60, 10, order=2)
            final_ecg = signal.filtfilt(b, a, denoised_detrended_ecg)

            iter=1000
            init=1
            bp=[]
            fser=[]
            for i in range(iter):
                n=np.random.rand(1)[0]
                escape=0
                lamb=2
                amp=200
                max=amp+escape
                min=amp-lamb*escape
                fs = min+n*(max-min)
                wd, m = hp.process(final_ecg, sample_rate = fs)
                l=m['bpm']
                if not math.isnan(l):
                    bp.append(m['bpm'])
                    fser.append(fs)

            print("BPM: ",np.mean(np.array(bp)))