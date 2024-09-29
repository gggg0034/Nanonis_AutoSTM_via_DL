# Zhiwen Zhu 2024/04/22    zhiwenzhu@shu.edu.cn
import json
import os
import pickle
import random
import shutil
import sys
import threading
import time
from math import sqrt
from multiprocessing import Process, Queue

from PyQt5.QtCore import QThread
from matplotlib.animation import FuncAnimation

from PyQt5 import QtCore, QtGui, QtWidgets
from core import NanonisController
import cv2
import keyboard
import matplotlib.pyplot as plt
import numpy as np

from core import NanonisController
from DQN.agent import *
from EvaluationCNN.detect import predict_image_quality
from modules.TipPath_Fuc import *
from mol_segment.detect import Segmented_image
from PyQt_GUI import create_pyqt_app
from tasks.LineScanchecker import *



class  Mustard_AI_Nanonis(NanonisController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))    # record the start time of the scan
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scan_start_time = None
        self.total_scan_time = 0.0
        self.per_scan_time = 0.0
        self.ScandataQueue_1 = Queue(5)
        self.tipdataQueue = Queue(5)
        self.Tipshaper_signal = 0
        self.iftipshaper = 0
        self.circle_list = []                       # the list to save the circle which importantly to the tip path
        self.circle_list_save = []                  # the list to save the circle as npy file
        self.nanocoodinate_list = []                # the list to save the coodinate which send to nanonis directly
        self.visual_circle_buffer_list = []

        self.good_image_count = 0
        self.count_choose = 100
        self.frame_move_signal = 0
        self.line_scan_signal = 0
        self.scale_image_for = []
        self.scale_image_back = []
        self.scale_signal = 0
        self.tippath_stop_event = threading.Event()
        self.batch_stop_event = threading.Event()
        self.savetip_stop_event = threading.Event()
        self.linescan_stop_event = threading.Event()
        self.mode = None

        self.trajectory_buffer_size = 5             # the size of the trajectory buffer
        self.trajectory_state_list = deque([], maxlen=self.trajectory_buffer_size)                   # the list to save the one step in memo
        self.trajectory_action_list = deque([], maxlen=self.trajectory_buffer_size)
        self.trajectory_reward_list = deque([], maxlen=self.trajectory_buffer_size)
        self.trajectory_next_state_list = deque([], maxlen=self.trajectory_buffer_size)
        
        self.line_scan_change_times = 0             # initialize the line scan change times to 0
        self.episode_count = 0                      # initialize the episode count to 0
        self.AdjustTip_flag = 0                     # 0: the tip is not adjusted, 1: the tip is adjusted just now

        self.Scan_edge = '50n'                    # set the initial scan square edge length                           30pix  ==>>  30nm   80
        self.scan_square_Buffer_pix = 208            # scan pix
        self.plane_edge = "2u"                       # plane size repersent the area of the Scanable surface  2um*2um    2000pix  ==>>  2um

        self.Z_fixed_scan_time = 10                   # if the scan is after the Z fixed, how many seconds will the scan cost
        self.without_Z_fixed_scan_time = 10           # if the scan is without the Z fixed, how many seconds will the scan cost
        self.linescan_max_min_threshold = '10000p'      # the max and min threshold , if the line scan data is out of the threshold
        self.scan_max_min_threshold = '10n'      # the max and min threshold , if the line scan data is out of the threshold
        self.len_threshold_list = 10
        self.threshold_list = deque([], maxlen=self.len_threshold_list)     # the list to save the threshold of the line scan data, if threshold_list is full with 1, the scan will be skiped
        self.skip_list = deque([], maxlen=10)         # the list to save the skip flag, if the skip_list is full with 1, gave the tip a aggressive tip shaper
        self.skip_flag = 0                            # 0: the scan is not skiped, 1: the scan is skiped
        self.aggressive_tip_flag = 0                  # 0: the tip is not aggressive, 1: the tip is aggressive

        self.real_scan_factor = 0.8                  # the real scan area is 80% of the scan square edge length

        self.tip_move_mode = 0                       # 0: continuous_random_move_mode, 1: para_or_vert_move_mode

        self.line_scan_activate = 1                  # 0: line scan is not activate, 1: line scan is activate

        self.equalization_alpha = 0.3                # the alpha of the equalization

        self.scan_qulity_threshold = 0.7             # the threshold of the scan qulity, higher means more strict for the scan qulity

        self.scan_qulity = 1                         # 1: the scan qulity is good, 0: the scan qulity is bad        

        self.nanonis_mode = 'real'                   # the nanonis mode, 'demo' or 'real'

        self.action_space = 6                             # the action space of the DQN

        self.DQN_Agent = tip_shaper_DQN_agent(self.action_space, self.device)  # the DQN agent
        self.memory = ReplayMemory()                # the memory to save the trajectory, a complete trajectory is an episode

        if self.nanonis_mode == 'demo':
            self.signal_channel_list = [0, 30]               # the mode channel list, 0 is Current, 30 is Z_m
        elif self.nanonis_mode == 'real':
            self.signal_channel_list = [0, 8, 14]            # the mode channel list, 0 is Current, 8 is Bias, 14 is Z_m
        
        self.agent_upgrate = 1                              # 0:the DQN model will not be upgrated, 1: the DQN model will be upgrated    

        self.quality_model_path = './EvaluationCNN/CNN.pth'  # the path of the quality model weights

        self.segment_model_path = './mol_segment/unet_modelsingle.pth'  # the path of the segment model weights

        self.DQN_init_moel_path = './DQN/pre-train_DQN.pth'     # the path of the pre-train DQN model

        self.DQN_save_main_path = './DQN/train_results/'+ self.start_time        # the path of the DQN model saving

        self.main_data_save_path = './log'

        self.log_path =  self.main_data_save_path + '/' + self.start_time

        self.memory_path = './DQN/memory/' + self.start_time    # the path of the memory saving

        self.Scan_edge_SI = self.convert(self.Scan_edge)                        # the scan edge in SI unit   Scan_edge_SI = 30 * 1e-9

        self.plane_size = int(self.convert(self.plane_edge)*10**9)            # the plane size in pix  plane_size = 2000
        self.scan_square_edge = int(self.convert(self.Scan_edge)*10**9)            # the scan square edge in pix  scan_square_edge = 30

        self.tip_path_img = np.ones((self.plane_size, self.plane_size, 3), np.uint8) * 255  # the tip path image

        self.inter_closest = (round(self.plane_size/2), round(self.plane_size/2))                          # initialize the inter_closest
        # R_init = round(scan_square_edge*(math.sqrt(2))*1.5)
        self.R_init = self.scan_square_edge -1                                                        # initialize the Radius of tip step
        self.R_max = self.R_init*3
        self.R_step = int(0.5*self.R_init)
        self.R = self.R_init


        # initialize the other parameters that appear in the function
        self.Scan_data = {}                                                                         # the dictionary to save the scan data
        self.image_for = None   # 2D nparray the image of the scan data, have been nomalized and linear background
        self.image_back = None
        self.equalization_for = None    # the equalization image of the image_for and image_back
        self.equalization_back = None
        self.image_for_tensor = None    # the tensor of the image, 4 dimension, [1, 1, 256, 256]
        self.image_back_tensor = None   
        self.image_save_time = None         # when the image is saved in log
        self.npy_data_save_path     =  self.log_path + '/' + 'npy'                                 # self.log_path = './log/' + self.start_time
        self.image_data_save_path   =  self.log_path + '/' + 'image'
        self.segmented_image_path = None    # the path of the segmented image saving
        self.nemo_nanocoodinate = None      # the nanocoodinate of the nemo point, the format is SI unit
        self.coverage = None                # the moleculer coverage of the image
        self.line_start_time = None
        self.episode_start_time = None    # the start time of the episode
        
        # initialize the queue, the Queue is used to communicate between different threads
        self.lineScanQueue = Queue(5)    # lineScan_data_producer → lineScan_data_consumer
        self.lineScanQueue_back = Queue(5)    # lineScan_data_consumer → lineScan_data_producer
        self.ScandataQueue = Queue(5)    # batch_scan_producer → batch_scan_consumer

        self.tippathvisualQueue = Queue(5)   # main program  → tip_path_visualization

        self.PyQtimformationQueue = Queue(5) # PyQt_GUI → main program
        self.PyQtPulseQueue = Queue(5)       # PyQt_GUI → HandPulsethreading
        self.PyQtTipShaperQueue = Queue(5)   # PyQt_GUI → TipShaperthreading

        self.scanqulityqueue = Queue() # TipShaperthreading → main program


    def is_serializable(self, value):
        """ Attempt to serialize the value, return True if serializable, False otherwise. """
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False
        
    def save_checkpoint(self):
        """ Save the current state of serializable instance attributes to a file. """
        serializable_attrs = {k: v for k, v in self.__dict__.items() if self.is_serializable(v)}
        filename = os.path.join(self.log_path,'checkpoint.json')
        with open(filename, 'w') as file:
            json.dump(serializable_attrs, file)
        
            
    def load_checkpoint(self):
        """Loads class attributes from a json file and updates the instance."""
        checkpoint_json = get_latest_checkpoint(self.main_data_save_path)
        with open(checkpoint_json, 'r') as file:
            data = json.load(file)
            self.__dict__.update(data)
        checkpoint_tip_img = get_latest_checkpoint(self.main_data_save_path,checkpoint_name="tip_path.jpg")
        self.tip_path_img = cv2.imread(checkpoint_tip_img, cv2.IMREAD_COLOR)
    # def a function to initialize the nanonis controller, set the tip to the center of the scan frame
    # mode = 'new' : the tip is initialized to the center and create a new log folder, mode = 'latest' : load the latest checkpoint
    
    def tip_init(self, mode = 'new'):
        self.mode = mode
        if mode == 'new':
            self.ScanPause()
            self.ZCtrlOnSet()
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)

        elif mode == 'latest':
            self.ScanPause()
            self.ZCtrlOnSet()        # TODO
            self.load_checkpoint()
        
    def DQN_init(self, mode = 'new'):
        self.mode = mode
        if mode == 'new':
            self.DQN_Agent.load_model(self.DQN_init_moel_path)
        elif mode == 'latest':
            wight_list = time_trajectory_list('./DQN/train_results/',file_extension='.pth')
            self.DQN_Agent.load_model(wight_list[-1])
        elif mode == 'agent':
            wight_list = time_trajectory_list('./DQN/train_results/',file_extension='.pth')
            self.DQN_Agent.load_model(wight_list[-1])
            self.agent_upgrate = 0
    def DQN_upgrate(self):
            self.DQN_Agent.optimize_model()                                              # optimize the model
            self.DQN_Agent.update_target_net()                                               # update the target network
            # time for save_model
            DQN_model_save_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))                                    # record the start time of the Pulse
            self.DQN_Agent.save_model(self.DQN_save_main_path + '/' + DQN_model_save_time + '_DQN.pth')                          # save the model


    # activate all the threads of monitor   
    def monitor_thread_activate(self):
        Safe_Tip_thread = threading.Thread(target=self.SafeTipthreading, args=('5n', 100),daemon=True)                                          # the SafeTipthrehold is 5n, the safe_withdraw_step is 100
        tip_visualization_thread = threading.Thread(target=self.tip_path_visualization, daemon=True)                                            # the tip path visualization thread
        batch_scan_consumer_thread = threading.Thread(target=self.batch_scan_consumer, daemon=True)                                             # the batch scan consumer thread
        Hand_Pulse_thread = threading.Thread(target=self.HandPulsethreading, args=(self.PyQtPulseQueue,), daemon=True)                          # the HandPulse thread
        TipShaper_thread = threading.Thread(target=self.Tipshaperthreading, args=(self.PyQtTipShaperQueue,self.scanqulityqueue), daemon=True)   # the TipShaper thread
        instruct_thread = threading.Thread(target=self.instructthreading, args=(self.PyQtimformationQueue,), daemon=True)                       # the instruct thread

        # if tip_visualization_thread.is_alive():
        #     Safe_Tip_thread.join()
        #     tip_visualization_thread.join()
        #     batch_scan_consumer_thread.join()
        #     self.tippath_stop_event = 0
        #     self.batch_stop_event = 0
        #     self.savetip_stop_event = 0
        #     Safe_Tip_thread.start()
        #     tip_visualization_thread.start()
        #     batch_scan_consumer_thread.start()
        # else:
        #     Safe_Tip_thread.start()
        #     tip_visualization_thread.start()
        #     batch_scan_consumer_thread.start()
        Safe_Tip_thread.start()
        tip_visualization_thread.start()
        batch_scan_consumer_thread.start()
        Hand_Pulse_thread.start()
        TipShaper_thread.start()
        instruct_thread.start()




    def SafeTipthreading(self, SafeTipthrehold = '5n', safe_withdraw_step = 100):
        print('The tipsafe monitoring activated')
        # if the type of SafeTipthrehold is string, convert it to float
        if type(SafeTipthrehold) == str:
            SafeTipthrehold = self.convert(SafeTipthrehold)

        current_list = []
        # while True:
        while not self.savetip_stop_event.is_set():
            time.sleep(0.5)
            try:
                current = self.CurrentGet()
            except:
                current = 0
            current_list.append(current)
            # print(current_list)
            if len(current_list) >= 8:
                current_list.pop(0)
                # if all the current absoulte value is bigger than SafeTipthrehold, stop the scan and withdraw the tip than move Motor Z- 50? steps
                if all(abs(current) > SafeTipthrehold for current in current_list):
                    #raise ValueError('The tunneling current is too large, the tip protection is activated, and the scan stops.')
                    print('The tunneling current is too large, the tip protection is activated.')
                    self.ScanStop()
                    self.Withdraw()
                    self.MotorMoveSet('Z-', safe_withdraw_step)
                    current_list = []
                    print('The tip is withdrawed')
                    break
    
    def HandPulsethreading(self, PyQtPulseQueue):
        print("Pulsethreading activated")
        number_of_HandPulse = 0
        while True:
            if not PyQtPulseQueue.empty():
                item = PyQtPulseQueue.get()  # Retrieve the next item from the queue.
                # Check for a special character and perform an action.
                if item[0] == "Pulse":
                    Pulse_start_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))                                    # record the start time of the Pulse
                    self.BiasPulse(item[1], width=0.05)
                    # check the save path exist or not
                    if not os.path.exists(self.log_path):
                        os.makedirs(self.log_path)
                    np.save(self.log_path + '/' + Pulse_start_time  + '_Pulse_' + str(number_of_HandPulse).zfill(5) +'.npy', [item[0],item[1] ,number_of_HandPulse], allow_pickle = True)
                    number_of_HandPulse += 1
                    print(f"Pulse {item[1]}V")
    
    def Tipshaperthreading(self, PyQtTipShaperQueue,scanqulityqueue):
        print("TipShaperthreading activated")
        number_of_Tipshaper = 0
        while True:
            item = PyQtTipShaperQueue.get()  # Retrieve the next item from the queue.
            # Check for a special character and perform an action.
            # scanqulityqueue.put(0)
            if item[0] == "TipShaper":
                scanqulityqueue.put(0)
                try:
                    Tipshaper_start_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))                                    # record the start time of the Pulse
    
                    self.TipShaper(TipLift = item[1])
    
                    self.ScanStop()  # stop the scan so that the batch scan can be interrupted
                    # check the save path exist or not
                    if not os.path.exists(self.log_path):
                        os.makedirs(self.log_path)
                    np.save(self.log_path + '/' + Tipshaper_start_time + '_TipShaper_' + str(number_of_Tipshaper).zfill(5) +'.npy', [item[0],item[1] ,number_of_Tipshaper], allow_pickle = True)
                    number_of_Tipshaper += 1
                    print(f"TipShaper {item[1]}")
                except:
                    print('The TipShaper is not successful')
                print("install the scan , move to the next point...")
                # nanonis.ScanStart()

    def instructthreading(self, PyQtimformationQueue):
        """
        Thread function to monitor a queue for special characters.
        :param q: The queue to monitor.
        :param identifier: A string to identify which queue (and hence which button) this thread is monitoring.
        """
        while True:
            item = PyQtimformationQueue.get()  # Retrieve the next item from the queue.
            # Check for a special character and perform an action.
            if item == "Home":
                self.ScanPause()
                self.ScanFrameSet(0, 0, "100n", "100n", angle=0)
                self.TipXYSet(0, 0)
                print(f"set Tip and Scan frame to center")
                PyQtimformationQueue.put('stop whole scan')
                break
            if item == "skip to next":
                self.ScanStop()
                print(f"skip to next")
                time.sleep(0.5)
                self.ScanStart()

    def batch_scan_producer(self,  Scan_posion = (0.0 ,0.0), Scan_edge = "30n", Scan_pix = 304 ,  angle = 0, ):
        if self.skip_flag == 1:
            print('creating skip data...')
            Scan_data_for = {'data': np.ones((Scan_pix, Scan_pix), np.uint8) * 0.1,
                'row': Scan_pix,
                'col': Scan_pix,
                'scan_direction': 0,
                'channel_name': 'Z (m)',
                }
            Scan_data_back = {'data': np.ones((Scan_pix, Scan_pix), np.uint8) * 0.1,
                'row': Scan_pix,
                'col': Scan_pix,
                'scan_direction': 0,
                'channel_name': 'Z (m)',
                }
        else:
            print('Scaning the image...')
            # ScanBuffer = self.ScanBufferGet()
            self.ScanBufferSet(Scan_pix, Scan_pix, self.signal_channel_list) # 14 is the index of the Z_m channel in real scan mode , in demo mode, the index of the Z_m channel is 30
            self.ScanPropsSet(Continuous_scan = 2 , Bouncy_scan = 2,  Autosave = 1, Series_name = ' ', Comment = 'inter_closest')# close continue_scan & bouncy_scan, but save all data
            self.ScanFrameSet(Scan_posion[0], Scan_posion[1], Scan_edge, Scan_edge, angle=angle)

            self.ScanStart()
            t = (self.ScanSpeedGet()['Forward time per line'] + self.ScanSpeedGet()[
                'Backward time per line']) * \
                self.ScanBufferGet()['Lines']
            self.calculate_total_scan_time(t)
            self.ScanStop()
            # time.sleep(1)
            # while self.ScanStatusGet() == 1: # detect the scan status until the scan is complete.
            #     #   !!!   Note：Do not use self.WaitEndOfScan() here   !!!!, it will block the program!!!!!!
            #     time.sleep(0.5)

            # self.WaitEndOfScan() # wait for the scan to be complete

            try:   # some times the scan data is not successful because of the TCP/IP communication problem
                Scan_data_for = self.ScanFrameData(self.signal_channel_list[-1], data_dir=1)
            except: #if is not successful, set fake data
                Scan_data_for = {'data': np.ones((Scan_pix, Scan_pix), np.uint8) * 0.1,
                                'row': Scan_pix,
                                'col': Scan_pix,
                                'scan_direction': 0,
                                'channel_name': 'Z (m)',
                                }
            time.sleep(1)
            try:
                Scan_data_back = self.ScanFrameData(self.signal_channel_list[-1], data_dir=0)
            except:
                Scan_data_back = {'data': np.ones((Scan_pix, Scan_pix), np.uint8) * 0.1,
                                'row': Scan_pix,
                                'col': Scan_pix,
                                'scan_direction': 0,
                                'channel_name': 'Z (m)',
                                }
            # if the first element and the last element of the Scan_data_for and Scan_data_back is NaN, the scan is not successful
            if np.isnan(Scan_data_for['data'][0][0]) or np.isnan(Scan_data_for['data'][-1][-1]) or np.isnan(Scan_data_back['data'][0][0]) or np.isnan(Scan_data_back['data'][-1][-1]):
                Scan_data_for['data'][np.isnan(Scan_data_for['data'])] = 0
                Scan_data_back['data'][np.isnan(Scan_data_back['data'])] = 0

        self.image_for = linear_normalize_whole(Scan_data_for['data'])                     # image_for and image_back are 2D nparray
        self.image_back = linear_normalize_whole(Scan_data_back['data'])

        self.image_for = images_equalization(self.image_for, alpha=self.equalization_alpha)
        self.image_back = images_equalization(self.image_back, alpha=self.equalization_alpha)

        self.image_for_tensor = torch.tensor(self.image_for, dtype=torch.float32, device=self.device).unsqueeze(0)  # for instance tensor.shape are [1, 1, 256, 256]  which is for DQN
        self.image_back_tensor = torch.tensor(self.image_back, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.Scan_data = {'Scan_data_for':Scan_data_for, 'Scan_data_back':Scan_data_back}

        self.ScandataQueue.put(self.Scan_data)
        time.sleep(0.5)
        self.ScandataQueue_1.put(self.Scan_data)
        # put the batch scan data into the queue, blocking if Queue is full
        print('Scaning complete! \n ready to save...')
        return self.Scan_data

    def lineScan_data_producer(self,angle=0, scan_continue_time = 90):
            
        end_time = time.time() + scan_continue_time

        self.ScanPropsSet(Continuous_scan = 1, Bouncy_scan = 1,  Autosave = 3, Series_name = ' ', Comment = 'LineScan')
        self.lineScanmode(3, angle=angle)
        Z_m_index = self.signal_channel_list[-1]

        self.ScanStart()
        self.WaitEndOfScan()
        while True:
            lineScandata = self.lineScanGet(Z_m_index)
            if self.lineScanQueue.full():
                self.lineScanQueue.get()
            if time.time() > end_time or self.skip_flag or self.linescan_stop_event.is_set(): # if the time is over or the scan is skiped, break the loop
                self.lineScanQueue.put('end')
                
                if self.aggressive_tip_flag: # aggressive_tip_flag = 1 means the scan is skiped 5 times continuously, so that might be the super terable tip!
                    time.sleep(1)
                    self.TipShaper(TipLift= '-6n')
                    time.sleep(1)
                    self.BiasPulse(-6, width=0.05)
                    time.sleep(1)
                    self.aggressive_tip_flag = 0    # reset the aggressive_tip_flag
                break

            
            self.lineScanQueue.put(lineScandata)

    def batch_scan_consumer(self):
        
        self.npy_data_save_path     =  self.log_path + '/' + 'npy'                                 # self.log_path = './log/' + self.start_time
        self.image_data_save_path   =  self.log_path + '/' + 'image'
        self.equalization_save_path = self.log_path + '/' + 'equalize'        
        
        if not os.path.exists(self.equalization_save_path):                                 # check the save path exist or not
            os.makedirs(self.equalization_save_path)

        if not os.path.exists(self.npy_data_save_path):
            os.makedirs(self.npy_data_save_path)

        if not os.path.exists(self.image_data_save_path):
            os.makedirs(self.image_data_save_path)

        # while True:
        while not self.batch_stop_event.is_set():
            # time.sleep(1)
            if not self.ScandataQueue.empty():
                Scan_data = self.ScandataQueue.get()
                Scan_data_for = Scan_data['Scan_data_for']['data']
                Scan_data_back = Scan_data['Scan_data_back']['data']
                # preprocess the scan data, and save the scan data and image
                image_for = linear_normalize_whole(Scan_data_for)
                image_back = linear_normalize_whole(Scan_data_back)

                equalization_for = images_equalization(image_for, alpha=self.equalization_alpha)
                equalization_back = images_equalization(image_back, alpha=self.equalization_alpha)      # equalize the image

                self.image_save_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))

                npy_data_save_path_for      = self.npy_data_save_path + '/' + 'Scan_data_for_'+ self.image_save_time +'.npy'
                npy_data_save_path_back     = self.npy_data_save_path + '/' + 'Scan_data_back_'+ self.image_save_time +'.npy'
                image_data_save_path_for    = self.image_data_save_path + '/' + 'Scan_data_for'+ self.image_save_time +'.png'
                image_data_save_path_back   = self.image_data_save_path + '/' + 'Scan_data_back'+ self.image_save_time +'.png'
                equalization_save_path_for  = self.equalization_save_path + '/' + 'Scan_data_for'+ self.image_save_time +'.png'
                equalization_save_path_back = self.equalization_save_path + '/' + 'Scan_data_back'+ self.image_save_time +'.png'

                cv2.imwrite(equalization_save_path_for, equalization_for)
                cv2.imwrite(equalization_save_path_back, equalization_back)     # save the equalization image                
                np.save(npy_data_save_path_for, Scan_data_for, allow_pickle = True)
                np.save(npy_data_save_path_back, Scan_data_back, allow_pickle = True)
                cv2.imwrite(image_data_save_path_for, image_for)                # save the image
                cv2.imwrite(image_data_save_path_back, image_back)              # save the image
                # cv2.namedWindow('image_for', cv2.WINDOW_NORMAL)
                # cv2.namedWindow('image_back', cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('image_for', 400, 400)
                # cv2.resizeWindow('image_back', 400, 400)
                # cv2.imshow('image_for', image_for)
                # cv2.imshow('image_back', image_back)
            cv2.waitKey(100)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

    def lineScan_data_consumer(self):
        number_of_line_scan = 0
        self.line_start_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
        lineScan_save_path = self.log_path + '/' + 'lineScan' +'/'
        if not os.path.exists(lineScan_save_path + self.line_start_time + '_line_scan_' + str(self.line_scan_change_times).zfill(5)):
            
            os.makedirs(lineScan_save_path + self.line_start_time + '_line_scan_' + str(self.line_scan_change_times).zfill(5))

        
        while True:
            lineScandata_1 = self.lineScanQueue.get()
            # print('lineScan_data_consumer_works')
            
            # time_per_line = nanonis.ScanSpeedGet()['Forward time per line']
            # time.sleep(time_per_line) # wait for the line scan data to be collected

            time.sleep(0.153) # wait for the line scan data to be collected
            if lineScandata_1 == 'end':                                                             # if the line scan data producer is over, stop the line scan data consumer than draw the line scan data
                self.line_scan_signal = 0
                print('lineScan complete! ')
                
                break                                                                               # end the lineScan_data_consumer
            if len(self.nanocoodinate_list) > 1:                                                         # if the nanocoodinate list have more than one nanocoodinate, save the line scan data
                lineScandata_1_for = fit_line(lineScandata_1['line_Scan_data_for'])                         # t-1 line scan data for
                lineScandata_1_back = fit_line(lineScandata_1['line_Scan_data_back'])                       # t-1 line scan data back
                # np.save(lineScan_save_path + self.line_start_time + '_line_scan_' + str(self.line_scan_change_times).zfill(5) +'/lineScandata_'+str(number_of_line_scan).zfill(5) +'.npy', 
                #         [self.AdjustTip_flag ,self.nanocoodinate_list[-1], self.nanocoodinate_list[-2] ,lineScandata_1_for, lineScandata_1_back])
                # print('lineScandata_{}_for.npy'.format(number_of_line_scan) + ' is saved')
                
                # if the line scan max - min is bigger than 500, set the skip flag to 1
                if linescan_max_min_check(lineScandata_1_for) > self.convert(self.linescan_max_min_threshold):
                    self.threshold_list.append(1)
                else:
                    self.threshold_list.append(0)
                # if the threshold_list is full with 1, the scan will be skiped
                if len(self.threshold_list) == self.len_threshold_list and all(threshold == 1 for threshold in self.threshold_list):
                    self.skip_flag = 1      # switch the skip flag to 1, because the line scan data is out of the threshold several times continuously
                    self.threshold_list.clear() # clear the threshold list, recount the threshold
                
                self.skip_list.append(self.skip_flag) # add the skip flag to the skip list
                    
                # if len(self.skip_list) ==10 and all(skip == 1 for skip in self.skip_list):


                if len(self.skip_list) >= 5 and all(skip == 1 for skip in self.skip_list):
                    self.aggressive_tip_flag = 1
                    self.skip_list.clear()      # clear the skip list, recount the skip flag
                    print('The line scan is skiped')

                number_of_line_scan += 1

    # def a function that content the line scan producer and consumer
    def line_scan_thread_activate(self):
        if self.line_scan_activate == 1:

            if self.AdjustTip_flag == 1:
                scan_continue_time = self.Z_fixed_scan_time
            else:
                scan_continue_time = -(1/200)*self.R**2 + 1.1*self.R
            print('Line scan will continue for {} seconds'.format(scan_continue_time))
            t1 = threading.Thread(target = self.lineScan_data_producer, args=(0, scan_continue_time), daemon=True)               # lunch the line scan data producer
            t2 = threading.Thread(target = self.lineScan_data_consumer, daemon=True)                                             # lunch the line scan data consumer
            t1.start()
            t2.start()
            self.line_scan_signal = 1
            print('line scanning...')
            t1.join()
            t2.join()

    def tip_path_visualization(self):
        if self.mode == 'new':
            self.start_time = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time()))
            self.ScandataQueue_1 = Queue(5)
            self.tipdataQueue = Queue(5)
            self.circle_list = []  # the list to save the circle which importantly to the tip path
            self.circle_list_save = []  # the list to save the circle as npy file
            self.nanocoodinate_list = []  # the list to save the coodinate which send to nanonis directly
            self.visual_circle_buffer_list = []
            self.line_scan_change_times = 0  # initialize the line scan change times to 0
            self.episode_count = 0  # initialize the episode count to 0
            self.AdjustTip_flag = 0
            self.len_threshold_list = 10
            self.threshold_list = deque([],
                                        maxlen=self.len_threshold_list)  # the list to save the threshold of the line scan data, if threshold_list is full with 1, the scan will be skiped
            self.skip_list = deque([],
                                   maxlen=10)  # the list to save the skip flag, if the skip_list is full with 1, gave the tip a aggressive tip shaper
            self.skip_flag = 0  # 0: the scan is not skiped, 1: the scan is skiped
            self.aggressive_tip_flag = 0
            self.agent_upgrate = 1
            self.log_path = self.main_data_save_path + '/' + self.start_time
            self.memory_path = './DQN/memory/' + self.start_time
            self.Scan_edge_SI = self.convert(self.Scan_edge)  # the scan edge in SI unit   Scan_edge_SI = 30 * 1e-9

            self.scan_square_edge = int(
                self.convert(self.Scan_edge) * 10 ** 9)  # the scan square edge in pix  scan_square_edge = 30

            self.tip_path_img = np.ones((self.plane_size, self.plane_size, 3), np.uint8) * 255  # the tip path image


            self.R_init = self.scan_square_edge - 1  # initialize the Radius of tip step
            self.R_max = self.R_init * 3
            self.R_step = int(0.5 * self.R_init)
            self.R = self.R_init

            # initialize the other parameters that appear in the function
            self.Scan_data = {}  # the dictionary to save the scan data
            self.image_for = None  # 2D nparray the image of the scan data, have been nomalized and linear background
            self.image_back = None
            self.equalization_for = None  # the equalization image of the image_for and image_back
            self.equalization_back = None
            self.image_for_tensor = None  # the tensor of the image, 4 dimension, [1, 1, 256, 256]
            self.image_back_tensor = None
            self.image_save_time = None  # when the image is saved in log
            self.npy_data_save_path = self.log_path + '/' + 'npy'  # self.log_path = './log/' + self.start_time
            self.image_data_save_path = self.log_path + '/' + 'image'
            self.equalization_save_path = self.log_path + '/' + 'equalize'
            self.segmented_image_path = None  # the path of the segmented image saving
            self.nemo_nanocoodinate = None  # the nanocoodinate of the nemo point, the format is SI unit
            self.coverage = None  # the moleculer coverage of the image
            self.line_start_time = None
            self.episode_start_time = None  # the start time of the episode

            # initialize the queue, the Queue is used to communicate between different threads
            self.lineScanQueue = Queue(5)  # lineScan_data_producer → lineScan_data_consumer
            self.lineScanQueue_back = Queue(5)  # lineScan_data_consumer → lineScan_data_producer
            self.ScandataQueue = Queue(5)  # batch_scan_producer → batch_scan_consumer

            self.tippathvisualQueue = Queue(5)
            self.scan_start_time = None
            self.per_scan_time = 0
            self.total_scan_time = 0
        square_color_hex = "#BABABA"                        # good image color
        square_bad_color_hex = "#FE5E5E"                    # bad image color
        line_color_hex = "#8AAEFA"                          # tip path line color
        border_color_hex = "#64FF00"                        # the border color of the whole plane
        scan_border_color_hex = "#FFC8CB"                   # usually the area of the 70% of the whole plane

        sample_bad_color_hex = "#FF6E6E"       #FF6E6E       #000000      # the color of the bad LineScan data

        color_max = "#385723"#385723  #C55A11
        color_min = "#C5E0B4"#C5E0B4  #2E75B6

        border_color = Hex_to_BGR(border_color_hex)
        scan_border_color = Hex_to_BGR(scan_border_color_hex)
        square_good_color = Hex_to_BGR(square_color_hex)
        square_bad_color = Hex_to_BGR(square_bad_color_hex)
        sample_bad_color = Hex_to_BGR(sample_bad_color_hex)
        line_color = Hex_to_BGR(line_color_hex)        

        # cv2.namedWindow('Tip Path', cv2.WINDOW_KEEPRATIO)
        # cv2.resizeWindow('Tip Path', 800, 800)
        

        # creat a 400*400 pix white image by numpy array
        cv2.rectangle(self.tip_path_img, (0, 0), (self.plane_size, self.plane_size), border_color, 10) # draw the border of the plane, which is the same color as the nanonis scan border on the scan controloer
        
        scan_border_left_top =  round(self.plane_size/2 * (1 - self.real_scan_factor))
        scan_border_right_bottom = round(self.plane_size/2 * (1 + self.real_scan_factor))
        cv2.rectangle(self.tip_path_img, (scan_border_left_top, scan_border_left_top), (scan_border_right_bottom, scan_border_right_bottom), scan_border_color, 10)

        # while True:
        while not self.tippath_stop_event.is_set():
            time.sleep(1)
            if not self.tippathvisualQueue.empty():
                circle = self.tippathvisualQueue.get()

                if len(circle) <= 2:                                                                        # data is a point which is the point of ready to scan 
                    cv2.circle(self.tip_path_img, (round(circle[0]), round(circle[1])), round(self.scan_square_edge/5) , (255, 0, 0), -1)
                elif len(circle) >= 4:                                                                      # data is a circle which is the circle of already scanned
                    left_top, right_bottom = center_to_square((circle[0], circle[1]), self.scan_square_edge)
                    t = circle[4]   #the coverge fiactor of the image
                    cover_color = interpolate_colors(color_min, color_max, t)                    
                    
                    if self.skip_flag == 1:                                                     # if the scan is skiped, the color of the square is set on
                        cover_color = sample_bad_color                    
                    if circle[3] == 1:
                        square_color = square_good_color    # good image color
                    elif circle[3] == 0:
                        square_color = square_bad_color     # bad image color
                    
                    if len(self.circle_list) == 1:
                        
                        # cv2.rectangle(self.tip_path_img, left_top, right_bottom, square_color, -1)
                        cv2.rectangle(self.tip_path_img, left_top, right_bottom, cover_color, -1)           # the box will be colored 
                        cv2.rectangle(self.tip_path_img, left_top, right_bottom, square_color, 3)
                        self.visual_circle_buffer_list.append(circle)                                            # add the first circle to the self.visual_circle_buffer_list

                    elif len(self.circle_list) > 1:
                        (Xn_1,Yn_1) = (self.visual_circle_buffer_list[-1][0], self.visual_circle_buffer_list[-1][1])
                        # cv2.rectangle(self.tip_path_img, left_top, right_bottom, square_color, -1)                        # draw the square
                        

                        cv2.rectangle(self.tip_path_img, left_top, right_bottom, cover_color, -1)           # the box will be colored 
                        cv2.rectangle(self.tip_path_img, left_top, right_bottom, square_color, 3)           # the edge of the box
                        cv2.line(self.tip_path_img, (round(Xn_1), round(Yn_1)), (round(circle[0]), round(circle[1])), line_color, 4)                                # ues the last circle center to draw the line to show the tip path
                        self.visual_circle_buffer_list.append(circle)                                            # add the new circle to the self.visual_circle_buffer_list
                        # delete the first circle in the self.visual_circle_buffer_list
                        if len(self.visual_circle_buffer_list) > 2:
                            self.visual_circle_buffer_list.pop(0)

                    else:
                        raise ValueError('the length of the circle list is not right')

                    cv2.imwrite(self.log_path + '/tip_path' +'.jpg', self.tip_path_img)
                    # save circle_list as a npy file
                    # np.save(self.log_path + '/circle_list.npy', self.circle_list)
                    # save coodinate_list as a npy file
                    # np.save(self.log_path + '/nanocoodinate_list.npy', self.nanocoodinate_list)
                elif circle == 'end':
                    # save the tip path image
                    cv2.imwrite(self.log_path + '/tip_path' +'.jpg', self.tip_path_img)
                    # np.save(self.log_path + '/circle_list.npy', self.circle_list)
                    # np.save(self.log_path + '/nanocoodinate_list.npy', self.nanocoodinate_list)
                    print('The tip path image is saved as ' + self.log_path + '/tip_path' +'.jpg')
                    break

                else:
                    raise ValueError('the circle data is not a point or a circle')

                self.tipdataQueue.put(self.tip_path_img)
                # cv2.imshow('Tip Path', self.tip_path_img)



            cv2.waitKey(100)

    # def a function to move the tip to the next point, and append all data to the circle_list and nanocoodinate_list
    def move_to_next_point(self):
        if len(self.nanocoodinate_list) == 0:
            self.nanocoodinate = pix_to_nanocoordinate(self.inter_closest,plane_size = self.plane_size) # init the first point of the nanocoodinate_list
            self.nanocoodinate_list.append(self.nanocoodinate)
        else:
            if self.tip_move_mode == 0:
                self.inter_closest = Next_inter(self.circle_list)                                       # calculate the next inter_closest
            self.nanocoodinate = pix_to_nanocoordinate(self.inter_closest,plane_size = self.plane_size)
            self.nanocoodinate_list.append(self.nanocoodinate)

        self.tippathvisualQueue.put(self.inter_closest)                                             # put the inter_closest to the tippathvisualQueue
        print('The scan center is ' + str(self.inter_closest))

        print('move to the area...')
        self.ScanFrameSet(self.nanocoodinate[0],self.nanocoodinate[1] + 0.5*self.Scan_edge_SI, self.Scan_edge, 1e-15, angle = 0)
        np.save(self.log_path + '/nanocoodinate_list.npy', self.nanocoodinate_list)
    
    
    # def a function to segment the image
    def image_segmention(self, Scan_image):
        self.segmented_image_path = self.image_data_save_path + '/segmented_image/'
        if not os.path.exists(self.segmented_image_path):                            # create the segmented_image folder if not exist
            os.makedirs(self.segmented_image_path)
        # self.nemo_pointnemo_point in here the dift persentage of the matrix image 
        nemo_point, self.coverage = Segmented_image(Scan_image, self.segmented_image_path, model_path = self.segment_model_path)    
        nemo_point_pix = (self.inter_closest[0]+nemo_point[0]*self.scan_square_edge, self.inter_closest[1]+nemo_point[1]*self.scan_square_edge)
        self.nemo_nanocoodinate = pix_to_nanocoordinate(nemo_point_pix, plane_size = self.plane_size)   

        print('The nemo point is ' + str(nemo_point_pix))
        print('The coverage is ' + str(round(self.coverage,2)))
    
    
    # def a function to predict the scan qulity
    def image_recognition(self):
        # judge the gap between the max and min of the image
        # if the gap is bigger than the threshold, the scan is skiped
        data_linear = linear_whole(self.Scan_data['Scan_data_for']['data'])
        print('The gap between the max and min of the image is ' + str(linescan_max_min_check(data_linear)))
        if linescan_max_min_check(linear_whole(data_linear)) >= self.convert(self.scan_max_min_threshold):
            self.skip_flag = 1
            print('The scan is skiped')
        
        # use CNN to predict the image quality
        probability = predict_image_quality(self.image_for, self.quality_model_path)
        print('The probability of the good image is ' + str(round(probability,2)))
        if probability > self.scan_qulity_threshold and self.skip_flag == 0 :  # 0.5 is the self.scan_qulity_threshold of the probability
            scan_qulity = 1 # good image
            self.good_image_count +=1
        else:
            scan_qulity = 0 # bad image

        # calculate the R depend on the scan_qulity
        if len(self.circle_list) == 0:                                                      # if the circle_list is empty, initialize the R 
            self.R = self.R_init
        else:
            self.R = increase_radius(scan_qulity, self.circle_list[-1][2], self.R_init, self.R_max, self.R_step)     # increase the R

        self.image_segmention(self.image_for)
        self.circle_list.append([self.inter_closest[0], self.inter_closest[1], self.R, scan_qulity])
        # save the circle_list as a npy file
        self.circle_list_save.append([self.inter_closest[0], self.inter_closest[1], self.R, scan_qulity, self.coverage])
        np.save(self.log_path + '/circle_list.npy', self.circle_list_save)
        self.tippathvisualQueue.put([self.inter_closest[0], self.inter_closest[1], self.R, scan_qulity, self.coverage])        
        #save the scan image via the scan_qulity

        # create the good_scan and bad_scan folder in self.image_data_save_path if not exist
        if not os.path.exists(self.image_data_save_path + '/good_scan'):
            os.makedirs(self.image_data_save_path + '/good_scan')
        if not os.path.exists(self.image_data_save_path + '/bad_scan'):
            os.makedirs(self.image_data_save_path + '/bad_scan')
        self.image_save_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
        if scan_qulity == 1:
            cv2.imwrite(self.image_data_save_path + '/good_scan/' + 'Scan_data_for'+ self.image_save_time +'.png', self.image_for)  #save the image in the good_scan folder
            cv2.imwrite(self.image_data_save_path + '/good_scan/' + 'Scan_data_back'+ self.image_save_time +'.png', self.image_back)  #save the image in the good_scan folder

        else:
            cv2.imwrite(self.image_data_save_path + '/bad_scan/' + 'Scan_data_for'+ self.image_save_time +'.png', self.image_for)   #save the image in the bad_scan folder
            cv2.imwrite(self.image_data_save_path + '/bad_scan/' + 'Scan_data_back'+ self.image_save_time +'.png', self.image_back)   #save the image in the bad_scan folder


        
        # return scan_qulity
        return scan_qulity

    
    #def a function to select the action for nanonis
    def select_nanonis_action(self, state, select_mode = 'agent'):
        # if the type of the state
        if select_mode == 'random':
            # select 0, 1, 2, 3, 4, 5 randomly
            nanonis_action_number = torch.tensor([[random.randint(0, 5)]], device=self.device)  # for instance: nanonis_action_number = tensor([[0]], device='cuda:0')
        elif select_mode == 'agent':
            nanonis_action_number = self.DQN_Agent.select_action(state)                   # according to the DQN, nanonis_action_number has 6 values: 0, 1, 2, 3, 4, 5 (in tensor format)
            # nanonis_action_number = torch.tensor([[4]], device=self.device)
        
        
        if nanonis_action_number.item() == 0:
            self.BiasPulse(-2, width=0.05)
        elif nanonis_action_number.item() == 1:
            self.BiasPulse(-4, width=0.05)
        elif nanonis_action_number.item() == 2:
            self.BiasPulse(-6, width=0.05)
        elif nanonis_action_number.item() >= 3:
            # move the tip to the point that should be tip shapered      
            # secondly, move the tip to the nemo_nanocoodinate
            self.TipXYSet(self.nemo_nanocoodinate[0], self.nemo_nanocoodinate[1])
            time.sleep(1)
            # thirdly, shaper the tip
            if nanonis_action_number.item() == 3:
                self.TipShaper(TipLift= '-1.5n')
            elif nanonis_action_number.item() == 4:
                self.TipShaper(TipLift= '-2.5n')
            elif nanonis_action_number.item() == 5:
                self.TipShaper(TipLift= '-6n')
        print('ACTION:', nanonis_action_number.item())
        return nanonis_action_number                                     # for instance: nanonis_action_number = tensor([[0]], device='cuda:0')

    # def a function to crate the reward
    def reward_function(self, action_evaluation):
        if action_evaluation:
            reward = 10
        else:
            reward = -1

        return torch.tensor([reward], device=self.device)
    
    # def a function to creata the trajectory
    def create_trajectory(self,scan_qulity):
        time.sleep(0.5)
        if self.skip_flag == 1:# if the scan is skiped, return
            self.skip_flag = 0  # reset the skip flag
            return
        if scan_qulity == 0: # meet the bad image
            # if the memory_path + '/' + episode_start_time is not exist, create the folder
            # episode_start_time = time.strftime('%H-%M-%S',time.localtime(time.time()))                                  # record the start time of the trajectory
            
            self.nanonis_action_number = self.select_nanonis_action(self.image_for, select_mode = 'agent')                              # implement the action which have the hightest Q value accordding to the image and policy_net
            
            if len(self.trajectory_state_list) == 0:
                self.episode_start_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))                                  # record the start time of the trajectory
                self.trajectory_state_list.append((self.image_for,self.image_back))
                self.trajectory_action_list.append(self.nanonis_action_number.item())

            elif len(self.trajectory_state_list) > 0:
                print('the tip fix is not successful...')
                action_evaluation = 0
                
                self.trajectory_state_list.append((self.image_for,self.image_back))                       # second bad image, state_list aways have one more element than next_state_list.
                self.trajectory_next_state_list.append((self.image_for,self.image_back))                  # second bad image
                self.trajectory_action_list.append(self.nanonis_action_number.item())                                          # the action which have the hightest Q value
                self.trajectory_reward_list.append(self.reward_function(action_evaluation).item())                             # the reward of the action

                print('reward:', self.trajectory_reward_list[-1])
                # set trajectory_for to a dictionary
                trajectory_for ={ 'state': self.trajectory_state_list[-2][0].tolist(), 
                                 'action': self.trajectory_action_list[-1], 
                                'reward': self.trajectory_reward_list[-1], 
                                'next_state': self.trajectory_next_state_list[-1][0].tolist()}
                
                trajectory_back = {'state': self.trajectory_state_list[-2][1].tolist(),
                                   'action': self.trajectory_action_list[-1], 
                                'reward': self.trajectory_reward_list[-1], 
                                'next_state': self.trajectory_next_state_list[-1][0].tolist()}
                
                trajectory_start_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
                trajectory_for_name = trajectory_start_time +'_trajectory_for'+ '.json'
                trajectory_back_name = trajectory_start_time +'_trajectory_back'+ '.json'
                # save the trajectory to the replay memory
                
                trajectory_path = self.memory_path + '/' + self.episode_start_time + 'episode'
                self.memory.save(trajectory_path, trajectory_for_name, trajectory_for)
                self.memory.save(trajectory_path, trajectory_back_name, trajectory_back)


        elif scan_qulity == 1:  # meet the good image
            if len(self.trajectory_state_list) > 0:   # first image after the bad image
                print('the tip fix is successful !!!')
                action_evaluation = 1

                self.trajectory_next_state_list.append((self.image_for,self.image_back))                 # add the last good image to the next_state_list but not the state_list to complete the episode
                self.trajectory_reward_list.append(self.reward_function(action_evaluation).item())

                print('reward:', self.trajectory_reward_list[-1])
                trajectory_for = {'state': self.trajectory_state_list[-1][0].tolist(),
                                  'action': self.trajectory_action_list[-1], 
                                'reward': self.trajectory_reward_list[-1], 
                                'next_state': self.trajectory_next_state_list[-1][0].tolist()}
                trajectory_back = {'state': self.trajectory_state_list[-1][1].tolist(),
                                    'action': self.trajectory_action_list[-1], 
                                    'reward': self.trajectory_reward_list[-1], 
                                    'next_state': self.trajectory_next_state_list[-1][1].tolist()}
                
                trajectory_start_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
                trajectory_for_name = trajectory_start_time +'_trajectory_for'+ '.json'
                trajectory_back_name = trajectory_start_time +'_trajectory_back'+ '.json'
                
                trajectory_path = self.memory_path + '/' + self.episode_start_time + 'episode'
                self.memory.save(trajectory_path, trajectory_for_name, trajectory_for)
                self.memory.save(trajectory_path, trajectory_back_name, trajectory_back)

                # initialize the trajectory list
                self.trajectory_state_list.clear()                # the list to save the one step in memo
                self.trajectory_action_list.clear()
                self.trajectory_reward_list.clear()
                self.trajectory_next_state_list.clear()
    # def a function to pulse
    def bias_pulse_set(self, width, bias, wait=True):
        while True:
            try:
                # 检查是否两个队列都为空
                if width is None and bias is None:
                    break
                # 如果队列有值，发送脉冲命令
                if width is not None or bias is not None:
                    self.send('Bias.Pulse', 'uint32', int(wait), 'float32', float(
                        width), 'float32', float(bias), 'uint16', 0, 'uint16', 0)
                    print(f"Pulse with voltage: {bias}V\n"
                          f"Pulse with width:{width}m")
                    bias = None
                    width = None

            except Exception as e:
                # 处理异常，例如队列操作失败或send方法失败
                print(f"An error occurred: {e}")
                # 可以选择在这里退出循环，或者设置标志来通知其他部分的代码
                break
    # def a function to record the statues of scan
    def controls_set(self, controls):
        while True:
            try:
                # 检查是否两个队列都为空
                if controls is None:
                    break
                # 如果队列有值，发送脉冲命令
                if controls == 1:
                    self.send('Scan.Action', 'uint16', controls, 'uint32', 0)
                    print(f"programming stop")
                    controls = None
                elif controls == 2:
                    self.send('Scan.Action', 'uint16', controls, 'uint32', 0)
                    print(f"programming pause")
                    controls = None
                elif controls == 3:
                    self.send('Scan.Action', 'uint16', controls, 'uint32', 0)
                    print(f"programming resume")
                    controls = None

            except Exception as e:
                # 处理异常，例如队列操作失败或send方法失败
                print(f"An error occurred: {e}")
                # 可以选择在这里退出循环，或者设置标志来通知其他部分的代码
                break
    # def a function to set the scan speed
    def speed_set(self, time_per_frame):
        Keep_parameter_constant = 2
        Speed_ratio = 1
        height = self.ScanBufferGet()['Lines']
        time_per_line = time_per_frame/height/2
        Forward_linear_speed = 1
        Backward_linear_speed = 1
        while True:
            try:
                # 检查是否两个队列都为空
                if time_per_frame is None:
                    break
                # 如果队列有值，发送脉冲命令
                if time_per_frame is not None:
                    self.send('Scan.SpeedSet', 'float32', Forward_linear_speed, 'float32', Backward_linear_speed,
                              'float32', time_per_line, 'float32', time_per_line, 'uint16',
                              Keep_parameter_constant, 'float32', Speed_ratio)
                    time_per_frame = None
            except Exception as e:
                # 处理异常，例如队列操作失败或send方法失败
                print(f"An error occurred: {e}")
                break
    def z_position_adjust(self):
        z_position = self.ZPosGet()
        if z_position < -2.3E-7:
            self.MotorMoveSet(4, 1)
            z_position_change = self.ZPosGet()
            q = 2.3E-7/(z_position_change - z_position)-1
            self.MotorMoveSet(4, q)
        elif z_position > 2.3E-7:
            self.MotorMoveSet(5, 1)
            z_position_change = self.ZPosGet()
            q = 2.3E-7/(z_position - z_position_change)-1
            self.MotorMoveSet(5, q)

    def TipShaper_set(self, TipLift, Switch_Off_Delay=0.05, Lift_Time_1=0.1, Bias_Setting_Time=0.06,
                  Lift_Time_2=0.06,  Lifting_Bias=-1.0, Lifting_Height='2n',
                  End_Wait_Time=0.01, stepstotarget=10):

        TipLift = self.try_convert(TipLift)
        Lifting_Height = self.try_convert(Lifting_Height)

        Origin_ScanStatus = self.ScanStatusGet()  # check the scan status
        if Origin_ScanStatus == 1:
            self.ScanPause()  # Pause the scan (not stop), if the scan is running

        ZCtrl_status = self.ZCtrlOnOffGet()  # get current Z control status

        Z_current = self.TipZGet()  # get current Z position

        Bias_current = self.BiasGet()  # get current Bias

        self.ZCtrlOnOffSet("off")  # close Z control

        time.sleep(Switch_Off_Delay)  # wait for Switch_Off_Delay

        # use for loop to simulate the uniform motion of the tip
        for i in range(stepstotarget):
            self.TipZSet(Z_current + (i + 1) * (TipLift / stepstotarget))  # set the new Z position

            time.sleep(Lift_Time_1 / stepstotarget)  # wait for Lift_Time_1/stepstotarget time

        time.sleep(1)

        self.BiasSet(Lifting_Bias)  # set the new Bias

        time.sleep(Bias_Setting_Time)  # wait for Bias_Setting_Time

        # use for loop to simulate the uniform motion of the tip
        for i in range(stepstotarget):
            self.TipZSet(Z_current + TipLift + (i + 1) * (
                        Lifting_Height - TipLift) / stepstotarget)  # set the new Z position → Lifting_Height

            time.sleep(Lift_Time_2 / stepstotarget)  # wait for Lift_Time_2/stepstotarget time

        time.sleep(1)
        # self.Tipshaper_signal = 1 #set a signal while tipshaper
        self.BiasSet(Bias_current)  # set the new Bias → Bias_current

        time.sleep(End_Wait_Time)  # wait for End_Wait_Time

        # reopen Z control or not?
        if self.iftipshaper == 1:
            x = self.ScanFrameGet()['center_x'] + 5E-8
            y = self.ScanFrameGet()['center_y'] + 5E-8
            self.ScanFrameSet(x, y, 5E-8, 5E-8, 0)
            self.iftipshaper = 0
        if ZCtrl_status == 1:
            self.ZCtrlOnOffSet("on")
        else:
            self.ZCtrlOnOffSet("off")

        if Origin_ScanStatus == 1:
            self.ScanResume()  # Resume the scan (not restart), if the scan was running

    # def a function to record pause_time while scan running
    def calculate_total_scan_time(self, expected_total_time):
        self.scan_start_time = None
        self.per_scan_time = 0
        self.total_scan_time = 0
        while True:
            status = self.ScanStatusGet()
            if self.Tipshaper_signal == 1:
                self.Tipshaper_signal = 0
                break
            elif self.frame_move_signal == 1:
                self.frame_move_signal = 0
                break
            elif expected_total_time != self.current_time_per_frame():
                break
            elif status == 1:
                if self.scan_start_time is None:
                    self.scan_start_time = time.time()  # 记录扫描开始时间
            elif status == 0:
                if self.scan_start_time is not None:
                    # 计算当前扫描周期的时间
                    self.per_scan_time = time.time() - self.scan_start_time
                    # 累加到总扫描时间
                    self.total_scan_time += self.per_scan_time
                    # 重置扫描开始时间和当前扫描时间
                    self.scan_start_time = None
                    self.per_scan_time = 0

                    # 检查是否达到预期的总扫描时间
                    if self.total_scan_time+3 >= expected_total_time:
                        break
                if self.scan_start_time is None:
                    pass
            # 每秒检查一次状态
            time.sleep(2)
    def current_time_per_frame(self):
        true_t = (self.ScanSpeedGet()['Forward time per line'] + self.ScanSpeedGet()[
            'Backward time per line']) * \
            self.ScanBufferGet()['Lines']
        return true_t

    def Gainset(self, proportional, integral):
        while True:
            try:
                if proportional is None and integral is None:
                    break
                if proportional is not None or integral is not None:
                    self.send('ZCtrl.GainSet', 'float32', float(proportional), 'float32', float(
                        proportional/integral), 'float32', float(integral))
                    print(f"P-gain set: {proportional}pm\n"
                          f"I-gain set:{integral}pm/s")
                    proportional = None
                    integral = None

            except Exception as e:
                print(f"An error occurred: {e}")
                break
    def get_current_value(self):
        q = self.SignalsValsGet(2, (0, 30))
        return q['value']
    # function to get current and z
    def plot_realtime_current(self, fig_max_len=500, update_interval=10):
        xdata, ydata = deque(maxlen=fig_max_len), deque(maxlen=fig_max_len)
        fig, ax = plt.subplots(figsize=(10,6))
        ln, = ax.plot(xdata, ydata, animated=True)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
        ax.set_ylabel('Current (pA)')
        ax.set_title('Real-time Current Measurement')

        # 更新图表
        def update(frame):
            current_value = abs(self.get_current_value()[0])*1E+12
            xdata.append(frame)
            ydata.append(current_value)
            ax.set_xlim(max(0, frame - fig_max_len), frame+1)
            ax.set_ylim(min(ydata), max(ydata))
            ln.set_data(xdata, ydata)
            return ln,

        ani = FuncAnimation(fig, update, interval=update_interval, blit=True)

        plt.show()

    def plot_realtime_z(self, fig_max_len=500):
        xdata, ydata = deque(maxlen=fig_max_len), deque(maxlen=fig_max_len)
        fig, ax = plt.subplots(figsize=(10,6))
        ln, = ax.plot(xdata, ydata, animated=True)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
        ax.set_ylabel('Z(m)')
        ax.set_title('Real-time Z Measurement')

        def update(frame):
            current_value = abs(self.get_current_value()[1]) * 1E+12
            xdata.append(frame)
            ydata.append(current_value)
            ax.set_xlim(max(0, frame - fig_max_len), frame + 1)
            ax.set_ylim(min(ydata), max(ydata))
            ln.set_data(xdata, ydata)
            return ln,

        ani = FuncAnimation(fig, update, interval=10, blit=True)
        plt.show()

    def if_fix_tip(self):
        self.iftipshaper = 1
    def bias_pulse(self):
        self.send('Bias.Pulse', 'uint32', int(True), 'float32', float(5E-2), 'float32', float(-2), 'uint16', 0,
                          'uint16', 0)
    def move_to_next_area(self):
        self.ScanStop()
        self.ScanFrameSet(0, 0, 5E-8, 5E-8)
        self.ZCtrlWithdraw(1)
        time.sleep(0.5)
        self.MotorMoveSet('Z-', 100)
        time.sleep(0.5)
        self.MotorMoveSet('X-', 10)
        time.sleep(0.5)
        self.MotorMoveSet('Y-', 10)
        # self.nanonis.ZCtrlOnOffSet(1)
        self.AutoApproachOpen()
        time.sleep(0.5)
        self.AutoApproachSet()
        print("area move succeed, wait for auto approach")

if __name__ == '__main__':
    nanonis = Mustard_AI_Nanonis()

    nanonis.tip_init(mode = 'new') # deflaut mode is 'new' mode = 'new' : the tip is initialized to the center and create a new log folder, mode = 'latest' : load the latest checkpoint
    
    nanonis.DQN_init(mode = 'new') # deflaut mode is 'latest' mode = 'new' : create a new model, mode = 'latest' : load the latest checkpoint    

    nanonis.monitor_thread_activate()                                                # activate the monitor thread


    while tip_in_boundary(nanonis.inter_closest, nanonis.plane_size, nanonis.real_scan_factor):
        
        nanonis.move_to_next_point()                                                    # move the scan area to the next point

        nanonis.AdjustTip_flag = nanonis.AdjustTipToPiezoCenter()                       # check & adjust the tip to the center of the piezo

        nanonis.line_scan_thread_activate()                                             # activate the line scan, producer-consumer architecture, pre-check the tip and sample


        nanonis.batch_scan_producer(nanonis.nanocoodinate, nanonis.Scan_edge, nanonis.scan_square_Buffer_pix, 0)    # Scan the area

    
        scan_qulity = nanonis.image_recognition()                                       # assement the scan qulity 

        # nanonis.image_segmention(nanonis.image_for)                                   # segment the image ready to tip shaper        

        # scan_qulity = 0
        nanonis.create_trajectory(scan_qulity)                                          # create the trajectory

        if scan_qulity == 0:
            nanonis.DQN_upgrate()                                                       # optimize the model and update the target network

        nanonis.save_checkpoint()                                                       # save the checkpoint


        time.sleep(3)









