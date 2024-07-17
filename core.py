# core.py by CoccaGuo at 2022/05/16 16:45
##############################################
# core.py renew by Zhiwen Zhu at 2023/07/21  Email address: zhiwenzhu@shu.edu.cn
##############################################
import logging
import time
from abc import ABCMeta, abstractmethod
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

from interface import *


class NanonisController(nanonis_programming_interface):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        setupLog()

    def try_convert(self, value):
        if type(value) is str:
            return self.convert(value)
        else:
            return float(value)

    '''
    Motor.StartMove
    Moves the coarse positioning device (motor, piezo actuator…).
    Arguments:
    - Direction (unsigned int32) selects in which direction to move. Note that depending on your motor
    controller and setup only the Z axis or even only Z- may work.
    Valid values are 0=X+, 1=X-, 2=Y+, 3=Y-, 4=Z+, 5=Z-
    - Number of steps (unsigned int16) defines the number of steps to move in the specified direction
    - Group (unsigned int32) is the selection of the groups defined in the motor control module. If the motor
    doesn't support the selection of groups, set it to 0.
    Valid values are 0=Group 1, 1=Group 2, 2=Group 3, 3=Group 4, 4=Group 5, 5=Group 6
    - Wait until finished (unsigned int32) defines if this function only returns (1=True) when the motor reaches
    its destination or the movement stops
    Return arguments (if Send response back flag is set to True when sending request message):
    - Error described in the Response message>Body section
    '''

    def MotorMoveSet(self, direction, steps, wait=True):
        direct_map = {'X+': 0, 'X-': 1, 'Y+': 2, 'Y-': 3, 'Z+': 4, 'Z-': 5}
        try:
            direction = direct_map[direction.upper()]
        except KeyError:
            raise nanonisException(
                'Invalid direction. Please use X+, X-, Y+, Y-, Z+, Z- expressions.')
        # if direction == 5:
        #     raise nanonisException('Moving Z- is not Safe.')
        self.send('Motor.StartMove', 'uint32', direction, 'uint16',
                  steps, 'uint32', 0, 'uint32', int(wait))

    def AutoApproachOpen(self):
        self.send('AutoApproach.Open')

    def AutoApproachSet(self, on=True):
        self.send('AutoApproach.OnOffSet', 'uint16', int(on))

    def AutoApproachGet(self):
        parsedResponse = self.parse_response(
            self.send('AutoApproach.OnOffGet'), 'uint16')
        return parsedResponse['0']

    def ZGainGet(self):
        parsedResponse = self.parse_response(
            self.send('ZCtrl.GainGet'), 'float32', 'float32', 'float32')
        return {
            'P': parsedResponse['0'],
            'T': parsedResponse['1'],
            'I': parsedResponse['2']
        }

    def ZGainSet(self, P, T, I):
        if type(P) is str:
            P_val = self.convert(P)
        else:
            P_val = float(P)
        if type(T) is str:
            T_val = self.convert(T)
        else:
            T_val = float(T)
        if type(I) is str:
            I_val = self.convert(I)
        else:
            I_val = float(I)
        self.send('ZCtrl.GainSet', 'float32', P_val,
                  'float32', T_val, 'float32', I_val)

    def ZGainPSet(self, P):
        if type(P) is str:
            P_val = self.convert(P)
        else:
            P_val = float(P)
        present = self.ZGainGet()
        self.ZGainSet(P_val, present['T'], present['I'])

    def ZGainTSet(self, T):
        if type(T) is str:
            T_val = self.convert(T)
        else:
            T_val = float(T)
        present = self.ZGainGet()
        self.ZGainSet(present['P'], T_val, present['I'])

    def PLLOutputSet(self, on=True):
        self.send('PLL.OutOnOffSet', 'int', 1, 'uint32', int(on))

    def PLLFreqShiftGet(self):
        parsedResponse = self.parse_response(
            self.send('PLL.FreqShiftGet', 'int', 1), 'float32')
        return parsedResponse['0']

    def PLLAmpCtrlSet(self, on=True):
        self.send('PLL.AmpCtrlOnOffSet', 'int', 1, 'uint32', int(on))

    def PLLPhasCtrlSet(self, on=True):
        self.send('PLL.PhasCtrlOnOffSet', 'int', 1, 'uint32', int(on))

    def ZCtrlOnOffGet(self):
        parsedResponse = self.parse_response(
            self.send('ZCtrl.OnOffGet'), 'uint32')
        return parsedResponse['0']

    def ZCtrlOnSet(self, on=True):
        self.send('ZCtrl.OnOffSet', 'uint32', int(on))
    
    def ZCtrlOff(self, off=False):
        self.send('ZCtrl.OnOffSet', 'uint32', int(off))

    def ZLimitsGet(self):
        parsedResponse = self.parse_response(
            self.send('ZCtrl.LimitsGet'), 'float32', 'float32')
        return {
            'high': parsedResponse['0'],
            'low': parsedResponse['1']
        }

    def ZLimitCheck(self):
        '''
        intermediate api
        '''
        z1 = self.TipZGet()
        z_limits = self.ZLimitsGet()
        z_max = z_limits['high']
        z_min = z_limits['low']
        if abs(z_max-z1)/z_max < 0.01:  # if the tip is at the piezo top limit
            self.Withdraw()
            logging.info('Piezo reached the z-high-limit, withdraw')
            raise nanonisException(
                'Piezo reached the z-high-limit', ExceptionType.Z_HIGH_LIMIT_REACHED)
        if abs(z1-z_min/z_min) < 0.01:
            logging.info('Piezo reached the z-low-limit, need auto approach')
            raise nanonisException(
                'Piezo reached the z-low-limit', ExceptionType.Z_LOW_LIMIT_REACHED)

    def ZLimitCheckWithAction(self):
        try:
            self.ZLimitCheck()
        except nanonisException as e:
            if e.code == ExceptionType.Z_HIGH_LIMIT_REACHED:
                self.Withdraw()
                self.MotorMoveSet('Z+', 1)
                self.AutoApproachSet()
            elif e.code == ExceptionType.Z_LOW_LIMIT_REACHED:
                self.Withdraw()
                self.AutoApproachSet()
            else:
                raise e
        finally:
            self.WaitForZCtrlWork()
    '''
    ZCtrl.Home
    Moves the tip to its home position.
    This function moves the tip to the home position defined by the Home Absolute (m)/ Home Relative (m) value.
    (Absolute and relative can be switched in the controller configuration panel in the software).
    Arguments: None
    Return arguments (if Send response back flag is set to True when sending request message):
    - Error described in the Response message>Body section
    '''
    def ZCtrlHome(self):
        self.send('ZCtrl.Home')

    '''
    Bias.Pulse
    Generates one bias pulse.
    Arguments:
        - Wait until done (unsigned int32), if True, this function will wait until the pulse has finished. 1=True and
        0=False
        - Bias pulse width (s) (float32) is the pulse duration in seconds
        - Bias value (V) (float32) is the bias value applied during the pulse
        - Z-Controller on hold (unsigned int16) sets whether the controller is set to hold (deactivated) during the
        pulse. Possible values are: 0=no change, 1=hold, 2=don’t hold
        - Pulse absolute/relative (unsigned int16) sets whether the bias value argument is an absolute value or
        relative to the current bias voltage. Possible values are: 0=no change, 1=relative, 2=absolute
    Return arguments (if Send response back flag is set to True when sending request message):
        - Error described in the Response message>Body section
    '''
    def BiasPulse(self, bais, width=0.1, wait=True):
        self.send('Bias.Pulse', 'uint32', int(wait), 'float32', float(
            width), 'float32', float(bais), 'uint16', 0, 'uint16', 0)

    def PiezoRangeGet(self):
        parsedResponse = self.parse_response(
            self.send('Piezo.RangeGet'), 'float32', 'float32', 'float32')
        return {
            'x': parsedResponse['0'],
            'y': parsedResponse['1'],
            'z': parsedResponse['2'],
        }

    def SignalsNamesGet(self):
        response = self.send('Signals.NamesGet')
        # print(response)
        cursor = 4
        name_number = from_binary('int', response['body'][cursor: cursor+4])
        cursor += 4
        names = []
        for i in range(name_number):
            name_size = from_binary('int', response['body'][cursor: cursor+4])
            cursor += 4
            name = from_binary(
                'string', response['body'][cursor: cursor+name_size])
            if self.channel_name_filter(name):
                names.append(name)
            cursor += name_size
        return names

    def SignalIndexGet(self, name):
        if not hasattr(self, 'signal_chart'):
            self.signal_chart = self.SignalsNamesGet()
        if name not in self.signal_chart:
            raise nanonisException('Invalid signal name.')
        return self.signal_chart.index(name)


    def SignalInSlotSet(self, Slot, RT_signal_index):           
        #Slot (int) is the index of the slot in the Signals Manager where one of the 128 RT signals is assigned, so that index could be any value from 0 to 23
        #RT signal index (int) is the index of the RT signal to assign to the selected slot, so that index could be any value from 0 to 127
        self.send('Signals.InSlotSet', 'int', Slot, 'int', RT_signal_index)

    def SignalInSlotGet(self):                                                     #need to be rewritten
        response = self.send('Signals.InSlotsGet')
        print(response)
        cursor = 0
        Signals_names_size = from_binary('int', response['body'][cursor: cursor+4])
        cursor += 4
        Number_of_Signals = from_binary('int', response['body'][cursor: cursor+4])
        cursor += 4
        Signals_names = []
        for i in range(Number_of_Signals):
            single_names_size = from_binary('int', response['body'][cursor: cursor+4])
            cursor += 4
            slot = from_binary('string', response['body'][cursor: cursor + single_names_size])
            cursor += single_names_size
            Signals_names.append(slot)
            # slot_size = len(slot)
            # cursor += len(slot)
        Signals_indexes_size = from_binary('int', response['body'][cursor: cursor+4])
        cursor += 4
        Signals_indexes_number = Signals_indexes_size/4
        Signals_indexes = {}
        Signals_Slot = {}
        for i in range(Signals_indexes_size):
            Signals_Slot[i] = Signals_names[i]
            Signals_indexes[(from_binary('int', response['body'][cursor: cursor+4]))] = Signals_names[i]
            cursor += 4

        return  {'Signals_Slot': Signals_Slot, 'Signals_indexes': Signals_indexes}

        # return slots


    def ScanStart(self, direction='down'):
        if direction == 'down':
            d = 0
        elif direction == 'up':
            d = 1
        else:
            raise nanonisException('Invalid direction. Please use down or up.')
        self.send('Scan.Action', 'uint16', 0, 'uint32', d)

    def ScanStop(self):
        self.send('Scan.Action', 'uint16', 1, 'uint32', 0)

    def ScanPause(self):
        self.send('Scan.Action', 'uint16', 2, 'uint32', 0)

    def ScanResume(self):
        self.send('Scan.Action', 'uint16', 3, 'uint32', 0)

    def ScanStatusGet(self):
        return self.parse_response(self.send('Scan.StatusGet'), 'uint32')['0']

    
    def ScanBufferGet(self):                                                            #need to be rewritten
        response = self.send('Scan.BufferGet')
        cursor = 0
        Number_of_channels = from_binary('int', response['body'][cursor: cursor+4])        
        cursor += 4
        Channel_indexes = []
        for i in range(Number_of_channels):
            Channel_indexes.append(from_binary('int', response['body'][cursor: cursor+4]))
            cursor += 4
        Pixels = from_binary('int', response['body'][cursor: cursor+4])
        cursor += 4
        Lines = from_binary('int', response['body'][cursor: cursor+4])

        return {'Pixels':Pixels, 'Lines':Lines, 'Channel_indexes':Channel_indexes}
    
    def ScanBufferSet(self, Pixels, Lines, Channel_indexes):       #need to be rewritten
        if len(Channel_indexes) == 2:
            self.send('Scan.BufferSet', 'int', len(Channel_indexes), 'int', Channel_indexes[0], 'int', Channel_indexes[1], 'int', Pixels, 'int', Lines)
        elif len(Channel_indexes) == 3:
            self.send('Scan.BufferSet', 'int', len(Channel_indexes), 'int', Channel_indexes[0], 'int', Channel_indexes[1], 'int', Channel_indexes[2], 'int', Pixels, 'int', Lines)
    
    def ScanFrameSet(self, center_x, center_y, width, height, angle=0):
        if type(center_x) is str:
            center_x = self.convert(center_x)
        else:
            center_x = float(center_x)
        if type(center_y) is str:
            center_y = self.convert(center_y)
        else:
            center_y = float(center_y)
        if type(width) is str:
            width = self.convert(width)
        else:
            width = float(width)
        if type(height) is str:
            height = self.convert(height)
        else:
            height = float(height)
        self.send('Scan.FrameSet', 'float32', center_x, 'float32',
                  center_y, 'float32', width, 'float32', height, 'float32', angle)
    
    '''
    Scan.SpeedSet
    Configures the scan speed parameters.
    Arguments:
    - Forward linear speed (m/s) (float32)
    - Backward linear speed (m/s) (float32)
    - Forward time per line (s) (float32)
    - Backward time per line (s) (float32)
    - Keep parameter constant (unsigned int16) defines which speed parameter to keep constant, where 0
    means no change, 1 keeps the linear speed constant, and 2 keeps the time per line constant
    - Speed ratio (float32) defines the backward tip speed related to the forward speed
    '''
    def ScanSpeedSet(self, Forward_linear_speed, Backward_linear_speed, Forward_time_per_line, Backward_time_per_line, Keep_parameter_constant = 0, Speed_ratio = 1):
        self.send('Scan.SpeedSet', 'float32', Forward_linear_speed, 'float32', Backward_linear_speed, 'float32', Forward_time_per_line, 'float32', Backward_time_per_line, 'uint16', Keep_parameter_constant, 'float32', Speed_ratio)

    '''
    Scan.SpeedGet
    Returns the scan speed parameters.
    Arguments: None
    Return arguments (if Send response back flag is set to True when sending request message):
    - Forward linear speed (m/s) (float32)
    - Backward linear speed (m/s) (float32)
    - Forward time per line (s) (float32)
    - Backward time per line (s) (float32)
    - Keep parameter constant (unsigned int16) defines which speed parameter to keep constant, where 0
    keeps the linear speed constant, and 1 keeps the time per line constant
    - Speed ratio (float32) is the backward tip speed related to the forward speed
    - Error described in the Response message>Body section
    '''
    def ScanSpeedGet(self):
        parsedResponse = self.parse_response(self.send('Scan.SpeedGet'), 'float32', 'float32', 'float32', 'float32', 'uint16', 'float32')
        return {
            'Forward linear speed': parsedResponse['0'],
            'Backward linear speed': parsedResponse['1'],
            'Forward time per line': parsedResponse['2'],
            'Backward time per line': parsedResponse['3'],
            'Keep parameter constant': parsedResponse['4'],
            'Speed ratio': parsedResponse['5']
        }
    
    
    def ScanFrameGet(self):
        parsedResponse = self.parse_response(
            self.send('Scan.FrameGet'), 'float32', 'float32', 'float32', 'float32', 'float32')
        return {
            'center_x': parsedResponse['0'],
            'center_y': parsedResponse['1'],
            'width': parsedResponse['2'],
            'height': parsedResponse['3'],
            'angle': parsedResponse['4']
        }
    
    ''' 
    - Continuous scan (unsigned int32) indicates whether the scan continues or stops when a frame has been completed. 0 means Off, and 1 is On
    - Bouncy scan (unsigned int32) indicates whether the scan direction changes when a frame has been completed. 0 means Off, and 1 is On
    - Autosave (unsigned int32) defines the save behavior when a frame has been completed. "All" saves all the future images. "Next" only saves the next frame. 0 is All, 1 is Next, and 2 means Off
    - Series name size (int) is the size in bytes of the Series name string
    - Series name (string) is base name used for the saved images
    - Comment size (int) is the size in bytes of the Comment string
    - Comment (string) is comment saved in the file
    - Error described in the Response message>Body section
    '''
    def ScanPropsGet(self):
        response = self.send('Scan.PropsGet')
        cursor = 0
        Continuous_scan = from_binary('uint32', response['body'][cursor: cursor+4])
        cursor += 4
        Bouncy_scan = from_binary('uint32', response['body'][cursor: cursor+4])
        cursor += 4
        Autosave = from_binary('uint32', response['body'][cursor: cursor+4])
        cursor += 4
        Series_name_size = from_binary('int', response['body'][cursor: cursor+4])
        cursor += 4
        Series_name = from_binary('string', response['body'][cursor: cursor+Series_name_size])
        cursor += Series_name_size
        Comment_size = from_binary('int', response['body'][cursor: cursor+4])
        cursor += 4
        Comment = from_binary('string', response['body'][cursor: cursor+Comment_size])  
        return {
            'Continuous scan': Continuous_scan,
            'Bouncy scan': Bouncy_scan,
            'Autosave': Autosave,
            'Series name size': Series_name_size,
            'Series name': Series_name,
            'Comment size': Comment_size,
            'Comment': Comment

        }
    ''' 
    - Continuous scan (unsigned int32) sets whether the scan continues or stops when a frame has been
    completed. 0 means no change, 1 is On, and 2 is Off
    - Bouncy scan (unsigned int32) sets whether the scan direction changes when a frame has been completed. 0
    means no change, 1 is On, and 2 is Off
    - Autosave (unsigned int32) defines the save behavior when a frame has been completed. "All" saves all the
    future images. "Next" only saves the next frame. 0 means no change, 1 is All, 2 is Next, and 3 sets this
    feature Off
    - Series name size (int) is the size in bytes of the Series name string
    - Series name (string) is base name used for the saved images
    - Comment size (int) is the size in bytes of the Comment string
    - Comment (string) is comment saved in the file
    '''
    def ScanPropsSet(self, Continuous_scan = 0, Bouncy_scan = 0, Autosave = 0, Series_name = 'unnamed', Comment = ''): 
        body = b''
        datatype = ''
        body_size = 20+len(Series_name)+len(Comment)
        body += to_binary('uint32', Continuous_scan)
        body += to_binary('uint32', Bouncy_scan)
        body += to_binary('uint32', Autosave)
        body += to_binary('int', len(Series_name))
        body += to_binary('string', Series_name)
        body += to_binary('int', len(Comment))
        body += to_binary('string', Comment)
        header = construct_header('Scan.PropsSet', body_size)
        try:
            # Acquire an atomic lock to prevent receiving a response that is unrelated to the request.
            # This is actually unnecessary right now since there are no features that use concurrency.
            self.lock.acquire()

            response = self.transmit(header+body)
        except:
            raise
        finally:
            self.lock.release() # Release lock

    
    def ScanFrameData(self, channel_index, data_dir=1):
        '''
        Unstable function!

        It will return the Scan data immediately, if call this function wihle the scan is running, the empty position will be filled with 'nan' value.

        Use the WaitEndOfScan() function to make sure the scan is finished before call this function if the complete image data is needed.

        !!!Try not to use this function in a loop, it may cause the TCP to crash!!!

        Request the data over and over again in a stort time will return many 'nan' values in the data array.

        more details can be get from the function 'lineScanGet()' and 'lineScanmode()'

        '''
        response = self.send('Scan.FrameDataGrab', 'uint32',
                             channel_index, 'uint32', data_dir)
        cursor = 0
        channel_name_length = from_binary('int', response['body'][cursor: cursor+4])
        cursor += 4
        channel_name = from_binary(
            'string', response['body'][cursor: cursor + channel_name_length])
        cursor += channel_name_length
        Scan_data_row = from_binary('int', response['body'][cursor: cursor + 4])
        cursor += 4
        Scan_data_col = from_binary('int', response['body'][cursor: cursor + 4])
        cursor += 4
        data = np.empty((Scan_data_row, Scan_data_col))
        for i in range(Scan_data_row):
            for j in range(Scan_data_col):
                data[i, j] = from_binary(
                    'float32', response['body'][cursor: cursor + 4])
                cursor += 4
        scan_direction = from_binary(
            'uint32', response['body'][cursor: cursor + 4])
        cursor += 4
        return {
            'data': data,
            'row': Scan_data_row,
            'col': Scan_data_col,
            'scan_direction': scan_direction,
            'channel_name': channel_name
        }

    def to_nano(self, value):
        return "{:.1f}n".format(value/1e-9)

    def channel_name_filter(self, channel_name: str):
        '''
        Local nanonis mechanine only have 8 Input/Output, so index with bigger than 8 should be removed.
        '''
        if channel_name.startswith('Input'):
            ind = int(channel_name[5:7])
            if ind > 8:
                return False
            else:
                return True
        elif channel_name.startswith('Output'):
            ind = int(channel_name[6:8])
            if ind > 8:
                return False
            else:
                return True
        else:
            return True

    def WaitEndOfScan(self):
        self.send('Scan.WaitEndOfScan', 'int', -1)

    def isZCtrlWork(self):
        '''
        make sure it is current mode.
        '''
        curr = self.CurrentGet()
        point = self.SetpointGet()
        if (abs(curr)-abs(point))/abs(point) < 50:
            return True
        else:
            return False

    def WaitForZCtrlWork(self):
        while not self.isZCtrlWork():
            time.sleep(0.5)

    '''
    ZCtrl.WithdrawRateGet
    Returns the Z-Controller withdraw slew rate in meters per second.
    Arguments: None
    Return arguments (if Send response back flag is set to True when sending request message):
    - Withdraw slew rate (m/s) (float32)
    - Error described in the Response message>Body section
    '''
    def WithdrawRateGet(self):
        parsedResponse = self.parse_response(
            self.send('ZCtrl.WithdrawRateGet'), 'float32')
        return parsedResponse['0']

    '''
    TipShaper

    Unfortunately, up to 2024/03/04 16:57, the TipShaper relevant TCP API is not available, so we can only use the code to simulate the correction process, 
    and there is no API in TCP to control the Z-axis moving speed, so we can only move the Tip step by step to simulate uniform motion. 
    
    TipShaper by Zhiwen Zhu at 2024/03/04 16:57      zhiwenzhu@shu.edu.cn
    
    Default parameter:
    Switch_Off_Delay = 0.05     #(s)  0.05s→ 50ms   , wait for Switch_Off_Delay then start the tip movement
    Lift_Time_1 = 0.1           #(s)  0.1s→ 100ms   , the tip will use Lift_Time_1 ms to lift to the target deepth
    Bias_Setting_Time = 0.06    #(s)  0.06s→ 60ms   , the tip will use Bias_Setting_Time ms to stay at the target deepth. (Bias at Lifting_Bias)
    Lift_Time_2 = 0.06          #(s)  0.06s→ 60ms   , the tip will use Lift_Time_2 ms to lift to the target deepth. (Bias at Lifting_Bias)

    TipLift = '-4n'             #(m) -4nm
    Lifting_Bias = -1.0         #(V) -1V
    Lifting_Height = '2n'       #(m)  2nm

    End_Wait_Time = 0.01        #(s)  0.01s→ 10ms   , wait for End_Wait_Time then stop the whole process

    stepstotarget = 10          # how many steps to the target. (simulated the uniform motion of the tip shaper)
    '''
    def TipShaper(self, Switch_Off_Delay = 0.05, Lift_Time_1 = 0.1, Bias_Setting_Time = 0.06, 
                       Lift_Time_2 = 0.06, TipLift = '-4n', Lifting_Bias = -1.0, Lifting_Height = '2n', 
                       End_Wait_Time = 0.01, stepstotarget = 10):

        TipLift = self.try_convert(TipLift)
        Lifting_Height = self.try_convert(Lifting_Height)
        
        Origin_ScanStatus = self.ScanStatusGet()#check the scan status
        if Origin_ScanStatus == 1:
            self.ScanPause()                                #Pause the scan (not stop), if the scan is running

        ZCtrl_status = self.ZCtrlOnOffGet()                 # get current Z control status
        
        Z_current = self.TipZGet()                          # get current Z position
        
        Bias_current = self.BiasGet()                       # get current Bias
        
        self.ZCtrlOnOffSet("off")                           # close Z control
        
        time.sleep(Switch_Off_Delay)                        # wait for Switch_Off_Delay

        # use for loop to simulate the uniform motion of the tip
        for i in range(stepstotarget):
            
            self.TipZSet(Z_current + (i+1)*(TipLift/stepstotarget))# set the new Z position
            
            time.sleep(Lift_Time_1/stepstotarget)           # wait for Lift_Time_1/stepstotarget time
        
        self.BiasSet(Lifting_Bias)                          # set the new Bias
        
        time.sleep(Bias_Setting_Time)                       # wait for Bias_Setting_Time

        # use for loop to simulate the uniform motion of the tip
        for i in range(stepstotarget):
            
            self.TipZSet(Z_current + TipLift + (i+1)*(Lifting_Height - TipLift)/stepstotarget)  # set the new Z position → Lifting_Height
            
            time.sleep(Lift_Time_2/stepstotarget)           # wait for Lift_Time_2/stepstotarget time
        
        self.BiasSet(Bias_current)                          # set the new Bias → Bias_current
        
        time.sleep(End_Wait_Time)                           # wait for End_Wait_Time

        # reopen Z control or not?
        if ZCtrl_status == 1:
            self.ZCtrlOnOffSet("on")
        else:
            self.ZCtrlOnOffSet("off")
        
        if Origin_ScanStatus == 1:
            self.ScanResume()# Resume the scan (not restart), if the scan was running

    



    def lineScanmode(self,lineScan_buffer, angle=0, abslute_1D = True):
        '''
        nanonis in TCP don't have a line scan mode, so we need to set it manually.
        '''
        ScanFrame = self.ScanFrameGet()
        # print(ScanFrame)
        ScanBuffer = self.ScanBufferGet()
        # print(ScanBuffer)
        if ScanBuffer['Pixels'] != lineScan_buffer and ScanFrame['height'] != ScanFrame['width']/ScanBuffer['Pixels']*lineScan_buffer:
            # only set 3 lines to scan, there will be some 'nan' vlues in the scan data if the scan lines is 1 or 2
            self.ScanBufferSet(ScanBuffer['Pixels'], lineScan_buffer, ScanBuffer['Channel_indexes']) 
            # downscale pixels simultaneously
            if abslute_1D :
                ScanFrame['height'] = 1e-15
            else:
                ScanFrame['height'] = ScanFrame['width']/ScanBuffer['Pixels']*lineScan_buffer

            self.ScanFrameSet( ScanFrame['center_x'], ScanFrame['center_y'], ScanFrame['width'], ScanFrame['height'], angle=angle ) 
        #print lineScan has been set
        print('lineScan has been set')
        return {'Pixels': ScanBuffer['Pixels'], 'width':ScanFrame['width'], 'angle':angle, 'Channel_indexes':ScanBuffer['Channel_indexes']} 
    
    def lineScanGet(self, signal_index ,fit_line = True):
        while True:
            Scaninfo_for = self.ScanFrameData(signal_index, data_dir=1)# 30 is the number of signal channel which present Z(m) in simulation mode
            Scaninfo_back = self.ScanFrameData(signal_index, data_dir=0)
        
            if Scaninfo_for['data'][0][0] == Scaninfo_for['data'][0][0] or Scaninfo_for['data'][-1][-1] == Scaninfo_for['data'][-1][-1]:
                if Scaninfo_for['data'][0][0] != Scaninfo_for['data'][0][0]: # the left top data is nan, which means the scan directer is 'up'!!!
                    for i in range(0, Scaninfo_for['row'], 1):
                        if Scaninfo_for['data'][i][0] == Scaninfo_for['data'][i][0] and Scaninfo_for['data'][i][-1] == Scaninfo_for['data'][i][-1] and Scaninfo_back['data'][i][0] == Scaninfo_back['data'][i][0] and Scaninfo_back['data'][i][-1] == Scaninfo_back['data'][i][-1]:
                            line_Scan_data_for = Scaninfo_for['data'][i]
                            line_Scan_data_back = Scaninfo_back['data'][i]
                            return {'line_Scan_data_for':line_Scan_data_for, 'line_Scan_data_back':line_Scan_data_back}
                elif Scaninfo_for['data'][-1][-1] != Scaninfo_for['data'][-1][-1]: # the right bottom data is nan, which means the scan directer is 'down'!!!
                    for i in range(Scaninfo_for['row']-1, 0, -1):

                        if Scaninfo_for['data'][i][0] == Scaninfo_for['data'][i][0] and Scaninfo_for['data'][i][-1] == Scaninfo_for['data'][i][-1] and Scaninfo_back['data'][i][0] == Scaninfo_back['data'][i][0] and Scaninfo_back['data'][i][-1] == Scaninfo_back['data'][i][-1]:
                            line_Scan_data_for = Scaninfo_for['data'][i]
                            line_Scan_data_back = Scaninfo_back['data'][i]
                            return {'line_Scan_data_for':line_Scan_data_for, 'line_Scan_data_back':line_Scan_data_back}
                
        

    def lineScanMoniterGet(self, lineScan_buffer=3, both_direction = True):    # lineScan_buffer should not be smaller than 3 and bigger than 50
        '''
        Can be proved.
        '''
        lineScanArgsGet = self.lineScanmode(lineScan_buffer = lineScan_buffer)
        Channel_indexes = lineScanArgsGet['Channel_indexes']
        Z_m_indexes = Channel_indexes[-1]
        self.ZCtrlOnSet()
        self.WaitForZCtrlWork()
        self.ScanStart()
        # self.WaitEndOfScan()
        #real-time line scan plot
        x = np.linspace(0, lineScanArgsGet['width'], lineScanArgsGet['Pixels'])
        y = np.zeros(lineScanArgsGet['Pixels'])

        plt.ion()
        
        figure, ax = plt.subplots(figsize=(5,3))
        line1, = ax.plot(x, y)
        line2, = ax.plot(x, y)
        
        plt.xlabel("width",fontsize=10)
        plt.ylabel("Z(m)",fontsize=10)
        # change the Handle of the figure as 'line scan monitor'
        figure.canvas.set_window_title('line Scan Monitor')
        while True:
            Scaninfo_for = self.ScanFrameData(Z_m_indexes, data_dir=1)# 30 is the number of signal channel which present Z(m), while 14 inm, real Scan
            if both_direction:
                Scaninfo_back = self.ScanFrameData(Z_m_indexes, data_dir=0)
            '''
            There is no corresponding TCP API provided,
            when Bouncy scan is turned on (usually turned on) for real-time scanning,
            which needs to be judged by scanning the data structure.
            When scanning down, traverse in reverse order to find the latest data;
            when scanning up, traverse in forward order to find the latest data
            '''
            if Scaninfo_for['data'][0][0] != Scaninfo_for['data'][0][0]: # the left top data is nan, which means the scan directer is 'up'!!!
                for i in range(0, Scaninfo_for['row'], 1):
                    if Scaninfo_for['data'][i][0] == Scaninfo_for['data'][i][0] and Scaninfo_for['data'][i][-1] == Scaninfo_for['data'][i][-1] and Scaninfo_back['data'][i][0] == Scaninfo_back['data'][i][0] and Scaninfo_back['data'][i][-1] == Scaninfo_back['data'][i][-1]:
                        line_Scan_data_for = Scaninfo_for['data'][i]
                        line_Scan_data_back = Scaninfo_back['data'][i]
                        print('No.{} row'.format(str(i)))
                        plt.ylim(line_Scan_data_for.min(), line_Scan_data_for.max())
                        line1.set_xdata(x)
                        line1.set_ydata(line_Scan_data_for)
                        line2.set_xdata(x)
                        line2.set_ydata(line_Scan_data_back)
                        figure.canvas.draw()
                        figure.canvas.flush_events()
                        break


            elif Scaninfo_for['data'][-1][-1] != Scaninfo_for['data'][-1][-1]: # the right bottom data is nan, which means the scan directer is 'down'!!!
                for i in range(Scaninfo_for['row']-1, 0, -1):

                    if Scaninfo_for['data'][i][0] == Scaninfo_for['data'][i][0] and Scaninfo_for['data'][i][-1] == Scaninfo_for['data'][i][-1] and Scaninfo_back['data'][i][0] == Scaninfo_back['data'][i][0] and Scaninfo_back['data'][i][-1] == Scaninfo_back['data'][i][-1]:
                        line_Scan_data_for = Scaninfo_for['data'][i]
                        line_Scan_data_back = Scaninfo_back['data'][i]
                        print('No.{} row'.format(str(i)))
                        plt.ylim(line_Scan_data_for.min(), line_Scan_data_for.max())
                        line1.set_xdata(x)
                        line1.set_ydata(line_Scan_data_for)
                        line2.set_xdata(x)
                        line2.set_ydata(line_Scan_data_back)
                        figure.canvas.draw()
                        figure.canvas.flush_events()
                        break





    '''
    Util.SessionPathGet
    Returns the session path.
    Arguments: None
    Return arguments (if Send response back flag is set to True when sending request message):
    - Session path size (int) is the number of characters of the Session path string
    - Session path (string)
    - Error described in the Response message>Body section
    '''
    def SessionPathGet(self):
        response = self.send('Util.SessionPathGet')
        cursor = 0
        Session_path_size = from_binary('int', response['body'][cursor: cursor+4])
        cursor += 4
        Session_path = from_binary('string', response['body'][cursor: cursor+Session_path_size])
        return {'Session path size': Session_path_size, 'Session path': Session_path}
    

    '''
    Util.SettingsLoad
    Loads the settings from the specified .ini file.
    Arguments:
    - Settings file path size (int) is the number of characters of the Settings file path string
    - Settings file path (string) is the path of the settings file to load
    - Load session settings (unsigned int32) automatically loads the current settings from the session file
    bypassing the settings file path argument, where 0=False and 1=True
    Return arguments (if Send response back flag is set to True when sending request message):
    - Error described in the Response message>Body section
    '''
    def SettingsLoad(self, Settings_file_path, Load_session_settings = 0):
        header = construct_header('Util.SettingsLoad', 4+len(Settings_file_path)+4)
        body = b''
        body += to_binary('int', len(Settings_file_path))
        body += to_binary('string', Settings_file_path)
        body += to_binary('uint32', Load_session_settings)
        self.transmit(header+body)
        
    '''
    Util.LayoutLoad
    Loads a layout from the specified .ini file.
    Arguments:
    - Layout file path size (int) is the number of characters of the layout file path string
    - Layout file path (string) is the path of the layout file to load
    - Load session layout (unsigned int32) automatically loads the layout from the session file bypassing the
    layout file path argument, where 0=False and 1=True
    Return arguments (if Send response back flag is set to True when sending request message):
    - Error described in the Response message>Body section
    '''
    def LayoutLoad(self, Layout_file_path, Load_session_layout = 0):
        header = construct_header('Util.LayoutLoad', 4+len(Layout_file_path)+4)
        body = b''
        body += to_binary('int', len(Layout_file_path))
        body += to_binary('string', Layout_file_path)
        body += to_binary('uint32', Load_session_layout)
        self.transmit(header+body)


    # def a function to stop the scan and withdraw the tip
    def StopScanAndWithdraw(self):
        self.ScanStop()
        self.Withdraw()

    # def a function adjust the tip to the center of the piezo after auto approach or thermal drift
    def AdjustTipToPiezoCenter(self,threshold = 0.7):
        # get the piezo range
        Z_lim = self.ZLimitsGet()
        time.sleep(0.5)
        # get the current tip position
        Z_pos = self.TipZGet()
        # if the Z position is out of the Z limit 80%, adjust the tip position
        if Z_pos > Z_lim['high'] * threshold or Z_pos < Z_lim['low'] * threshold:
            self.ScanPause()
            self.Withdraw()
            if Z_pos > Z_lim['high']*threshold:                     # the tip is too close to the surface, so move the tip Z-
                Tip_move_direction = 'Z-'
                gorl_Z_limit = [Z_lim['low'], Z_lim['low'] * 0.2]   
            elif Z_pos < Z_lim['low']*threshold:                    # the tip is too far from the surface, so move the tip Z+ 
                Tip_move_direction = 'Z+'                 
                gorl_Z_limit = [Z_lim['high'] * 0.2, Z_lim['high']]
            
            self.MotorMoveSet(Tip_move_direction, 2)        # move the Motor 1 step
            self.AutoApproachSet()                          # open the auto approach
            self.ZCtrlOnSet()
            # self.WaitForZCtrlWork()          
            time.sleep(4)                                   # wait the tip stable on the surface
            Z_pos_after = self.TipZGet()                    # get the tip Z position again
            # step_length = Z_pos - Z_pos_after               # calculate the step length
            # # try set tip Z position to the center of the piezo
            # self.MotorMoveSet(Tip_move_direction, 2)

            while Z_pos_after >= gorl_Z_limit[1] or Z_pos_after <= gorl_Z_limit[0]:

                self.Withdraw()
                self.MotorMoveSet(Tip_move_direction, 1)
                self.AutoApproachSet()
                self.ZCtrlOnSet()
                # self.WaitForZCtrlWork()
                time.sleep(3.5)
                Z_pos = Z_pos_after
                Z_pos_after = self.TipZGet()
            # if the tip pos has been adjusted, return True
            return 1
        # if the tip pos is acceptable(no need to adjust), return False
        return 0





class Operate(metaclass=ABCMeta):
    '''
    The base class of all Operations.
    '''

    def __init__(self, session: NanonisController):
        self.session = session

    @abstractmethod
    def safety_check(self):
        pass

    @abstractmethod
    def _operate(self):
        pass

    def _reset(self):
        pass

    def do(self):
        if self.safety_check():
            result = self._operate()
            self._reset()
            return result
        else:
            raise nanonisException('Safety check failed.')















class ExceptionType(Enum):
    '''
    Exception types
    '''
    UNDEFINED = 0

    Z_TIP_TOUCHED = 101
    Z_HIGH_LIMIT_REACHED = 110
    Z_LOW_LIMIT_REACHED = 111

    PROCESS_FINISHED = 200


def setupLog():
    r'''
    simple logger setup
    '''

    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y/%d/%m %H:%M:%S"
    logging.basicConfig(level=logging.DEBUG,
                        format=LOG_FORMAT, datefmt=DATE_FORMAT)



