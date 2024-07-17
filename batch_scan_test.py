
import os
import shutil
import threading
import time
from multiprocessing import Pool, Process, Queue
from queue import Queue

import cv2
import keyboard
import matplotlib.pyplot as plt
import numpy as np

from core import NanonisController
from modules.TipPath_Fuc import *
from modules.TipShaper import TipShaper
from tasks.LineScanchecker import *

# initialize the nanonis thread status
lineScan_data_producer_status = 1
lineScan_data_consumer_status = 1
batch_scan_producer_status = 0
batch_scan_consumer_status = 0



nanonis = NanonisController()
# set a Queue for lineScanGet and the size of the Queue is 3
lineScanQueue = Queue(3)
ScandataQueue = Queue(4)
tippathvisualQueue = Queue(3)






def lineScan_threading():
    nanonis.lineScanMoniterGet(lineScan_buffer = 3)

def lineScan_data_producer(angle=0):
    global lineScan_data_producer_status
    nanonis.ScanPropsSet(Continuous_scan = 1, Bouncy_scan = 1,  Autosave = 3, Series_name = ' ', Comment = 'LineScan')  # do not save the line scan data
    lineScanArgsGet = nanonis.lineScanmode(3, angle=angle)                                                              #set nanonis in line scan parameters
    nanonis.ScanStart()                                                                                                 #start the line scan
    nanonis.WaitEndOfScan()                                                                                             #wait the line scan end
    while lineScan_data_producer_status:
        # print('lineScan_data_producer_works')
        lineScandata = nanonis.lineScanGet()                                                                            #get the line scan data     {'line_Scan_data_for':line_Scan_data_for, 'line_Scan_data_back':line_Scan_data_back}
        
        # print(lineScandata)
        lineScanQueue.put(lineScandata)                                                                                 #put the line scan data into the queue, blocking if Queue is full
    
    # clean the line scan data queue
    lineScanQueue.queue.clear()

def lineScan_data_consumer():                       #LineScan domian knowledge
    #########################################
    # the parameters for the line scan check
    #########################################
    for_back_threshold = 0.5
    before_after_threshold = 0.5
    max_min_threshold = 1000                    #  pm (*1e-12)
    #########################################
    
    global lineScan_data_producer_status
    global lineScan_data_consumer_status
    global batch_scan_producer_status
    global batch_scan_consumer_status
    global lineScan_check


    lineScan_data_consumer_status = 1
    check_times = 10
    result_list = []
    
    lineScandata_1 = lineScanQueue.get()
    # print('lineScan_data_consumer_works')
    lineScandata_1_for = fit_line(lineScandata_1['line_Scan_data_for'])                     # t-1 line scan data for
    lineScandata_1_back = fit_line(lineScandata_1['line_Scan_data_back'])                   # t-1 line scan data back
    # lineScandata_1_for = normalize_to_image(lineScandata_1_for)                             # normalize the line scan data
    # lineScandata_1_back = normalize_to_image(lineScandata_1_back)    
    
    
    while lineScan_data_consumer_status:

        for_back_result = linescan_similarity_check(lineScandata_1_for, lineScandata_1_back, threshold = for_back_threshold) # check the for and back data
        # print("for_back_result" + str(for_back_result))

        max_min_check_for = linescan_max_min_check(lineScandata_1_for, threshold = max_min_threshold)                  # check the max and min value of the line scan data
        max_min_check_back = linescan_max_min_check(lineScandata_1_back, threshold = max_min_threshold)

        # phase_diff1 = find_phase_diff(lineScandata_1_for, lineScandata_1_back)
        # print("phase_diff_for_back" + str(phase_diff1))

        lineScandata_2 = lineScanQueue.get()                                                    #get the line scan data from the queue, blocking if Queue is empty

        lineScandata_2_for = fit_line(lineScandata_2['line_Scan_data_for'])                     # t line scan data for
        lineScandata_2_back = fit_line(lineScandata_2['line_Scan_data_back'])                   # t line scan data back
        # lineScandata_2_for = normalize_to_image(lineScandata_2_for)                             # normalize the line scan data
        # lineScandata_2_back = normalize_to_image(lineScandata_2_back)

        phase_diff2 = find_phase_diff(lineScandata_1_for, lineScandata_2_for)                    # calculate the phase difference between t-1 and t
        if phase_diff2 < 10:
            Thermo_diff = 1
        else:
            Thermo_diff = 0
        # print("phase_diff" + str(phase_diff2))

        before_after_result = linescan_similarity_check(lineScandata_1_for, lineScandata_2_for, threshold = before_after_threshold) # check the before and after data
        
        # print("before_after_result" + str(before_after_result))

        lineScandata_1_for = lineScandata_2_for                                                 # update the line scan data 
        lineScandata_1_back = lineScandata_2_back

        if for_back_result and max_min_check_for and max_min_check_back and before_after_result and Thermo_diff:
            result_list.append(1)
        else:
            result_list.append(0)
        
        # print(result_list)

        if len(result_list) > check_times + 1:
            result_list.pop(0)
            last_10 = result_list[-10:]
            if all(x == 1 for x in last_10):
                
                batch_scan_producer_status = 1
                batch_scan_consumer_status = 1
                lineScan_data_consumer_status = 0                                                   # stop the line scan data consumer
                lineScan_data_producer_status = 0                                                   # stop the line scan data producer
                lineScanQueue.queue.clear()                                                         # clear the line scan data queue

                lineScan_check = 1
                # lineScan_check = 0
                print('the line scan is good, start the batch scan')
                break
            elif all(x == 0 for x in last_10):
                
                batch_scan_producer_status = 0
                batch_scan_consumer_status = 0
                lineScan_data_consumer_status = 0                                                   # stop the line scan data consumer
                lineScan_data_producer_status = 0                                                   # stop the line scan data producer
                lineScanQueue.queue.clear()                                                         # clear the line scan data queue

                lineScan_check = 0
                print('the line scan is bad, stop the batch scan')
                break

# start the batch scan
def batch_scan_producer(Scan_posion = (0.0 ,0.0), Scan_edge = "30n", Scan_pix = 304 ,  angle = 0):
    if batch_scan_producer_status:
        
        nanonis.ScanPropsSet(Continuous_scan = 2 , Bouncy_scan = 2,  Autosave = 1, Series_name = ' ', Comment = 'inter_closest')# close continue_scan & bouncy_scan, but save all data
        nanonis.ScanFrameSet(Scan_posion[0], Scan_posion[1], Scan_edge, Scan_edge, angle=angle)
        nanonis.ScanBufferSet(Scan_pix, Scan_pix)
        time.sleep(0.5)
        nanonis.ScanStart()
        nanonis.WaitEndOfScan()
        Scan_data_for = nanonis.ScanFrameData(30, data_dir=1)
        time.sleep(0.5)
        Scan_data_back = nanonis.ScanFrameData(30, data_dir=0)

        ScandataQueue.put({'Scan_data_for':Scan_data_for, 'Scan_data_back':Scan_data_back}) # put the batch scan data into the queue, blocking if Queue is full

        return {'Scan_data_for':Scan_data_for, 'Scan_data_back':Scan_data_back}

# def a function to visualize and save the batch scan data
##################
# can be improved
##################
def batch_scan_consumer():
    global start_time
    while True:
        if not ScandataQueue.empty():
            Scan_data = ScandataQueue.get() # get the batch scan data from the queue, blocking if Queue is empty
            # print('batch_scan_consumer_works')
            
            
            Scan_data_for = Scan_data['Scan_data_for']['data']
            
            Scan_data_back = Scan_data['Scan_data_back']['data']
            image_for = normalize_to_image(subtract_plane (normalize_to_image(Scan_data_for)))
            image_back = normalize_to_image(subtract_plane (normalize_to_image(Scan_data_back)))



            main_data_save_path = '.\\log'
            right_now = time.strftime('%H-%M-%S',time.localtime(time.time()))
            txt_data_save_path =  main_data_save_path + '\\' + start_time + '\\' + 'txt'
            image_data_save_path =  main_data_save_path + '\\' + start_time + '\\' + 'image'
            txt_data_save_path_for = txt_data_save_path + '\\' + 'Scan_data_for_'+ right_now +'.txt'
            txt_data_save_path_back = txt_data_save_path + '\\' + 'Scan_data_back_'+ right_now +'.txt'
            image_data_save_path_for = image_data_save_path + '\\' + 'Scan_data_for'+ right_now +'.png'
            image_data_save_path_back = image_data_save_path + '\\' + 'Scan_data_back'+ right_now +'.png'

            # check the save path exist or not
            if not os.path.exists(txt_data_save_path):
                os.makedirs(txt_data_save_path)
            if not os.path.exists(image_data_save_path):
                os.makedirs(image_data_save_path)
            # save Scan_data_for and Scan_data_back in one txt file
            np.savetxt(txt_data_save_path_for, Scan_data_for, fmt='%f', delimiter=' ')
            np.savetxt(txt_data_save_path_back, Scan_data_back, fmt='%f', delimiter=' ')
            # save the image_for and image_back in one png file
            cv2.imwrite(image_data_save_path_for, image_for)
            cv2.imwrite(image_data_save_path_back, image_back)


            

            # save Scan_data_for and Scan_data_back in one txt file
            np.savetxt(txt_data_save_path_for, Scan_data_for, fmt='%f', delimiter=' ')
            np.savetxt(txt_data_save_path_back, Scan_data_back, fmt='%f', delimiter=' ')
            # save the image_for and image_back in one png file
            cv2.imwrite(image_data_save_path_for, image_for)
            cv2.imwrite(image_data_save_path_back, image_back)
            # name the two images widows
            cv2.namedWindow('image_for', cv2.WINDOW_NORMAL)
            cv2.namedWindow('image_back', cv2.WINDOW_NORMAL)
            # set the size of the two images windows
            cv2.resizeWindow('image_for', 400, 400)
            cv2.resizeWindow('image_back', 400, 400)
            # show the two images
            cv2.imshow('image_for', image_for)
            cv2.imshow('image_back', image_back)
        cv2.waitKey(100)

        if cv2.waitKey(100) & 0xFF == ord('q') or batch_scan_consumer_status == 0:                                                         # press 'q' to quit the tip path visualization and save the tip path image
            break
    
# def a function to visualize the tip path
def tip_path_visualization(scan_square_edge):
    global tip_path_visual
    global circle_list
    global start_time
    visual_circle_buffer_list = []
    
    square_color_hex = "#BABABA"
    square_bad_color_hex = "#FE5E5E"
    line_color_hex = "#C7C7C7"
    square_good_color = Hex_to_BGR(square_color_hex)
    square_bad_color = Hex_to_BGR(square_bad_color_hex)
    line_color = Hex_to_BGR(line_color_hex)

    cv2.namedWindow('Tip Path', cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('Tip Path', 800, 800)
    

    # creat a 400*400 pix white image by numpy array
    img = np.ones((plane_size, plane_size, 3), np.uint8) * 255
    while True:

        if not tippathvisualQueue.empty():
            circle = tippathvisualQueue.get()
            # print('tip_path_visualization_works')
            
            try :                                                                                       # try to get the scan_qulity
                if circle[3] == 1:
                    square_color = square_good_color
                else:
                    square_color = square_bad_color
            except:
                pass

            if len(circle) == 2:                                                                        # data is a point which is the point of ready to scan 
                cv2.circle(img, (round(circle[0]), round(circle[1])), round(scan_square_edge/5) , (255, 0, 0), -1)
            elif len(circle) == 4:                                                                      # data is a circle which is the circle of already scanned
                left_top, right_bottom = center_to_square((circle[0], circle[1]), scan_square_edge)
                if len(circle_list) == 1:
                    
                    cv2.rectangle(img, left_top, right_bottom, square_color, -1)
                    visual_circle_buffer_list.append(circle)                                            # add the first circle to the visual_circle_buffer_list

                elif len(circle_list) > 1:
                    (Xn_1,Yn_1) = (visual_circle_buffer_list[-1][0], visual_circle_buffer_list[-1][1])
                    cv2.rectangle(img, left_top, right_bottom, square_color, -1)                        # draw the square
                    cv2.line(img, (round(Xn_1), round(Yn_1)), (round(circle[0]), round(circle[1])), line_color, 2)                                # ues the last circle center to draw the line to show the tip path
                    visual_circle_buffer_list.append(circle)                                            # add the new circle to the visual_circle_buffer_list
                    # delete the first circle in the visual_circle_buffer_list
                    if len(visual_circle_buffer_list) > 2:
                        visual_circle_buffer_list.pop(0)

                else:
                    raise ValueError('the length of the circle list is not right')
            else:
                raise ValueError('the circle data is not a point or a circle')

            cv2.imshow('Tip Path', img)
        cv2.waitKey(100)
        
        if cv2.waitKey(100) & 0xFF == ord('q') or tip_path_visual == 0:                                                         # press 'q' to quit the tip path visualization and save the tip path image
            # check the save path exist or not
            if not os.path.exists('.\\log'):
                os.makedirs('.\\log')
            cv2.imwrite('.\\log\\Tip Path_'+ start_time +'.png', img)                # save the tip path image named by the date
            print('the tip path image is saved')
            cv2.destroyAllWindows()
            break

#  a function to shut all the threading
def shut_all_threading():
    global lineScan_data_producer_status
    global lineScan_data_consumer_status
    global batch_scan_producer_status
    global batch_scan_consumer_status
    global tip_path_visual

    lineScan_data_producer_status = 0
    lineScan_data_consumer_status = 0
    batch_scan_producer_status = 0
    batch_scan_consumer_status = 0
    tip_path_visual = 0


if __name__ == '__main__':


    ##################################################################################################################################
    Scan_edge = "30n"                       # set the initial scan square edge length                           30pix  ==>>  30nm 
    scan_square_Buffer_pix = 256            # scan pix
    plane_edge = "2u"                       # plane size repersent the area of the Scanable surface  2um*2um    2000pix  ==>>  2um
    ##################################################################################################################################


    # get the time now as the format of '2020-01-01 00:00:00'
    start_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
    print('The start time is ' + start_time)
    # get the session by default
    default_session = '.\\log'
    
    sxm_save_path =  '.\\log\\' + start_time
    # create a folder named by the now
    if not os.path.exists(sxm_save_path):
        os.makedirs(sxm_save_path)
    # copy the '.ini' file in default_session into sxm_save_path
    for file in os.listdir(default_session):
        if file.endswith('.ini'):
            ini_file_path = os.path.join(default_session, file)
            shutil.copy(ini_file_path, sxm_save_path)
    

    # change the relative path to absolute path
    sxm_save_path = os.path.abspath(sxm_save_path)

    # load the Layout
    # nanonis.LayoutLoad(sxm_save_path)
    # nanonis.SettingsLoad(sxm_save_path)



    plane_size = int(nanonis.convert(plane_edge)*10**9)
    scan_square_edge = int(nanonis.convert(Scan_edge)*10**9)                                  # get the scan square edge length
    # set the initial radius of the first circle                       
    R_init = round(scan_square_edge*(math.sqrt(2))*1.5)
    R_max = R_init*3
    R_step = int(0.5*R_init)
    R = R_init

    
    circle_list = [] # initialize the circle list
    inter_closest = (round(plane_size/2), round(plane_size/2)) # initialize the inter_closest
    nanocoodinate_list = [] # initialize the nanocoodinate list
    
    
    tip_path_visual = 1
    # print('Lunch the tip path visualization')
    tip_path_visualization_thread = threading.Thread(target = tip_path_visualization, args=(scan_square_edge, ), daemon=True) # lunch the tip path visualization thread
    tip_path_visualization_thread.start()

    nanonis.ZCtrlOnSet() # turn on the Z control
    nanonis.WaitForZCtrlWork()

    time.sleep(0.5)

    while inter_closest[0] + 2*R <= plane_size and inter_closest[0] - 2*R >= 0 and inter_closest[1] + 2*R <= plane_size and inter_closest[1] - 2*R >= 0:  # check the circle is in the plane or not
        
        
        print('Scanning...')
        inter_closest = Next_inter(circle_list)
        tippathvisualQueue.put(inter_closest)
        if len(circle_list) <= 1:                                                                           # if the circle list is empty or only have one circle, initialize the angle and R
            angle = 0   
            R = R_init
        else:
            angle = angle_calculate(circle_list[-1][2:], inter_closest)                                     # Line scan direction
            print('The angle is ' + str(angle))

        nanocoodinate = pix_to_nanocoordinate((inter_closest))                                              # Line scan start point
        print('The nanocoodinate is ' + str(nanocoodinate))
        print('The inter_closest is ' + str(inter_closest))
        nanocoodinate_list.append(nanocoodinate)                                                             # add the nanocoodinate to the nanocoodinate list

        time.sleep(0.5)
        nanonis.TipXYSet(nanocoodinate[0], nanocoodinate[1])                                                 # set the tip position, the program will wait until the tip is stable
        time.sleep(0.5)

        nanonis.ScanFrameSet(nanocoodinate[0], nanocoodinate[1], Scan_edge, 1e-15, angle=angle)             # intialize the scan frame (0.0, 0.0)
        

        t1 = threading.Thread(target = lineScan_data_producer, args=(angle, ), daemon=True)                 # lunch the line scan data producer
        t2 = threading.Thread(target = lineScan_data_consumer, daemon=True)                                 # lunch the line scan data consumer
        t1.start()
        t2.start()

        t1.join()
        t2.join()

        time.sleep(0.5)
        if lineScan_check:
            R = R_init
            t3 = threading.Thread(target = batch_scan_producer, args=(nanocoodinate, Scan_edge, scan_square_Buffer_pix , 0 ), daemon=True)      # lunch the batch scan data producer
            t4 = threading.Thread(target = batch_scan_consumer, daemon=True)                                                                    # lunch the batch scan data consumer 
            t3.start()
            t4.start()

            t3.join()
            # t4.join()
            scan_qulity = 1                                                         # the scan is good
        else:                                                                                        
            if R > R_max:                                                           # if the radius is too large, stop increasing the radius
                R = R_max
            else:
                R += R_step                                                         # increase the radius
            scan_qulity = 0                                                         # the scan is bad

        circle_list.append([inter_closest[0], inter_closest[1], R, scan_qulity])                                    # add the circle to the circle list
        tippathvisualQueue.put([inter_closest[0], inter_closest[1], R, scan_qulity])                                # put the circle into the tip path visualization queue
        
        lineScan_data_producer_status = 1                                                                           # restart the line scan data producer
        lineScan_data_consumer_status = 1                                                                           # restart the line scan data consumer

    shut_all_threading()                                                                                            # shut all the threading

    nanonis.StopScanAndWithdraw()                                                                                   # stop the scan and withdraw the tip

    print('All the threading is shut \n the batch scan is over \n please check the log {}'.format(start_time))
    #get the time now as the format of '2020-01-01 00:00:00'
    end_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    print('The end time is ' + end_time)





    








    # lineScan_data_producer()
    # lineScan_threading()
    # initialize the analysis parameters


    
    
    
    
    
    
    # LinescanGet()
    # t1 = threading.Thread(target=lineScan_threading, daemon=True)
    # t1.start()
    # keyboard.wait('esc')

