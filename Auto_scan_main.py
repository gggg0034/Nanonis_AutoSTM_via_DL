from Auto_scan_class import *

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

        nanonis.create_trajectory(scan_qulity)                                          # create the trajectory

        if scan_qulity == 0 and nanonis.agent_upgrate:
            nanonis.DQN_upgrate()                                                       # optimize the model and update the target network

        nanonis.save_checkpoint()                                                       # save the checkpoint


        time.sleep(3)
    
    # nanonis.ScanStop()
    # nanonis.Withdraw()
    # nanonis.MotorMoveSet('Z-', 2000)