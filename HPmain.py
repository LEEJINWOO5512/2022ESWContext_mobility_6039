#!/usr/bin/env python3
import serial
import time
import threading
import ringing_alarm as alarm

HP = serial.Serial('/dev/ttyACM2',9600, timeout=1)
MT = serial.Serial('/dev/ttyACM1',9600)
BT = serial.Serial('/dev/ttyACM0',9600)


def controlMotor(ain):
    # print("MTain: ",ain, "\n")
    if(ain==1):
        MT.write(b'1')
    elif(ain==0):
        MT.write(b'0')

def sendingBT(ain):
    # print("BTain: ",ain, "\n")
    if(ain==1):
        BT.write(b'1')
    elif(ain==0):
        BT.write(b'0')

def main_func(av):
    
    tM = threading.Thread(target=controlMotor, args=(av,))
    tB = threading.Thread(target=sendingBT, args=(av,))
    t1 = threading.Thread(target = alarm.select_alarm, args = (av, ))
    t1.deamon = True
    
    tM.start()
    tB.start()
    t1.start()
    
    tM.join()
    tB.join()

if __name__ == '__main__':
    
    while True:
        HP.reset_input_buffer()
        hp = HP.readline().decode('utf-8').strip()
        hpi=0        
        if(len(hp)!=0): #Str to Int
            hpi=int(hp)
        if(0<=hpi<80):
            a=1
        else:
            a=0
        print("HPulse: ", hpi, "  accident: ", a,"\n")
        
        main_func(a)
        


