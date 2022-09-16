
# coding: utf-8
# In[ ]:

import time
'''
[라이브러리 불러오기]
https://wikidocs.net/77
1) 모듈 전체 가져오기: import 모듈
2) 모듈 내에서 필요한 것만 가져오기: from 모듈 import 이름

[time 라이브러리]
https://www.daleseo.com/python-time/
* time 라이브러리: epoch time(Unix timestamp)을 다룰 때 사용
* epoch time: UTC(GMT+0) 기준 1970.1.1. 00:00:00부터의 경과 시간을 나타냄(=timestamp)
- time() 함수: 현재 unix timestamp 소수 값으로 return
'''

def check_fps(prev_time) : # 웹캠에서 받는 영상의 fps(frames per second; 초당 프레임 수)를 확인하는 함수
    cur_time = time.time() #Import the current time in seconds
    one_loop_time = cur_time - prev_time
    prev_time = cur_time
    fps = 1/one_loop_time
    return prev_time, fps

