
# coding: utf-8
# In[ ]:

import pygame
'''
[pygame 라이브러리]
* pygame 라이브러리: 대표적인 게임 작성용 라이브러리
* mixer 모듈: pygmame 내에서 음악 재생 기능 제공

[pygame.mixer에서 음악 재생하기]
https://www.jbmpa.com/pygame/10
1) Sound 객체 방법: Sound 객체 생성 -> play(), stop() 사용하기
    mySound = pygame.mixer.Sound(filename)
    mySound.play()
    mySound.stop() 
2) Streaming music 방법: 객체 생성 없이 파일을 스트리밍(music.load() 사용)
    pygame.mixer.music.load(filename)  // 한 번에 한 개 음악만 재생 가능
    pygame.mixer.music.play()  // play(-1): 무한 반복
    pygame.mixer.music.stop()
'''

def select_alarm(result) :
    if result == 1:
        sound_alarm("power_alarm.wav")

def sound_alarm(path) :  # path에 해당하는 음악파일 재생 함수
    pygame.mixer.init()  # mixer 모듈의 초기화 함수(라이브러리 초기화 필요)
    pygame.mixer.music.load(path)  # 음악파일 load
    pygame.mixer.music.play()  # load한 음악파일 재생
    

