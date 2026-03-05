import os
import cv2
import random
import numpy as np
from gtuner import *


class GCVWorker:
    def __init__(self, width, height):
        os.chdir(os.path.dirname(__file__))
        self.gcvdata = bytearray([0x00])
        self.noun = "user"
        self.nouns = ["user", "player", "member", "patron", "client", "consumer", "operator", "utilizer", "handler", "contender", "competitor", "contestant", "performer", "actor", "rival", "protagonist", "antagonist", "challenger", "adversary", "opponent", "friend", "teammate", "peer", "counterpart", "enemy"]
        self.frame_count = 0


    def __del__(self):
        del self.gcvdata


    def process(self, frame):
        self.gcvdata = bytearray([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])

        self.draw_menu_buttons(frame)
        self.draw_face_buttons(frame)
        self.draw_dpad(frame)
        self.draw_triggers(frame)
        self.draw_bumpers(frame)
        self.draw_right_stick(frame)
        self.draw_left_stick(frame)

        cv2.putText(frame, "Hello, " + str(self.noun), (790,200), cv2.FONT_HERSHEY_COMPLEX, 0.666, (255, 255, 255), 1, cv2.LINE_AA)
        if self.frame_count == 60:
            self.noun = self.nouns[random.randint(0, len(self.nouns)-1)]
            self.frame_count = 0
        cv2.putText(frame, str(self.frame_count), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.666, (255, 255, 255), 1, cv2.LINE_AA)
        self.frame_count += 1

        self.gcvdata = bytearray([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        self.gcvdata[0] = self.frame_count

        return frame, self.gcvdata
    
    def draw_menu_buttons(self, frame):
        buttons = {
            BUTTON_2: (250, 170),
            BUTTON_3: (320, 170)
        }
        byte_info = {
            BUTTON_2: (2, 1),
            BUTTON_3: (3, 1)
        }
        pos_offset = (0, 50)
        for button, pos in buttons.items():
            cv2.circle(frame, (pos[0] + pos_offset[0], pos[1] + pos_offset[1]), 12, (255, 255, 255), 3, cv2.LINE_AA)
            value = get_actual(button)
            color = int(value * 2.55)
            self.gcvdata[byte_info[button][0]] = int(value)
            cv2.circle(frame, (pos[0] + pos_offset[0], pos[1] + pos_offset[1]), 12, (color, color, color), -1, cv2.LINE_AA)
    
    def draw_face_buttons(self, frame):
        buttons = {
            BUTTON_14: (430, 140),
            BUTTON_15: (460, 170),
            BUTTON_16: (430, 200),
            BUTTON_17: (400, 170)
        }
        byte_info = {
            BUTTON_14: (20, 1),
            BUTTON_15: (21, 1),
            BUTTON_16: (22, 1),
            BUTTON_17: (23, 1)
        }
        pos_offset = (0, 50)
        for button, pos in buttons.items():
            cv2.circle(frame, (pos[0] + pos_offset[0], pos[1] + pos_offset[1]), 15, (255, 255, 255), 3, cv2.LINE_AA)
            value = get_actual(button)
            color = int(value * 2.55)
            self.gcvdata[byte_info[button][0]] = int(value)
            cv2.circle(frame, (pos[0] + pos_offset[0], pos[1] + pos_offset[1]), 15, (color, color, color), -1, cv2.LINE_AA)
    
    def draw_dpad(self, frame):
        dpad = {
            BUTTON_10: (190, 120),
            BUTTON_11: (190, 180),
            BUTTON_12: (160, 150),
            BUTTON_13: (220, 150)
        }
        byte_info = {
            BUTTON_10: (16, 1),
            BUTTON_11: (17, 1),
            BUTTON_12: (18, 1),
            BUTTON_13: (19, 1)
        }
        pos_offset = (0, 150)
        for button, pos in dpad.items():
            cv2.rectangle(frame, (pos[0] + pos_offset[0], pos[1] + pos_offset[1]), (pos[0] + pos_offset[0] + 27, pos[1] + pos_offset[1] + 27), (255, 255, 255), 3, cv2.LINE_AA)
            value = get_actual(button)
            color = int(value * 2.55)
            self.gcvdata[byte_info[button][0]] = int(value)
            cv2.rectangle(frame, (pos[0] + pos_offset[0], pos[1] + pos_offset[1]), (pos[0] + pos_offset[0] + 27, pos[1] + pos_offset[1] + 27), (color, color, color), -1, cv2.LINE_AA)
    
    def draw_triggers(self, frame):
        triggers = {
            BUTTON_8: (90, 260),
            BUTTON_5: (440, 260)
        }
        byte_info = {
            BUTTON_8: (11, 4),
            BUTTON_5: (5, 4)
        }
        pos_offset = (0, 50)
        for button, pos in triggers.items():
            value = get_actual(button)
            cv2.rectangle(frame, (pos[0] + pos_offset[0], pos[1] + pos_offset[1] + (int(value / 2))), (pos[0] + pos_offset[0] + 30, pos[1] + pos_offset[1] + 50), (255, 255, 255), 3, cv2.LINE_AA)
            color = int(value * 2.55)
            clamped = max(-100.0, min(100.0, float(value)))
            raw_int = int(clamped * 0x10000)
            self.gcvdata[byte_info[button][0]:byte_info[button][0] + byte_info[button][1]] = raw_int.to_bytes(byte_info[button][1], 'big', signed=True)
            cv2.rectangle(frame, (pos[0] + pos_offset[0], pos[1] + pos_offset[1] + (int(value / 2))), (pos[0] + pos_offset[0] + 30, pos[1] + pos_offset[1] + 50), (color, color, color), -1, cv2.LINE_AA)
    
    def draw_bumpers(self, frame):
        bumpers = {
            BUTTON_7: (95, 120),
            BUTTON_4: (395, 120)
        }
        byte_info = {
            BUTTON_7: (10, 1),
            BUTTON_4: (4, 1)
        }
        pos_offset = (0, 0)
        for button, pos in bumpers.items():
            cv2.rectangle(frame, (pos[0] + pos_offset[0], pos[1] + pos_offset[1]), (pos[0] + pos_offset[0] + 65, pos[1] + pos_offset[1] + 20), (255, 255, 255), 3, cv2.LINE_AA)
            value = get_actual(button)
            color = int(value * 2.55)
            self.gcvdata[byte_info[button][0]] = int(value)
            cv2.rectangle(frame, (pos[0] + pos_offset[0], pos[1] + pos_offset[1]), (pos[0] + pos_offset[0] + 65, pos[1] + pos_offset[1] + 20), (color, color, color), -1, cv2.LINE_AA)
    
    def draw_left_stick(self, frame):
        anchor = (130, 220)
        cv2.circle(frame, (int(anchor[0]), int(anchor[1])), 45, (255, 255, 255), 2, cv2.LINE_AA)
        x_val = get_actual(STICK_2_X)
        y_val = get_actual(STICK_2_Y)
        click = get_actual(BUTTON_9)
        click_color = int(click * 2.55)
        self.gcvdata[15] = int(click)
        cv2.circle(frame, (int(anchor[0] + x_val*0.20), int(anchor[1] + y_val*0.20)), 37, (click_color, click_color, click_color), -1, cv2.LINE_AA)
        xy_color = int((abs(x_val) + abs(y_val)) * 2.25) + 30
        self.gcvdata.extend(int(x_val * 0x10000).to_bytes(4, 'big', signed=True))
        self.gcvdata.extend(int(y_val * 0x10000).to_bytes(4, 'big', signed=True))
        cv2.circle(frame, (int(anchor[0] + x_val*0.20), int(anchor[1] + y_val*0.20)), 35, (xy_color, xy_color, xy_color), 7, cv2.LINE_AA)

    def draw_right_stick(self, frame):
        anchor = (360, 305)
        cv2.circle(frame, (int(anchor[0]), int(anchor[1])), 45, (255, 255, 255), 2, cv2.LINE_AA)
        x_val = get_actual(STICK_1_X)
        y_val = get_actual(STICK_1_Y)
        click = get_actual(BUTTON_6)
        click_color = int(click * 2.55)
        self.gcvdata[9] = int(click)
        cv2.circle(frame, (int(anchor[0] + x_val*0.20), int(anchor[1] + y_val*0.20)), 37, (click_color, click_color, click_color), -1, cv2.LINE_AA)
        xy_color = int((abs(x_val) + abs(y_val)) * 2.25) + 30
        self.gcvdata.extend(int(x_val * 0x10000).to_bytes(4, 'big', signed=True))
        self.gcvdata.extend(int(y_val * 0x10000).to_bytes(4, 'big', signed=True))
        cv2.circle(frame, (int(anchor[0] + x_val*0.20), int(anchor[1] + y_val*0.20)), 35, (xy_color, xy_color, xy_color), 7, cv2.LINE_AA)