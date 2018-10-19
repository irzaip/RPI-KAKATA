# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 11:14:17 2018

@author: Admin
"""

import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './gspeech.json'

import mikrophon

rets = ""

while (rets != "keluar"):
    rets = mikrophon.mendengarkan(thres=800,waittime=2)
    mikrophon.katakan(rets)

