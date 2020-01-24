# -*- coding: utf-8 -*- 
import os
import sys
import numpy as np
import pandas as pd


class TestClass():
    def __init__(self):
        print('init したよ')
        self.aa = 12

    def __getitem__(self, item):
        print(item)


def main():
    a = TestClass()
    print(a.aa)
    a['22']

if __name__ == '__main__':
    main()