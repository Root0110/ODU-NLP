#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/20 15:27
# @Author  : Xiaohan Wang
# @Site    : 
# @File    : divide_into_single_file.py
# @Software: PyCharm


def intoSingle(dir,domain):
    read_loc = dir + '/' + domain + '/combined_{}.txt'.format(domain.lower())
    write_dir = dir + '/' + domain
    with open(read_loc, 'r') as f1:
        temp =[]
        no = 1
        lines = f1.readlines()
        for i in range(len(lines)):
            if 'DOCSTART' in lines[i] or i==len(lines)-1:
                if temp:
                    write_loc = write_dir + '/file{}.txt'.format(no)
                    with open(write_loc, 'w') as f2:
                        f2.writelines(temp)
                    temp = []
                    no += 1
            temp.append(lines[i])


if __name__ == '__main__':
    work_dir = '/Users/W/PycharmProjects/odu-NLP/OA-STM-domains'
    domains = ['Arg', 'Astr', 'Bio', 'Chem', 'CS', 'Eng', 'ES', 'Math', 'Med', 'MS', 'Overall']
    for d in domains:
        intoSingle(work_dir, d)

