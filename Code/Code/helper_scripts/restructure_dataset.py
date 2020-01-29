# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 16:45:59 2019

@author: Marija
"""
import os

data_dir = r'D:/DIPLOMSKA/Database/train/2'
os.chdir(data_dir)

for folder in os.listdir(data_dir):
    print(folder)
    if os.path.isdir(os.path.join(data_dir, folder)):
        for file in os.listdir(os.path.join(data_dir, folder)):
            if str(file).endswith('.txt'):
                os.remove(data_dir+os.sep+folder+os.sep+file)



for folder in os.listdir(data_dir):
    print(folder)
    if os.path.isdir(os.path.join(data_dir, folder)):
        for file in os.listdir(os.path.join(data_dir, folder)):
            if str(file).endswith('.txt'):
                print(file)
                os.remove(data_dir+os.sep+folder+os.sep+file)


idx = '102_2018'
path = data_dir + os.sep + idx
new_path = data_dir + os.sep + 'newfiles'

filename='4-000001.trans.txt' 
newfilename='7-000001.trans.txt' 


def confirm_consistent_naming_of_files():
    for folder in os.listdir(data_dir):
        print(folder)
        if os.path.isdir(os.path.join(data_dir, folder)):
            for subfolder in os.listdir(os.path.join(data_dir, folder)):
                if os.path.isdir(os.path.join(data_dir+os.sep+folder,subfolder)):
                    for file in os.listdir(os.path.join(data_dir+os.sep+folder,subfolder)):
                        #print(folder)
                        subdir = data_dir + os.sep + folder + os.sep + subfolder 
                        print(file)
                        filename = file.split('-')
                        filename = "-".join(filename[1:])
                        name = str(folder) + '-' + filename
                        print(name)
                        if(file != name):
                            os.rename(os.path.join(subdir,file),os.path.join(subdir,name))
            
            
def confirm_consistent_naming_inside_files():             
    for i in range(10,26):
        change_book_num=7
        print(i)
        filename='4-0000'+str(i)+'.trans.txt' 
        newfilename='7-0000'+str(i)+'.trans.txt'
        full_path = path + os.sep + filename
        with open(new_path+os.sep+newfilename, 'w', encoding='utf-16-le') as wp:
            with open(full_path, encoding='utf-16-le') as fp:
                print(full_path)
                line = fp.readline()
                while line:
                    line = fp.readline()
                    if line:
                        line=list(line)
                        #print(line)
                        #id_num = line.split(' ')[0]
                        #book_num = id_num.split('-')[0]
                        line[0] = str(change_book_num)
                        line = "".join(line)
                        wp.write(line)
                        
                      
data_dir = r'D:/DIPLOMSKA/Database/train/3'
os.chdir(data_dir)
change_book_num = 3

for folder in os.listdir(data_dir):
    print(folder)
    if os.path.isdir(os.path.join(data_dir, folder)):
        for file in os.listdir(os.path.join(data_dir, folder)):
            if str(file).endswith('.txt'):
                    new_path=data_dir+os.sep+file
                    old_path=data_dir+os.sep+folder+os.sep+file
                    with open(new_path, 'w', encoding='utf8') as wp:
                        with open(old_path, 'r', encoding='utf8') as fp:
                            print(new_path)
                            line = fp.readline()
                            while line:
                                if line:
                                    line=list(line)
                                    #print(line)
                                    #id_num = line.split(' ')[0]
                                    #book_num = id_num.split('-')[0]
                                    line[0] = str(change_book_num)
                                    line = "".join(line)
                                    wp.write(line)
                                    line = fp.readline()