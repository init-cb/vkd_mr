# !/dicoms_2_niis python
# -*-coding:utf-8 -*-

"""
# File       : preprocess_Dicoms.py
# Time       ：23-5-15 16:44
# Author     ：Bo Cao
# version    ：python 3.9
# Description：
"""

import os
import sys
import shutil
import json

with open('config.json', 'r', encoding='utf-8') as json_f:
    config = json.load(json_f)
    RENAME_ADDRESS = 'F:\颈动脉\质量4用于图文\质量4用于图文'
    RENAME_TARGET = 'F:\颈动脉\整理后160'

    MRICROGL_path = config['MRICROGL_path']
    BASE_PATH = r'F:\颈动脉\整理后160'
    TARGET_PATH = r'F:\颈动脉\整理后160_nii'


# 用于规范化重命名dicom文件
def rename_dicoms(rename_address, rename_target):
    # address = 'D:\DatasOfBOBO\PostGraduate\IMAGE\\test\\5'
    # target = 'D:\DatasOfBOBO\PostGraduate\IMAGE\\test\\10'
    address = rename_address
    target = rename_target

    filenames = os.listdir(address)
    add_now = ''

    for file in filenames:
        str_file = str.split(file, '_')
        if str_file.__len__() < 3:
            continue
        else:
            file_0 = str_file[0]
            print(str_file)
            file_1 = str_file[2]

            new_name = file_0 + '_' + file_1
            print(new_name)
            if os.path.exists(target + '\\' + new_name):
                print('existed!' + new_name)
                continue
            else:
                add_now = address + '\\' + file
                file_sequences = os.listdir(add_now)
                os.makedirs(target + '\\' + new_name)
                for sequence in file_sequences:
                    shutil.move(address + '\\' + file + '\\' + sequence, target + '\\' + new_name)
                print('moved :' + new_name)


# 用于将规范化后的dicom文件转换为nii文件
def turn_dicoms_2_niis(MrcroGL_path, Base_Path, Target_path):
    sys.path.append(MrcroGL_path)
    # base_path = r'D:\DatasOfBOBO\PostGraduate\IMAGE\test\10'
    # target_path = r'D:\DatasOfBOBO\PostGraduate\replay\VKD\vkd_in_mr-master\Dataset'
    base_path = Base_Path
    target_path = Target_path
    i = 0
    for patient in os.listdir(base_path):
        print("\n==================第" + str(i) + "病人=========================")
        sequences_of_patient = base_path + '\\' + patient
        for sequence in os.listdir(sequences_of_patient):
            dist_dir = sequences_of_patient + '\\' + sequence
            new_name = dist_dir

            # UTF-8
            os.system('chcp 65001')
            os.system(
                r'E:\Programs\MRIcroGL\Resources\dcm2niix -e n -f %i_%n_%f_%p_%t_%s -o {0} {1}'.format(target_path,
                                                                                                       dist_dir))
        i += 1


if __name__ == '__main__':
    print("开始处理")
    # rename_dicoms(RENAME_ADDRESS,RENAME_TARGET)
    # turn_dicoms_2_niis(MRICROGL_path, BASE_PATH, TARGET_PATH)
