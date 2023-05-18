import nibabel as nib
import os
import pandas as pd
import csv
from tqdm import tqdm
from datetime import datetime
import json

with open('../config.json', 'r', encoding='utf-8') as json_f:
    config = json.load(json_f)
    left_right_niis_path = config['nii_path_left_right_niis']
    cropped_left_right_path = config['nii_path_cropped_left_right']
    nii_path = config['nii_path_cropped_left_right']
    patient_path = config['nii_path_patients']
    target_path = config['target_CSV_PATH']


# ：切成两半
def split_nii_files(filename, save_dir):
    # split nii files into left and right halves along the transverse plane and save them in a given directory
    # input is a text file containing the paths of nii files and the name of the save directory
    files = open_file(filename)
    niis_left_right = os.listdir(save_dir)
    for file in tqdm(files):
        # for file in files:
        print('切块处理文件：{}中………………'.format(file))
        # name
        left_name = os.path.join(save_dir, os.path.basename(file)[:-4] + '_left.nii')
        right_name = os.path.join(save_dir, os.path.basename(file)[:-4] + '_right.nii')
        name_temp = os.path.basename(file)[:-4] + '_right.nii'
        if name_temp in niis_left_right:
            print('文件存在，跳过。')
            continue
        else:
            print('文件不存在，生成中…………')
            # img
            img = nib.load(file)
            data = img.get_fdata()
            shape = data.shape
            left = data[:shape[0] // 2, :, :]
            right = data[shape[0] // 2:, :, :]
            left_img = nib.Nifti1Image(left, img.affine, img.header)
            right_img = nib.Nifti1Image(right, img.affine, img.header)

            nib.save(left_img, left_name)
            nib.save(right_img, right_name)
            print('===============切块处理完成该文件================！')
    print('********************切块全部处理完成********************！')


# 切成两半之后，再切块，要求符合同样大小。
# 找中心，然后上下左右各找五十，就是一个100*100*100的方块
# https://blog.csdn.net/HuaCode/article/details/89222573
def crop_left_right_niis(niis_path, target_path):
    niis = os.listdir(niis_path)
    for nii_path in tqdm(niis):
        if nii_path.endswith(".nii"):
            # print('当前处理：{}'.format(nii_path))
            nii = nib.load(os.path.join(niis_path, nii_path))
            nii_data = nii.get_fdata()
            nii_affine = nii.affine
            nii_shape = nii_data.shape
            name_splited = nii_path.split('_')
            id_patient = int(name_splited[0])
            seq = name_splited[4]
            print("编号：{},序列：{},nii_shape={}".format(id_patient, seq, nii_shape))

            center = [int(x / 2) for x in nii_shape]
            cropped_nii_data = nii_data[center[0] - 90:center[0] + 90, center[1] - 100:center[1] + 100,
                               center[2] - 50:center[2] + 50]
            print(cropped_nii_data.shape)
            cropped_nii = nib.Nifti1Image(cropped_nii_data, nii_affine)
            nib.save(cropped_nii, os.path.join(target_path, nii_path[:-4] + '_cropped.nii'))


# 打开行存储文件，并读入每行的地址
def open_file(filename):
    with open(filename) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like \n at the end of each line
    content = [x.strip() for x in content]
    return content


# 1：保存到txt文件
def save_nii_files(dir_name, txt_name):
    # save the paths of nii files in a given directory to a txt file
    # input is the name of the directory and the name of the txt file
    nii_files = []
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            print('处理文件：{}中………………'.format(file))
            if file.endswith(".nii"):
                nii_files.append(os.path.join(root, file))
            with open(txt_name, "w") as f:
                for file in nii_files:
                    f.write(file + "\n")
            print('===============处理完成该文件================！')
    print('********************全部处理完成********************！')


def save_file_paths(dir_name, csv_name):
    # save the paths of files in a given directory to the first column of a csv file with header "PATH_img"
    # input is the name of the directory and the name of the csv file
    file_paths = []
    for file in os.listdir(dir_name):
        if file.endswith(".nii"):
            file_paths.append(os.path.join(dir_name, file))
            df = pd.DataFrame(file_paths, columns=["PATH_img"])
            df.to_csv(csv_name, index=False)


def read_csv_file(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = [row for row in reader]
    return data


def compare_time(time1, time2):
    date_str_1 = "2001.2.8"
    date_str_2 = "20010208"
    print(time1)
    print(time2)
    date_1 = datetime.strptime(time1, "%Y.%m.%d")
    date_2 = datetime.strptime(time2, "%Y%m%d")

    if date_1 == date_2:
        print("两个日期相同")
        return True
    else:
        print("两个日期不同")
        return False


def get_EXPLAIN_LABEL(half, row):
    explain = ''
    label = ''
    if half == 'left' and pd.notnull(row['左-易损性']):
        explain = str(row['左-影像描述']) + 'AHA分型：' + str(row['左-分型'])
        label = row['左-易损性']
    elif half == 'right' and pd.notnull(row['右-易损性']):
        explain = str(row['右-影像描述']) + 'AHA分型：' + str(row['右-分型'])
        label = row['右-易损性']
    else:
        print('读取影像描述与易损性出错')
    return explain, label


# 从patient文件中读取数据，从niis路径中读取文件名与地址，最后输出到target文件中
def generate_csv_from_halfniis_path(niis_path, patients_path, target_csv):
    with open(target_csv, mode='w', newline='') as target:
        writer = csv.writer(target)
        writer.writerow(['PATH_img', 'EXPLAIN', 'LABEL'])

        niis = os.listdir(niis_path)
        patients = pd.read_csv(patients_path, encoding="utf-8", index_col=0, header=1)
        print("patients:{}".format(patients))

        # https://blog.csdn.net/yyhhlancelot/article/details/82257985
        # 获取索引列
        ids_list = patients.index.to_list()
        print(ids_list)
        print(type(ids_list[0]))

        for nii in tqdm(niis):
            if nii.endswith(".nii"):
                PATH_img = os.path.join(niis_path, nii)
                EXPLAIN = ''
                LABEL = ''
                name_splited = nii.split('_')
                id_patient = int(name_splited[0])
                scan_time_from_nii = str(name_splited[len(name_splited) - 4])[:-6]

                print("id_patient:{}处理中…………".format(id_patient))
                half_patient = name_splited[len(name_splited) - 2]  # left or right

                # http://t.csdn.cn/N6fHE
                row = patients.loc[id_patient]

                if row.empty:
                    print("找不到该行:{}".format(id_patient))
                    continue
                elif isinstance(row, pd.DataFrame):  # 多行
                    for idx in range(len(row)):
                        row_line = row.iloc[idx]
                        scan_time = row_line['入组日期']
                        if compare_time(scan_time, scan_time_from_nii):
                            EXPLAIN, LABEL = get_EXPLAIN_LABEL(half_patient, row.iloc[idx])
                            print('是这个时间：{}和{}'.format(scan_time, scan_time_from_nii))
                        else:
                            print('！！！！不是这个时间：{}和{}'.format(scan_time, scan_time_from_nii))
                            continue
                elif isinstance(row, pd.Series):  # 单行
                    EXPLAIN, LABEL = get_EXPLAIN_LABEL(half_patient, row)

                # print("当前label：{}".format(LABEL))
                if LABEL != '' and LABEL != ' ':
                    writer.writerow([PATH_img, EXPLAIN, LABEL])
                    print('已保存：label={}'.format(LABEL))


if __name__ == '__main__':
    print(1)

    # save_nii_files("C:/Users/80798/Desktop/test", 'C:/Users/80798/Desktop/test/nii_paths.txt')
    # split_nii_files("C:/Users/80798/Desktop/test/nii_paths.txt", "C:/Users/80798/Desktop/test/left_and_right_niis")
    # save_file_paths("C:/Users/80798/Desktop/test/left_and_right_niis",
    #                 "C:/Users/80798/Desktop/test/left_and_right_niis/paths.csv")

    # path_unprocessed = 'C:/Users/80798/Desktop/test/left_and_right_niis/paths_utf8.csv'
    # path_processed = 'C:/Users/80798/Desktop/test/left_and_right_niis/paths.pkl'
    # resize_before_split_val(path_unprocessed, path_processed)

    # print(split_train_val("C:/Users/80798/Desktop/test/left_and_right_niis/paths.csv"))
    # print('完成')
    # generate_csv_from_halfniis_path("F:/test",
    #                                 "E:/Documents/PostGraduate/replay/VKD/vkd_in_mr-master/Dataset/颈动脉信息_CSV.csv",
    #                                 "E:/Documents/PostGraduate/replay/VKD/vkd_in_mr-master/Dataset/target.csv")
    # save_nii_files(r'F:\颈动脉\整理后160_nii\NIIS', r'F:\颈动脉\整理后160_nii\NIIS.txt')
    # split_nii_files(r'F:\颈动脉\整理后160_nii\NIIS.txt', r'F:\颈动脉\整理后160_nii\NIIS_left_right')
    # generate_csv_from_halfniis_path('F:/颈动脉/整理后160_nii/NIIS_left_right',
    #                                 "E:/Documents/PostGraduate/replay/VKD/vkd_in_mr-master/Dataset/【颈动脉】v3.0 CSV.csv",
    #                                 "E:/Documents/PostGraduate/replay/VKD/vkd_in_mr-master/Dataset/target.csv")
    # crop_left_right_niis(left_right_niis_path, cropped_left_right_path)
    generate_csv_from_halfniis_path(nii_path, patient_path, target_path)
