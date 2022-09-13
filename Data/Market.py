import os
from shutil import copyfile


def prepare_train_test(input_dir, output_dir):
    os.makedirs(output_dir)
    for root, dirs, files in os.walk(input_dir, topdown=True):
        for name in files:
            if not name[-3:] == 'jpg':
                continue
            ID = name.split('_')
            src_path = input_dir + '/' + name
            dst_path = output_dir + '/' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)