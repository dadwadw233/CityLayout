import os
import argparse
import numpy as np
import tqdm
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, required=False, help='Input file', default='/home/admin/workspace/yuyuanhong/code/CityLayout/data/raw/osm/cities')
    parser.add_argument('--output', type=str, required=False, help='Output file', default='/home/admin/workspace/yuyuanhong/code/CityLayout/data')

    args = parser.parse_args()

    raw_dir = args.raw_dir
    output = args.output

    if not os.path.exists(output):
        os.makedirs(output)

    output_train = os.path.join(output, 'train')
    output_val = os.path.join(output, 'val')
    output_test = os.path.join(output, 'test')

    print('output_train: ', output_train)
    print('output_val: ', output_val)
    print('output_test: ', output_test)

    train_data_rate = 0.7
    val_data_rate = 0.2
    test_data_rate = 0.1

    print('train : val : test', train_data_rate, val_data_rate, test_data_rate)

    train_data_list = []
    val_data_list = []
    test_data_list = []
    sum_list = os.listdir(raw_dir)

    train_data_num = int(len(sum_list) * train_data_rate)
    val_data_num = int(len(sum_list) * val_data_rate)
    test_data_num = int(len(sum_list) * test_data_rate)

    


    # random select train data
    for i in range(train_data_num):
        index = np.random.randint(0, len(sum_list))
        train_data_list.append(sum_list[index])
        sum_list.pop(index)

    # random select val data
    for i in range(val_data_num):
        index = np.random.randint(0, len(sum_list))
        val_data_list.append(sum_list[index])
        sum_list.pop(index)
    
    # test data is the rest
    test_data_list = sum_list

    print('train_data_num: ', len(train_data_list))
    print('val_data_num: ', len(val_data_list))
    print('test_data_num: ', len(test_data_list))


    if not os.path.exists(output_train):
        os.makedirs(output_train)
    
    if not os.path.exists(output_val):
        os.makedirs(output_val)
    
    if not os.path.exists(output_test):
        os.makedirs(output_test)


    for city in tqdm.tqdm(os.listdir(raw_dir), desc='Split data'):
        city_dir = os.path.join(raw_dir, city)
        if not os.path.isdir(city_dir):
            continue
        data_file = os.path.join(city_dir, city + '.h5')
        if not os.path.exists(data_file):
            continue

        if city in train_data_list:
            os.system('mv {} {}'.format(data_file, output_train))
        elif city in val_data_list:
            os.system('mv {} {}'.format(data_file, output_val))
        elif city in test_data_list:
            os.system('mv {} {}'.format(data_file, output_test))
