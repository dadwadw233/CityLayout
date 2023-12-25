import os
import argparse
import numpy as np
import tqdm
import shutil
def escape_path(path):
    # 定义需要转义的字符及其转义后的形式
    escape_chars = {
        " ": "\\ ",  # 空格
        "(": "\\(",  # 左括号
        ")": "\\)",  # 右括号
        "&": "\\&",  # 和号
        "'": "\\'",  # 单引号
        '"': '\\"',  # 双引号
        "!": "\\!",  # 感叹号
        "@": "\\@",  # At
        "#": "\\#",  # 井号
        "$": "\\$",  # 美元符
        "%": "\\%",  # 百分号
        "^": "\\^",  # 脱字符
        "*": "\\*",  # 星号
        "=": "\\=",  # 等号
        "+": "\\+",  # 加号
        "|": "\\|",  # 竖线
        "{": "\\{",  # 左花括号
        "}": "\\}",  # 右花括号
        "[": "\\[",  # 左中括号
        "]": "\\]",  # 右中括号
        "\\": "\\\\",  # 反斜杠
        ":": "\\:",  # 冒号
        ";": "\\;",  # 分号
        "<": "\\<",  # 小于号
        ">": "\\>",  # 大于号
        "?": "\\?",  # 问号
        ",": "\\,",  # 逗号
        ".": "\\.",  # 英文句号
        "`": "\\`",  # 重音符
        "~": "\\~",  # 波浪号
    }

    # 对每个需要转义的字符进行替换
    for char, escaped_char in escape_chars.items():
        path = path.replace(char, '-')

    return path
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=False,
        help="Input file",
        default="/home/admin/workspace/yuyuanhong/code/CityLayout/data/raw/osm/cities",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Output file",
        default="/home/admin/workspace/yuyuanhong/code/CityLayout/data",
    )

    args = parser.parse_args()

    raw_dir = args.raw_dir
    output = args.output

    if not os.path.exists(output):
        os.makedirs(output)

    output_train = os.path.join(output, "train_128")
    output_val = os.path.join(output, "val")
    output_test = os.path.join(output, "test")

    print("output_train: ", output_train)
    print("output_val: ", output_val)
    print("output_test: ", output_test)

    train_data_rate = 1
    val_data_rate = 0.
    test_data_rate = 0.

    print("train : val : test", train_data_rate, val_data_rate, test_data_rate)

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

    print("train_data_num: ", len(train_data_list))
    print("val_data_num: ", len(val_data_list))
    print("test_data_num: ", len(test_data_list))

    if not os.path.exists(output_train):
        os.makedirs(output_train)

    if not os.path.exists(output_val):
        os.makedirs(output_val)

    if not os.path.exists(output_test):
        os.makedirs(output_test)

    for city in tqdm.tqdm(os.listdir(raw_dir), desc="Split data"):
        city_dir = os.path.join(raw_dir, city)
        if not os.path.isdir(city_dir):
            continue
        data_file = os.path.join(city_dir, escape_path(city) + ".h5")
        if not os.path.exists(data_file):
            continue

        if city in train_data_list:
            shutil.move(data_file, output_train)
        elif city in val_data_list:
            shutil.move(data_file, output_val)
        elif city in test_data_list:
            shutil.move(data_file, output_test)
