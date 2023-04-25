def main():
    filename = "data.txt" 
    separator = "\t"

    # 读取文件并分离标签和文本
    with open(filename, "r") as file:
        data = [line.strip().split(separator) for line in file]

    # 将数据转换为 (label, text) 元组列表
    labeled_data = [(label, text) for label, text in data]

if __name__ == '__main__':
    main()