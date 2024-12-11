import os

# 定义文件夹路径和目标文件路径
folder_path = "/home/tsinghuaair/xwj/motionfm-main/dataset/KIT-ML/texts"  # 替换为包含 .txt 文件的文件夹路径
output_file = "./assets/merged_output.txt"   # 合并后的文件名


# 打开目标文件以写入模式
with open(output_file, "w", encoding="utf-8") as outfile:
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 确保只处理 .txt 文件
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            
            # 逐行读取当前文件内容
            with open(file_path, "r", encoding="utf-8") as infile:
                for line in infile:
                    # 去掉 "#" 及其后内容
                    processed_line = line.split("#")[0].strip()
                    if processed_line:  # 确保非空行才写入
                        # 如果句子没有句号，自动添加句号
                        if not processed_line.endswith("."):
                            processed_line += "."
                        outfile.write(processed_line + "\n")
                        
print(f"合并完成，结果保存到: {output_file}")
