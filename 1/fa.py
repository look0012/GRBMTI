import re

def format_fasta(input_file, output_file):#用于将reslut.fa中的mrnaID和序列提取出来
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        sequence_buffer = []  # 用于存储序列的缓冲区
        for line in infile:
            if line.startswith('>'):
                if sequence_buffer:  # 如果缓冲区非空，先将之前的序列写入文件
                    # 写入序列数据，紧跟在头部之后，用逗号隔开
                    outfile.write(',' + ''.join(sequence_buffer) + '\n')
                    sequence_buffer = []  # 清空缓冲区
                # 使用正则表达式提取 NM_XXXX.X 格式的 ID 和括号内的基因名
                match = re.match(r'>([^ ]*) .*?\(([^)]+)\)', line)
                if match:
                    nm_id, gene_name = match.groups()
                    current_header = f'>{nm_id},{gene_name}'
                else:
                    current_header = '>Unknown'  # 如果没有匹配到，使用默认值
                outfile.write(current_header)  # 写入头部，不换行
            else:
                # 添加当前行的序列到缓冲区，移除可能的换行符
                sequence_buffer.append(line.strip())
        # 确保最后一个序列也被写入文件
        if sequence_buffer:
            outfile.write(',' + ''.join(sequence_buffer) + '\n')

# 调用函数处理文件
# format_fasta('reslut.fa', 'output.fa')

import pandas as pd

def read_fasta(fasta_file):
    """读取 .fa 文件并返回 DataFrame，包含 miRNA ID, MIMA ID, 序列"""
    data = []
    with open(fasta_file, 'r') as file:
        for line in file:
            if line.startswith('>'):
                # 去除开头的 '>'，然后以逗号分隔整行
                header_parts = line[1:].strip().split(',')
                if len(header_parts) < 3:
                    print(f"Warning: Malformed line skipped: {line.strip()}")
                    continue  # 跳过格式错误的行
                miRNA_ID = header_parts[0].strip()
                MIMA_ID = header_parts[1].strip()
                sequence = header_parts[2].strip()  # 序列紧跟在最后
                data.append([miRNA_ID, MIMA_ID, sequence])
    return pd.DataFrame(data, columns=['miRNA_ID', 'MIMA_ID', 'sequence'])

def find_and_write_sequences(mirna_df, inter_df, fasta_file, output_file, append_mode,label):
    # 读取 mRNA 序列
    gene_to_sequence = {}
    with open(fasta_file, 'r') as fasta:
        for line in fasta:
            if line.startswith('>'):
                parts = line.strip().split(',')
                if len(parts) < 3:
                    continue
                nm_id = parts[0][1:].strip()
                gene_name = parts[1].strip()
                mrna_sequence = parts[2].strip()
                if gene_name not in gene_to_sequence:
                    gene_to_sequence[gene_name] = (nm_id, mrna_sequence)

    # 写入新的文件
    with open(output_file, 'a' if append_mode else 'w') as outfile:
        for index, row in inter_df.iterrows():
            mirna_id = row.iloc[0]  # 假设第一列为 miRNA ID
            gene_name = row.iloc[1]  # 假设第二列为 mRNA ID
            if gene_name in gene_to_sequence:
                nm_id, mrna_sequence = gene_to_sequence[gene_name]
                # 检查 mirna_id 是否存在于 mirna_df 中
                filtered_df = mirna_df[mirna_df['miRNA_ID'] == mirna_id]
                if not filtered_df.empty:
                    mirna_info = filtered_df.iloc[0]
                    mima_id = mirna_info['MIMA_ID']
                    mirna_sequence = mirna_info['sequence']
                    # 写入新的行，包括标签 '1'
                    outfile.write(f">{mirna_id}, {mima_id}, {gene_name}, {nm_id}, {mirna_sequence}, {mrna_sequence},{label}\n")
                else:
                    print(f"Warning: No sequence found for miRNA {mirna_id}")
            else:
                print(f"Warning: No sequence found for mRNA {gene_name}")

# # 读取文件
mirna_df = read_fasta('mirna_new.fa')

inter_df = pd.read_csv('NegativeSample_Validation.csv', header=None)  # 如果没有列名，可以使用 header=None
negative=pd.read_csv('NegativeSample_Validation.csv', header=None)
# 处理并写入数据，指定覆盖模式
find_and_write_sequences(mirna_df, inter_df, 'mRNA.fa', 'val.fa', append_mode=False,label=1)
find_and_write_sequences(mirna_df, inter_df, 'mRNA.fa', 'val.fa', append_mode=True,label=0)





def format_fasta(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        content = infile.readlines()
        i = 0
        while i < len(content):
            if content[i].startswith('>'):
                # 解析行
                header = content[i].strip()
                parts = header.split()
                mirna_id = parts[0][1:]  # 移除前面的 '>'
                mima_id = parts[1]
                sequence = content[i + 1].strip()  # 序列在下一行
                # 写入新的格式
                outfile.write(f">{mirna_id},{mima_id},{sequence}\n")
                i += 2  # 移动到下一个头部行
            else:
                i += 1  # 正常情况下不会执行到这里

# 调用函数
# input_file_path = 'mirna.fa'
# output_file_path = 'mirna_new.fa'
# format_fasta(input_file_path, output_file_path)


import csv
#将关系对从fa文件中提出来，并保存到csv文件
def extract_ids_to_csv(input_file, output_file):
    # 我们需要提取的信息的列数，假设是前四列
    max_columns = 4
    data = []

    try:
        with open(input_file, 'r') as file:
            for line in file:
                if line.startswith('>'):  # 确认是数据行
                    # 移除字符串首尾的空白, 移除开头的'>', 并按逗号分割
                    parts = line.strip()[1:].split(',')
                    # 只提取每列的第一部分
                    if len(parts) >= max_columns:  # 确保有足够的数据
                        # 提取前四个数据项，每个项目取第一部分
                        selected_parts = [parts[i].strip().split()[0] for i in range(max_columns)]
                        data.append(selected_parts)

        # 将结果写入CSV文件
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in data:
                writer.writerow(row)

    except Exception as e:
        print("Error:", e)
#
# # 调用函数
# input_path = 'updated_new.fa'  # 替换为你的实际文件路径
# output_path = 'output.csv'  # 输出CSV文件的路径
# extract_ids_to_csv(input_path, output_path)

#除去空格，只用逗号分割
def clean_fa_file(file_path, output_path):
    with open(file_path, 'r') as file, open(output_path, 'w') as output:
        for line in file:
            # 移除行尾的空白字符，然后替换所有的逗号后跟空格为单个逗号
            cleaned_line = line.strip().replace(', ', ',')
            output.write(cleaned_line + '\n')  # 写入新的文件

# 使用函数，输入文件路径和输出文件路径
# clean_fa_file('val.fa', 'val1.fa')


