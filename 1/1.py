# 更新脚本以正确识别和提取所有以 "hsa-" 开头的人类miRNA

def filter_human_miRNA_corrected(fa_file, output_file):
    import re

    # 打开原始文件和输出文件
    with open(fa_file, 'r') as file, open(output_file, 'w') as out_file:
        count_matched = 0
        count_processed = 0

        for line in file:
            count_processed += 1
            parts = line.strip().split(',')
            if len(parts) >= 3:
                miRNA_id = parts[0].strip()
                sequence = parts[2].strip()

                # 检查是否为人类miRNA，确保前导空格或其他字符不影响匹配
                if re.match(r'^>\s*hsa-', miRNA_id):
                    # 写入miRNA ID
                    out_file.write(f"{miRNA_id}\n")
                    # 写入对应的序列
                    out_file.write(f"{sequence}\n")
                    count_matched += 1

    print(f"Total lines processed: {count_processed}")
    print(f"Total human miRNA matched: {count_matched}")

# # 调用函数进行处理，重新运行脚本以提取所有以 "hsa-" 开头的条目
# output_fa_corrected = 'human.fa'
# input_fa='mirna_new.fa'
# filter_human_miRNA_corrected(input_fa, output_fa_corrected)


def process_and_clean_fasta(file_path, output_path):
    with open(file_path, 'r') as file, open(output_path, 'w') as output:
        for line in file:
            # 替换逗号后跟空格为单个逗号来统一分隔符
            line = line.strip().replace(', ', ',')
            elements = line.split(',')  # 根据逗号分割每行
            if len(elements) > 6:  # 确保行中元素足够
                mrna_sequence = elements[-2]  # 获取倒数第二个元素，即mrna序列
                # 删除原行中的mrna序列
                del elements[-2]
                clean_line = ','.join(elements)  # 重新组合行，元素之间只用逗号分隔
                output.write(clean_line + '\n')  # 写入处理后的当前行
                output.write(mrna_sequence + '\n')  # 在新的一行写入mrna序列

# 使用函数，输入文件路径和输出文件路径
# process_and_clean_fasta('updated_new.fa', '1111.fa')

# 使用函数，输入文件路径和输出文件路径
#反转mrna序列
def reverse_mrna_sequences(file_path, output_path):
    with open(file_path, 'r') as file, open(output_path, 'w') as output:
        for line_number, line in enumerate(file, start=1):
            elements = line.strip().split(',')
            if len(elements) >= 6:
                mrna_sequence = elements[-2]
                reversed_mrna = mrna_sequence[::-1]
                elements[-2] = reversed_mrna
                new_line = ','.join(elements)
                output.write(new_line + '\n')
            else:
                # 打印警告或记录到日志文件
                print(f"Warning: Line {line_number} skipped due to insufficient elements.")
                # 可以选择在这里记录到日志文件或者其他形式的错误处理

# # 调用函数，输入原始文件路径和输出文件路径
reverse_mrna_sequences('val.fa', 'val1.fa')


def filter_fa_by_csv_id(csv_file_path, fa_file_path, output_fa_path):
    # Step 1: Load miRNA IDs from the CSV file
    miRNA_ids = set()
    with open(csv_file_path, 'r') as csv_file:
        for line in csv_file:
            miRNA_id = line.strip().split(',')[0]  # Assuming the ID is in the first column
            miRNA_ids.add(miRNA_id)

    # Step 2: Filter and write matching IDs and sequences from the FA file
    with open(fa_file_path, 'r') as fa_file, open(output_fa_path, 'w') as output_fa:
        for line in fa_file:
            parts = line.strip().split(',')
            if len(parts) > 1 and parts[1] in miRNA_ids:  # Assuming the ID is in the second column
                output_fa.write('>'+parts[1] + '\n')  # Write the ID
                output_fa.write(parts[-1] + '\n')  # Write the sequence, assuming it's in the last column

# # Usage of the function
# csv_file_path = 'all_M.csv'
# fa_file_path = 'mRNA.fa'
# output_fa_path = 'm.fa'
#
# filter_fa_by_csv_id(csv_file_path, fa_file_path, output_fa_path)

