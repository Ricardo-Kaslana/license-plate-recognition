# CCPD字符集 + 特殊标记
# <s>: start of sequence
# </s>: end of sequence
# <pad>: padding

# CCPD字符集 + 特殊标记
CHARSET = "皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
SPECIAL_TOKENS_DICT = {"<s>": 0, "</s>": 1, "<pad>": 2}
CHARSET_SIZE = len(CHARSET)
VOCAB_SIZE = CHARSET_SIZE + len(SPECIAL_TOKENS_DICT)

# 创建 char -> index 和 index -> char 的映射
char_to_idx = {char: i + len(SPECIAL_TOKENS_DICT) for i, char in enumerate(CHARSET)}
char_to_idx.update(SPECIAL_TOKENS_DICT)

idx_to_char = {i: char for char, i in char_to_idx.items()}

# 车牌最大长度（例如，新能源车牌是8位）+ 2个特殊token(<s>, </s>)
# 我们设置为8位，因为新能源车牌最长。
MAX_PLATE_LEN = 8
MAX_SEQ_LEN = MAX_PLATE_LEN + 2 # 加上 <s> 和 </s>

# 用于CCPD文件名解析的省份映射
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "挂"]
ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'I', 'O']
ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'I', 'O']

def parse_plate_number(label_indices):
    """从CCPD文件名中的索引解析出车牌字符串"""
    label_str = ""
    # 第一个字符是省份
    label_str += PROVINCES[label_indices[0]]
    # 第二个字符是字母
    label_str += ALPHABETS[label_indices[1]]
    # 后续是字母或数字
    for i in range(2, len(label_indices)):
        label_str += ADS[label_indices[i]]
    return label_str

def convert_video_for_web(input_path):
    """使用FFmpeg转换视频为浏览器兼容格式"""
    import subprocess
    output_path = input_path.replace('.mp4', '_web.mp4')
    
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-c:v', 'libx264',
        '-profile:v', 'main',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-crf', '23',
        '-preset', 'fast',
        '-c:a', 'aac',
        '-b:a', '128k',
        output_path
    ]
    
    subprocess.run(cmd, check=True)
    return output_path