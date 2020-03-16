# The structure of this file is inspired by: https://github.com/baidu-research/ba-dls-deepspeech

"""
Defines two dictionaries for converting 
between text and integer sequences.
"""
 
char_map_str = """
' 0
<SPACE> 1
а 2
б 3
в 4
г 5
д 6
ѓ 7
е 8
ж 9
з 10
ѕ 11
и 12
ј 13
к 14
л 15
љ 16
м 17
н 18
њ 19
о 20
п 21
р 22
с 23
т 24
ќ 25
у 26
ф 27
х 28
ц 29
ч 30
џ 31
ш 32
"""

char_map = {}
index_map = {}

for line in char_map_str.strip().split('\n'):
    ch, index = line.split()
    char_map[ch] = int(index)
    index_map[int(index)+1] = ch	
index_map[2] = ' '