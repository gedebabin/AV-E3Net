# import os
# import shutil

# PATH = '/data/LRS3_30h/'

# with open(PATH + 'valid_avgen.tsv', 'r') as f:
#     lines = f.readlines()[1:]
#     print(len(lines))
#     print(lines[1])

#     for line in lines:
#         splitted = line.split("\t")

#         id = splitted[0].split('/')[1]

#         video_src = PATH + 'trainval/video/' + splitted[1].split('trainval/')[1]
#         clean_src = PATH + 'trainval/clean/' + splitted[2].split('trainval/')[1]
#         noisy_src = PATH + 'trainval/noisy/' + splitted[3].split('trainval/')[1]

#         video_dest = PATH + 'valid/video/' + splitted[1].split('trainval/')[1]
#         clean_dest = PATH + 'valid/clean/' + splitted[2].split('trainval/')[1]
#         noisy_dest = PATH + 'valid/noisy/' + splitted[3].split('trainval/')[1]

#         if not os.path.exists('/data/LRS3_30h/valid/video/' + id):
#             os.mkdir('/data/LRS3_30h/valid/video/' + id)
#             os.mkdir('/data/LRS3_30h/valid/clean/' + id)
#             os.mkdir('/data/LRS3_30h/valid/noisy/' + id)

#         # print(video_src, '->', video_dest)

#         shutil.move(video_src, video_dest)
#         shutil.move(clean_src, clean_dest)
#         shutil.move(noisy_src, noisy_dest)

#     # print(os.listdir(PATH + 'test/clean'))
