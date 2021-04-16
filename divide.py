import os
import random
import shutil

source = 'data\\1728090'
dest = 'data\\data\\val\\1728090'
files = os.listdir(source)

for file_name in random.sample(files, 7):
    shutil.move(os.path.join(source, file_name), dest)