import os

for path, subdirs, files in os.walk("/home/geesara/dataset/data"):
    for name in files:
        if(name.endswith(('.jpg', '.png'))):
            print os.path.join(path, name)