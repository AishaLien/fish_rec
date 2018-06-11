#rename
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os, sys
count = 0
new_count = 0;
total_picture = 2000
# 列出目录
#print "目录为: %s"%os.listdir(os.getcwd())

while count < total_picture :
    try:
        # 重命名
        os.rename('U:/git/fish_Recognition/dataset/test/White sturgeon/'+str(count)+'.jpg','U:/git/fish_Recognition/dataset/test/White sturgeon/'+str(new_count)+'.jpg')
        print ('重命名成功。')
        new_count = new_count+1;
    except:
        pass
    count = count+1;
    