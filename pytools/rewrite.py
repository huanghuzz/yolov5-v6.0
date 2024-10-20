#先修改路径，将xml文件对应图片的真实路径替换。这里图片的名称是采用12位数字排序的

import xml.dom.minidom
import os

# path = r'D:\RM-DATASET\RM-DATASET\RM-ARMOR-VOC2007\VOC2007\VOC2017\val1'  # xml文件存放路径
# sv_path = r'D:\RM-DATASET\RM-DATASET\RM-ARMOR-VOC2007\VOC2007\VOC2017\val'  # 修改后的xml文件存放路径
# files = os.listdir(path)
# cnt = 0
#
# for xmlFile in files:
#     dom = xml.dom.minidom.parse(os.path.join(path, xmlFile))  # 打开xml文件，送到dom解析
#     root = dom.documentElement  # 得到文档元素对象
#     item = root.getElementsByTagName('path')  # 获取path这一node名字及相关属性值
#     for i in item:
#         i.firstChild.data = f'D:/RM-DATASET/RM-DATASET/RM-ARMOR-VOC2007/VOC2007/VOC2017/train/' + str(cnt).zfill(5) + '.jpg'  # xml文件对应的图片路径
#
#     with open(os.path.join(sv_path, xmlFile), 'w', encoding='utf-8') as fh:
#         dom.writexml(fh)
#     cnt += 1

#修改图片名称

import xml.dom.minidom
import os

# path = r'D:\test\xmltest\xml_source'  # xml文件存放路径
# sv_path = r'D:\test\xmltest\xml_save'  # 修改后的xml文件存放路径
path = r'C:\Users\PC\Desktop\yolov5-v6.0\VOC\Annotations'  # xml文件存放路径
sv_path = r'C:\Users\PC\Desktop\yolov5-v6.0\VOC\labels'  # 修改后的xml文件存放路径
files = os.listdir(path)

for xmlFile in files:
    dom = xml.dom.minidom.parse(os.path.join(path, xmlFile))  # 打开xml文件，送到dom解析
    root = dom.documentElement  # 得到文档元素对象
    names = root.getElementsByTagName('filename')
    a, b = os.path.splitext(xmlFile)  # 分离出文件名a
    for n in names:
        n.firstChild.data = a + '.jpg'
    with open(os.path.join(sv_path, xmlFile), 'w', encoding='utf-8') as fh:
        dom.writexml(fh)

#*****************************************************************************

# -*- coding:utf-8 -*-

# 将a替换成b

# import os
#
# xmldir = r'D:\test\xmltest\xml_source'
# savedir = r'D:\test\xmltest\xml_save'
# xmllist = os.listdir(xmldir)
# for xml in xmllist:
#     if '.xml' in xml:
#         fo = open(savedir + '/' + '{}'.format(xml), 'w', encoding='utf-8')
#         print('{}'.format(xml))
#         fi = open(xmldir + '/' + '{}'.format(xml), 'r', encoding='utf-8')
#         content = fi.readlines()
#         for line in content:
#             # line = line.replace('a', 'b')        # 例：将a替换为b
#             line = line.replace('<?xml version="1.0" ?>', '')
#             line = line.replace('<folder>测试图片</folder>', '<folder>车辆图片</folder>')
#             line = line.replace('<name>class1</name>', '<name>class2</name>')
#             fo.write(line)
#         fo.close()
#         print('替换成功')

# 如通b为空字符串，就是删除

#************************************************************************************