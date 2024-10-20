#第一步
#把某种类型的图片改为.jpg格式
# import os
# import string
# dirName = "/home/dlut/网络/make_database/11.直升机巡视_数据集/VOCdevkit/VOC2018/JPEGImages/" #这里改成自己的图片所在路径
# li=os.listdir(dirName)
# for filename in li:
#     newname = filename
#     newname = newname.split(".")
#     if newname[-1]=="JPG":  #这里是你图片的原格式的后缀
#         newname[-1]="jpg"
#         newname = str.join(".",newname)  #这里要用str.join
#         filename = dirName+filename
#         newname = dirName+newname
#         os.rename(filename,newname)
#         print(newname,"updated successfully")

#*****************************************************************************************
import os

img_path = "E:/标定/number/guard"
def rename():
    i = 12372
    for item in os.listdir(img_path):
        print("the raw picture name:", item)
        old_file = os.path.join(img_path, item)
        new_file = os.path.join(img_path, (str(i).zfill(5) + '.jpg'))
        # new_file = os.path.join(img_path, (str(i).zfill(5)))
        os.rename(old_file, new_file)
        i += 1

if __name__ == '__main__':
    rename()
    print("Naming completed!")

#******************************************************************************************