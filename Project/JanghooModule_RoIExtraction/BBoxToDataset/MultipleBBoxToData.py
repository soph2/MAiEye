import cv2
import os
import xml.etree.ElementTree as et
import argparse

'''
<?xml version="1.0"?>
<annontation>
    <folder>images</folder>
    <filename>10.jpg</filename>
    <size>
        <width>450</width>
        <height>328</height>
    </size>
    <object>
        <name>pig</name>
        <bndbox>
            <xmin>19</xmin>
            <ymin>84</ymin>
            <xmax>144</xmax>
            <ymax>236</ymax>
        </bndbox>
    </object>
</annontation>
'''

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--debuging", type=str, help = "디버그 모드에서는 임의의 파일로 실험해볼 수 있습니다.")
ap.add_argument("-f", "--filename", type=str, help="select file")
args = vars(ap.parse_args())


class FileData :

    def __init__(self, filefullpath, savefolder, debugmode = False):
        self.root = et.Element("annotation")
        self.folder = et.SubElement(self.root, "folder")
        self.filename = et.SubElement(self.root, "filname")
        self.size = et.SubElement(self.root, "size")
        self.obj = et.SubElement(self.root, "object")
        self.file_fullpath = filefullpath
        self.file_folderpath,   self.file_fullname = os.path.split(filefullpath)
        self.file_justfilename, self.file_justext  = os.path.splitext(self.file_fullname)
        self.file_savefolder = savefolder

        if debugmode == True :
            print('------------------OBJECT-------------------')
            print('full path  : ', self.file_fullpath)
            print('folderpath : ', self.file_folderpath)
            print('file name  : {}, (name : {}, ext : {})\n'.format(self.file_fullname, self.file_justfilename, self.file_justext))


    def writeAndSave(self) :
        file_name = str(self.file_justfilename) + '.xml'
        tree = et.ElementTree(self.root)
        tree.write(file_or_filename = os.path.join(self.file_savefolder, file_name))




if True :

    file_name = args['filename']
    current_dir = os.getcwd()
    file_full_path = os.path.join(current_dir, file_name)

    print("\nfilename       : ", file_name,
          "\nfile_full_path : ", file_full_path,
          "\ncurrent_dir    : ", current_dir)

    print("\n\n\n")
    testfile = FileData(file_full_path, current_dir, debugmode = True)
    testfile.writeAndSave()



root = et.Element("annotation")

folder = et.SubElement(root, "folder")

filenmae = et.SubElement(root, "filname")

size = et.SubElement(root, "size")
width = et.SubElement(size, "size")
height = et.SubElement(size, "height")

obj = et.SubElement(root, "object")
name = et.SubElement(obj, "name")
bndbox = et.SubElement(obj, "bndbox")
xmin = et.SubElement(bndbox, "xmin")
ymin = et.SubElement(bndbox, "ymin")
xmax = et.SubElement(bndbox, "xmax")
ymax = et.SubElement(bndbox, "ymax")