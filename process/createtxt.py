import os

def createtxt(origin_data_path):
    image_path=origin_data_path+'images/'
    GT_path=origin_data_path+'label/'
    filenames = os.listdir(origin_data_path)
    data_list = []
    GT_list = []

    for filename in filenames:
        ext = os.path.splitext(filename)[-1]
        if ext == '.jpg':
            filename = filename.split('_')[-1][:-len('.jpg')]
            data_list.append('ISIC_' + filename + '.jpg')
            GT_list.append('ISIC_' + filename + '_segmentation.png')

if __name__=="__main__":
    origin_data_path="./dataset/train/"
    createtxt(origin_data_path)
    print('create!')