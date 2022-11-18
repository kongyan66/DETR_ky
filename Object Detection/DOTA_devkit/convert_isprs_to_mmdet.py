import os
import os.path as osp
from tqdm import tqdm
import mmcv
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from mmdet.core import poly_to_rotated_box_single  # 8 point to 5 point xywha

wordname_37 = ['Boeing737', 'Boeing747', 'Boeing777', 'Boeing787', 'C919', 'A220', 'A321', 'A330', 'A350', 'ARJ21', 'other-airplane', 'Passenger Ship',
               'Motorboat', 'Fishing Boat', 'Tugboat', 'Engineering Ship', 'Liquid Cargo Ship', 'Dry Cargo Ship', 'Warship', 'other-ship',
               'Small Car', 'Bus', 'Cargo Truck', 'Dump Truck', 'Van', 'Trailer', 'Tractor', 'Excavator', 'Truck Tractor', 'other-vehicle', 'Basketball Court',
               'Tennis Court', 'Football Field', 'Baseball Field', 'Intersection', 'Roundabout', 'Bridge']
# output dict{'Boeing737': 1, 'Boeing747': 2,~}
label_ids = {name: i + 1 for i, name in enumerate(wordname_37)}
# get an .xml file annotation
def parse_ann_info(label_base_path, img_name):
    lab_path = osp.join(label_base_path, img_name + '.xml')
    bboxes, labels, bboxes_ignore, labels_ignore = [], [], [], []

    tree = ET.parse(lab_path)
    # Get the root node
    root = tree.getroot()

    objs = root.find('objects')
    bbox_base = []
    for obj in objs.findall('object'):
        class_name = obj.find('possibleresult').find('name').text
        points = obj.find('points')
        for i,point in enumerate(points.findall('point')):
            if i<4:
              bbox_base.append(int(float((point.text).split(',')[0])))
              bbox_base.append(int(float((point.text).split(',')[1])))

        bbox_base[2],bbox_base[3], bbox_base[6], bbox_base[7] = bbox_base[6],bbox_base[7], bbox_base[2],bbox_base[3]
        bbox = tuple(poly_to_rotated_box_single(bbox_base).tolist())
        bboxes.append(bbox)
        labels.append(label_ids[class_name])
        bbox_base = []
    return bboxes, labels, bboxes_ignore, labels_ignore

def convert_isprs_to_mmdet(src_path, out_path, trainval=True, ext='.tif'):
    """Generate .pkl format annotation that is consistent with mmdet.
    Args:
        src_path: dataset path containing images and labelTxt folders.
        out_path: output pkl file path
        trainval: trainval or test
    """
    img_path = os.path.join(src_path, 'images')
    label_path = os.path.join(src_path, 'labelXml')
    img_lists = os.listdir(img_path)

    data_dict = []
    for id, img in tqdm(enumerate(img_lists)):
        img_info = {}
        img_name = osp.splitext(img)[0]
        label = os.path.join(label_path, img_name + '.xml')
        img = Image.open(osp.join(img_path, img))
        img_info['filename'] = img_name + ext
        img_info['height'] = img.height
        img_info['width'] = img.width
        if trainval:
            if not os.path.exists(label):
                print('Label:' + img_name + '.xml' + ' Not Exist')
            else:
                bboxes, labels, bboxes_ignore, labels_ignore = parse_ann_info(label_path, img_name)
                ann = {}
                ann['bboxes'] = np.array(bboxes, dtype=np.float32)
                ann['labels'] = np.array(labels, dtype=np.int64)
                ann['bboxes_ignore'] = np.array(bboxes_ignore, dtype=np.float32)
                ann['labels_ignore'] = np.array(labels_ignore, dtype=np.int64)
                img_info['ann'] = ann
        data_dict.append(img_info)
    print(data_dict)
    mmcv.dump(data_dict, out_path)


if __name__ == '__main__':
    # bboxes, labels, bboxes_ignore, labels_ignore = parse_ann_info('/home/z/code/s2anet/data/competion/train/part1/labelXml','0')
    # print(bboxes)
    # print(labels)
    convert_isprs_to_mmdet('/home/z/code/s2anet/data/competion/train/test',
                         '/home/z/code/s2anet/data/competion/train/test/train_isprs.pkl')
    # convert_dota_to_mmdet('data/dota_1024/test_split',
    #                      'data/dota_1024/test_split/test_s2anet.pkl', trainval=False)
    # print('done!')
