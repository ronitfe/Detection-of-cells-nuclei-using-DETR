import pandas as pd

path = 'bboxes.csv'
save_json_path = 'traincoco.json'
data = pd.read_csv(path)



def image(row):
    image = {}
    image["height"] = row.h
    image["width"] = row.w
    image["id"] = row.fileid
    image["file_name"] = str(row.train_id) + '/images/' + str(row.train_id) + '.png'
    return image

def category(row):
    category = {}
    category["supercategory"] = 'None'
    category["id"] = row.train_id
    category["name"] = 'None'
    return category

def annotation(row):
    annotation = {}
    area = (row.w)*(row.h)
    annotation["segmentation"] = []
    annotation["iscrowd"] = 0
    annotation["area"] = area
    annotation["image_id"] = row.fileid

    annotation["bbox"] = [row.x, row.y, row.w,row.h ]

    annotation["category_id"] = row.categoryid
    annotation["id"] = row.annid
    return annotation
