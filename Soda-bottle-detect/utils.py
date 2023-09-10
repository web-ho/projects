import json
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches


class_labels = {'coca-cola':0, 'fanta':1, 'sprite':2}


def make_df(modes):
    for mode in modes:
        with open(f'{mode}/_annotations.coco.json', 'r') as f:
            data = json.load(f)

        file_names = [image['file_name'] for image in data['images']]
        heights = [image['height'] for image in data['images']]
        widths = [image['width'] for image in data['images']]

        image_ids_annot = [annot['image_id'] for annot in data['annotations']]
        category_ids = [annot['category_id'] for annot in data['annotations']]
        bboxes = [annot['bbox'] for annot in data['annotations']]
        areas = [annot['area'] for annot in data['annotations']]

        df = pd.DataFrame({
            'image_id': image_ids_annot,
            'file_name': [file_names[id] for id in image_ids_annot],
            'height': [heights[id] for id in image_ids_annot],
            'width': [widths[id] for id in image_ids_annot],
            'category_id': category_ids,
            'bbox': bboxes,
            'area': areas,
        })

        category_names = {category['id']: category['name'] for category in data['categories']}
        df['category_name'] = df['category_id'].map(category_names)
        df['category_id'] = df['category_name'].map(class_labels)
        df.to_csv(f'{mode}_data.csv', index=False)



def count_bottles(df):
    data = []
    file_names = df['file_name'].unique()
    for f in file_names:
        new = df[df['file_name'] == f]
        coca_cola = 0
        fanta = 0
        sprite = 0
        for category in new['category_name']:
            if category == 'coca-cola':
                coca_cola += 1
            elif category == 'fanta':
                fanta += 1
            elif category == 'sprite':
                sprite += 1
        count_arr = {'coca-cola':coca_cola, 'fanta':fanta, 'sprite':sprite}
        data.append({'file_name': f, 'true_counts': count_arr})
    #count_df = pd.DataFrame(data)
    #new_df = df.merge(count_df, on='file_name', how='left')
    return data


def preprocess_bbox(df):
    bboxs = np.stack(df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        df[column] = bboxs[:,i]
    df.drop(columns=['bbox'], inplace=True)
    #new_df = count_bottles(df)
    return df


def display_img(image, results):

    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.axis('off')
    int2labels = {0:'coca_cola', 1:'fanta', 2:'sprite'}

    for r in results:
        #name = r.path.split('\\')[-1]
        boxes = r.boxes.xyxy.cpu().numpy()
        labels = r.boxes.cls.cpu().numpy()
        _, label_counts = np.unique(labels, return_counts=True)
        coca_cola = label_counts[0]
        fanta = label_counts[1]
        sprite = label_counts[2]
        total = coca_cola+fanta+sprite
        
        print(f"'Total Bottles':{total}, 'Coca-Cola':{coca_cola}, 'Fanta':{fanta}, 'Sprite':{sprite}")
        for bbox, label in zip(boxes, labels):
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            label = int2labels[label]
            ax.text(bbox[0], bbox[1] - 2, label, fontsize=10, color='r', verticalalignment='top')
    return fig, ax, coca_cola, fanta, sprite

