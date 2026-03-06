import json
data = json.load(open('datasets/oxford_pets_subset_split/annotations_all.json'))
print('Categories:')
for c in data['categories'][:3]:
    print(f'{c["id"]}: {c["name"]}')
print('Sample annotations:')
for a in data['annotations'][:3]:
    print(f'img_id: {a["image_id"]}, cat_id: {a["category_id"]}')