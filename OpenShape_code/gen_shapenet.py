import json

# 读取 JSON 文件
with open('/home/ssf/data/meta_data/split/train_all.json', 'r') as file:
    data = json.load(file)

# 筛选出 dataset 为 'shapenet' 的元素，并修改 data_path
shapenet_data = []
for item in data:
    if item.get('dataset') == 'ShapeNet':
        # 修改 data_path 路径中的 /mnt/data 为 /data
        item['data_path'] = item['data_path'].replace('/mnt/data', '/home/ssf/data')
        shapenet_data.append(item)

# 将筛选出的元素写入新的 JSON 文件
with open('/home/ssf/data/meta_data/split/train_shapenet.json', 'w') as file:
    json.dump(shapenet_data, file, indent=4)

print("筛选完成，并已修改 data_path 路径，已生成 shapenet_data.json 文件")