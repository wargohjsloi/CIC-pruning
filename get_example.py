import os
import shutil
import random
from PIL import Image

# 定义要搜索的类别
categories = ['n01530575', 'n02123045', 'n02091134', 'n04285008']


def select_random_images(dataset_path, output_path, categories, num_images=20):
    # 确保输出路径存在
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 遍历每个类别
    for category in categories:
        # 构建目标类别的路径
        category_path = os.path.join(dataset_path, category)
        # 确保类别路径存在
        if not os.path.exists(category_path):
            print(f"Category path {category_path} does not exist.")
            continue

        # 获取所有图片的路径
        image_paths = [os.path.join(category_path, f) for f in os.listdir(category_path) if
                       f.endswith(('.jpg', '.jpeg', '.png', '.JPEG'))]
        print(len(image_paths))

        # 随机选择图片
        selected_images = random.sample(image_paths, min(num_images, len(image_paths)))

        # 创建输出子文件夹
        output_category_path = os.path.join(output_path, category)
        if not os.path.exists(output_category_path):
            os.makedirs(output_category_path)

        # 复制图片到输出路径
        for image_path in selected_images:
            # 打开图片
            with Image.open(image_path) as img:
                # 确保图片可以打开
                if img.format:
                    # 构建目标路径
                    target_path = os.path.join(output_category_path, os.path.basename(image_path))
                    # 复制图片
                    shutil.copy(image_path, target_path)
                    print(f"Copied {image_path} to {target_path}")


# 输入参数
dataset_path = '/data/imagenet/train'
output_path = './outputs/example'

# 调用函数
select_random_images(dataset_path, output_path, categories)