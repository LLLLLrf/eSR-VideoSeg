import os
from PIL import Image

def resize_image(input_image_path, output_image_path, size):
    with Image.open(input_image_path) as image:
        resized_image = image.resize(size, Image.LANCZOS)
        resized_image.save(output_image_path)

def main():
    input_directory = '.'
    output_directory = './process540x960'

    target_size = (540, 960)  # 目标尺寸

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_image_path = os.path.join(input_directory, filename)
            output_image_path = os.path.join(output_directory, filename)
            resize_image(input_image_path, output_image_path, target_size)
            print(f'{filename} 已被调整为 {target_size} 并保存到 {output_directory}')

if __name__ == '__main__':
    main()