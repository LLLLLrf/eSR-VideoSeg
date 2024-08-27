import os
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

HR = 'HR'
LR = 'LR'
SR = 'SR'
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [
        '.png', '.tif', '.jpg', '.jpeg', '.bmp', '.pgm', '.PNG'
    ])

def add_label(image, label):
    draw = ImageDraw.Draw(image)
    # 使用默认字体，您可以选择其他字体
    font = ImageFont.load_default(size=55)
    # 获取文本的边界框
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    # 在图片底部添加标签
    draw.text(((image.width - text_width) / 2, image.height - text_height - 30), label, font=font, fill="white")
    return image
    
def concat_images():
    for hr_file in Path(HR).iterdir():
        if is_image_file(hr_file.name):
            hr_img = Image.open(hr_file)
            lr_file = Path(LR) / hr_file.name
            sr_file = Path(SR) / hr_file.name
            lr_img = Image.open(lr_file)
            sr_img = Image.open(sr_file)

            # 添加标签
            hr_img = add_label(hr_img, "HR")
            sr_img = add_label(sr_img, "SR")
            lr_img = lr_img.resize((hr_img.width, hr_img.height), Image.NEAREST)
            lr_img = add_label(lr_img, "LR")

            # 创建拼接图像
            concat_img = Image.new('RGB', (hr_img.width, hr_img.height * 3))
            concat_img.paste(hr_img, (0, 0))
            concat_img.paste(sr_img, (0, hr_img.height))
            concat_img.paste(lr_img, (0, hr_img.height*2))

            # 保存拼接后的图像
            concat_img.save(Path(output_dir) / hr_file.name)

if __name__ == '__main__':
    concat_images()