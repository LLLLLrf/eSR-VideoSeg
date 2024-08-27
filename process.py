import cv2
import os

def process_video(output='.', image_folder = './vid_img'):
    # 获取图像文件列表并按文件名排序
    # image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
    image_files = ["output{}.jpg".format(i) for i in range(1,749)]
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = output
    print(output_path)
    out = cv2.VideoWriter(output_path, fourcc, 25, (1728, 972))

    # 读取第一张图像并写入视频
    first_image_file = image_files[0]
    first_image_path = os.path.join(image_folder, first_image_file)
    first_frame = cv2.imread(first_image_path)
    out.write(first_frame)

    # 逐个读取剩余图像文件并写入视频
    # for image_file in image_files[1:80]:

    #     image_path = os.path.join(image_folder, image_file)
    #     frame = cv2.imread(image_path)
    #     out.write(frame)
    for image_file in image_files[1:]:
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)
        out.write(frame)
    # 释放视频写入器
    out.release()
    
if __name__ == '__main__':
    process_video()