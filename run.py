from PIL import Image
from tqdm import tqdm
from torchvision.transforms import ToTensor, ToPILImage
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from models import edgeSR_MAX, edgeSR_TM, edgeSR_CNN, edgeSR_TR, FSRCNN, ESPCN, Classic
from pathlib import Path
import thop
import cv2
import torch
import numpy as np
import torchvision.transforms.functional as TF
import time
import os
import skimage
import torch
from estimate.concat import concat_images
from process import process_video
from v_concat import v_concat

_ = torch.manual_seed(123)
lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [
        '.png', '.tif', '.jpg', '.jpeg', '.bmp', '.pgm', '.PNG'
    ])

times={}
count={}

def execution_time_decorator(func):
    def wrapper(*args, **kwargs):
        global times
        global count
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        # 存在则累加，不存在则初始化
        if func.__name__ in times:
            times[func.__name__] += execution_time
            count[func.__name__] += 1
        else:
            times[func.__name__] = execution_time
            count[func.__name__] = 1
        # print("函数", func.__name__, "执行：", round(execution_time,6), "秒")
        return result
    return wrapper

@execution_time_decorator
def process_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('YCbCr')
    y, cb, cr = img.split()
    
    if device == torch.device('cpu'):
        input_tensor = ToTensor()(y).unsqueeze(0).to(device)
    else:
        input_tensor = ToTensor()(y).unsqueeze(0).to(device).half()
        
    # export_model = pnnx.export(model, 'eSR.pt', input_tensor)
    # exit()
    # print("inputshape", input_tensor.shape)
    # exit()
    
    output_y = model(
        input_tensor
    ).data[0].clamp(0, 1.).expand(1, -1, -1).permute(1, 2, 0).cpu().numpy() * 255.

    output_y = Image.fromarray(np.uint8(np.round(output_y[:, :, 0])), mode='L')
    output_cb = cb.resize(output_y.size, Image.BICUBIC)
    output_cr = cr.resize(output_y.size, Image.BICUBIC)
    output_rgb = Image.merge('YCbCr', [output_y, output_cb, output_cr]).convert('RGB')

    return cv2.cvtColor(np.array(output_rgb), cv2.COLOR_RGB2BGR)

@execution_time_decorator
def process_img(y, cb, cr):
    # 准备输入张量
    input_tensor = ToTensor()(y).unsqueeze(0).to(device)
    if device != torch.device('cpu'):
        input_tensor = input_tensor.half()

    # 模型推理
    with torch.no_grad():  # 禁用梯度计算以加快推理速度
        # output_y = model(input_tensor).cpu().numpy()[0] * 255.
        output_y = model(input_tensor).clamp(0, 1.).cpu().numpy()[0] * 255.
        
    output_y = Image.fromarray(np.uint8(np.round(output_y[0])), mode='L')
    # 如果不考虑速度，可用LANCZOS插值，否则使用BICUBIC
    output_cb = cb.resize(output_y.size, Image.BICUBIC)
    output_cr = cr.resize(output_y.size, Image.BICUBIC)
    output_rgb = Image.merge('YCbCr', [output_y, output_cb, output_cr]).convert('RGB')

    return cv2.cvtColor(np.array(output_rgb), cv2.COLOR_RGB2BGR)

@execution_time_decorator
def bgr2yuv(img):
    # 将 BGR 图像转换为 RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 转换为 YCbCr 色彩空间
    img_ycbcr = Image.fromarray(img_rgb).convert('YCbCr')
    y, cb, cr = img_ycbcr.split()
    return y, cb, cr

def hr2lr(hr_img,scale=0.5):
    lr_img = cv2.resize(hr_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return lr_img

@execution_time_decorator
def sharpen_image(image, degree=1.2):
    # blur_img = cv2.GaussianBlur(image, (0, 0), 5)
    # usm = cv2.addWeighted(image, 1.5, blur_img, -0.5, 0)
    # return usm
    
    # 按照degree值进行锐化
    blur_img = cv2.GaussianBlur(image, (0, 0), degree)
    usm = cv2.addWeighted(image, 1.5, blur_img, -0.5, 0)
    return usm

@execution_time_decorator
def denoise_image(image, degree=5):
    # 可调整降噪程度
    return cv2.fastNlMeansDenoisingColored(image, None, degree, degree, 7, 21)

@execution_time_decorator
def clahe_enhancement(image, clip_limit=1.0, tile_grid_size=(8, 8)):
    '''
    param image: 输入图像
    param clip_limit: 对比度限制因子 (1.0 ~ 4.0)
    param tile_grid_size: 划分图像的网格大小 (8, 8), (16, 16), (32, 32) 等
    '''
    # 将图像从 BGR 转换为 LAB 色彩空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)  # 分离 L, A, B 通道

    # 创建 CLAHE 对象并应用于 L 通道
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)

    # 合并增强后的 L 通道与原始 A 和 B 通道
    enhanced_lab = cv2.merge((cl, a, b))

    # 将增强后的 LAB 图像转换回 BGR 色彩空间
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr



@execution_time_decorator
def calcu_psnr(img1, img2):
    mse = np.mean((img1/255. - img2/255.) ** 2)
    if mse == 0:
        return 100
    return 10 * np.log10(1. / mse)

@execution_time_decorator
def calcu_ssim(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ssim = skimage.metrics.structural_similarity(img1, img2, data_range=255, multichannel=False)
    return ssim

@execution_time_decorator
def calcu_lpips(img1, img2):
    img1 = TF.to_tensor(img1).unsqueeze(0).to(device)
    img2 = TF.to_tensor(img2).unsqueeze(0).to(device)
    lpips_score = lpips(img1, img2)
    return lpips_score

if __name__ == '__main__':
    # ############################# CHANGE HERE ###############################
    # 选择设备
    # device = torch.device('cuda:0')
    device = torch.device('cpu')
    # 选择模型 (K3_C1速度快精度低，K3_C3速度慢精度高)
    # model_file = 'model-files/eSR-MAX_s2_K3_C1.model'
    model_file = 'model-files/eSR-MAX_s2_K3_C3.model'
    # 视频处理
    input_video = 'video/480p.mp4'
    output_video = 'output/output_video.mp4'
    # 图像处理
    input_imgs = 'estimate/HR'
    processed_imgs = 'estimate/LR'
    output_imgs = 'estimate/SR'
    # 任务类型
    input_type = 'img'  # img|vid
    estimate = 0        # 是否计算评估指标 psnr, ssim, lpips (暂只支持HR图像输入)
    save = 0            # 是否保存处理后的图像/视频
    model_detail = 1    # 是否显示模型详细信息

    if not os.path.exists(output_imgs):
        os.makedirs(output_imgs)
    if not os.path.exists(processed_imgs):
        os.makedirs(processed_imgs)

    model_id = model_file.split('.')[-2].split('/')[-1]
    torch.backends.cudnn.benchmark = True
    # #########################################################################

    # with torch.cuda.device(device):
    with torch.no_grad(), torch.jit.optimized_execution(False):
        print('\n- Load model:', model_id)
        print('- On device:', device,'\n')
        if model_id.startswith('eSR-MAX_'):
            model = edgeSR_MAX(model_id)
        elif model_id.startswith('eSR-TM_'):
            model = edgeSR_TM(model_id)
        elif model_id.startswith('eSR-TR_'):
            model = edgeSR_TR(model_id)
        elif model_id.startswith('eSR-CNN_'):
            model = edgeSR_CNN(model_id)
        elif model_id.startswith('FSRCNN_'):
            model = FSRCNN(model_id)
        elif model_id.startswith('ESPCN_'):
            model = ESPCN(model_id)
        elif model_id.startswith('Bicubic_'):
            model = Classic(model_id)
        else:
            assert False

        model.load_state_dict(
            torch.load(model_file, map_location=lambda storage, loc: storage),
            strict=True
        )
        # 查看模型输入输出
        # import pnnx
        # export_model = pnnx.export(model, 'eSR.pt',torch.randn(1, 1, 256, 256))
        # torch.save(model.state_dict(), 'model.pt')
        # exit()
        
        if device == torch.device('cpu'):
            model.to(device)
        else:
            model.to(device).half()

        if input_type == 'img':
            psnr=[]
            ssim=[]
            lpips_scores=[]
            input_list = [
                str(f) for f in Path(input_imgs).iterdir() if is_image_file(f.name)
            ]
            
            for input_file in tqdm(input_list):
                hr_img = cv2.imread(input_file)
                # 检查是否为1920*1080
                if hr_img.shape[0] != 1080 or hr_img.shape[1] != 1920:
                    print('{}不是标准图片，shape={}'.format(input_file), hr_img.shape)
                lr_img = hr2lr(hr_img, scale=0.5)
                y, cb, cr = bgr2yuv(lr_img)
                processed_img = process_img(y, cb, cr)
                # degree值越大，锐化效果越明显
                processed_img = sharpen_image(processed_img, degree=1.2)
                # 暂未使用降噪
                # processed_img = denoise_image(processed_img, degree=5)
                processed_img = clahe_enhancement(processed_img)
                if save:
                    cv2.imwrite(os.path.join(output_imgs, Path(input_file).name), processed_img)
                    cv2.imwrite(os.path.join(processed_imgs, Path(input_file).name), lr_img)
                if estimate:
                    psnr.append(calcu_psnr(hr_img, processed_img))
                    ssim.append(calcu_ssim(hr_img, processed_img))
                    lpips_scores.append(calcu_lpips(Image.fromarray(hr_img), Image.fromarray(processed_img)))
            if estimate:
                print('PSNR:', round(np.mean(psnr),8), '\tmax:', round(np.max(psnr),8), '\tmin:', round(np.min(psnr),8))
                print('SSIM:', round(np.mean(ssim),8), '\tmax:', round(np.max(ssim),8), '\tmin:', round(np.min(ssim),8))
                print('LPIPS:', round(np.mean(lpips_scores),8), '\tmax:', round(np.max(lpips_scores),8), '\tmin:', round(np.min(lpips_scores),8))
            print('fps:', round(1/(times['process_img']/count['process_img']),6))
            for key in times:
                print('函数', key,'\t平均：',round(times[key]/count[key],6), round(1/(times[key]/count[key]),6),'\t共', round(times[key],6), '秒， 执行次数：', count[key])
            if save:
                # 路径进入estimate文件夹
                os.chdir('estimate')
                concat_images()
        else:
            cap = cv2.VideoCapture(input_video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # out = cv2.VideoWriter(output_video, fourcc, fps, (1728, 972))
            # out = cv2.VideoWriter(output_video, fourcc, fps, (width*2, height*2))

            psnr=[]
            ssim=[]
            # get frame count
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            pbar = tqdm(total=frame_count)
            while cap.isOpened():
                ret, frame = cap.read()
                index = cap.get(cv2.CAP_PROP_POS_FRAMES)
                pbar.update(1)
                if not ret:
                    break

                processed_frame = process_frame(frame)
                processed_frame = sharpen_image(processed_frame)
                # process_frame = denoise_image(processed_frame, degree=5)
                processed_frame = clahe_enhancement(processed_frame)
                
                # 计算psnr和ssim
                # psnr.append(calcu_psnr(frame, processed_frame))
                # ssim.append(calcu_ssim(frame, processed_frame))
                
                # if index > 100:
                #     break            
                if save:
                    cv2.imwrite('vid_img/output{}'.format(int(index))+'.jpg', processed_frame)
                    # 直接写入视频会出错，文件损坏，暂未解决
                    # out.write(processed_frame)
            cap.release()

            if save:
                process_video(output=output_video, image_folder='vid_img')
                print('视频已保存至', output_video)
                # os.chdir('..')
                print('now path:', os.getcwd())
                v_concat()
                print('视频已拼接')
                
            print('fps:', round(1/(times['process_frame']/count['process_frame']),6))
            for key in times:
                print('函数', key,'\t平均：',round(times[key]/count[key],6), round(1/(times[key]/count[key]),6),'\t共', round(times[key],6), '秒， 执行次数：', count[key])
            print('总计')
            total_time = 0
            for key in times:
                total_time += times[key]
            print('总计时间：', round(total_time,6), '秒')
            print("总计帧率：", round(frame_count/total_time,6))
            # print('PSNR:', np.mean(psnr))
            # print('SSIM:', np.mean(ssim))
            
            # out.release()
        
        if model_detail:
            print('\n- Model detail:', model_id)
            macs, params = thop.profile(model, inputs=(torch.randn(1, 1, 1920//2, 1080//2),), verbose=False)
            print(f"- FLOPs: {macs / 1e6:.6f} MFLOPs")
            print(f"- Params: {params / 1e6:.6f} M")
        