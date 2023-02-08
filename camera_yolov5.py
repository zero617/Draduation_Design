from base_camera import BaseCamera

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

from TTS import *

class Camera(BaseCamera):
    @staticmethod
    # @smart_inference_mode    
    def frames():
        weights='yolov5s.pt'
        source='0'  # file/dir/URL/glob/screen/0(webcam)
        data='data/coco128.yaml'  # dataset.yaml path
        imgsz=(640, 640)  # inference size (height, width)
        conf_thres=0.25  # confidence threshold 置信度阈值
        iou_thres=0.45  # NMS IOU threshold 做nms的iou阈值
        max_det=1000  # maximum detections per image
        device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False  # show results
        save_txt=False  # save results to *.txt
        save_conf=False  # save confidences in --save-txt labels
        save_crop=False # save cropped prediction boxes
        nosave=False  # do not save images/videos
        classes=None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False  # class-agnostic NMS 进行nms是否也去除不同类别之间的框
        augment=False  # augmented inference 推理的时候进行多尺度，翻转等操作(TTA)推理
        visualize=False  # visualize features 是否可视化网络层输出特征
        update=False  # update all models 对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息
        project='runs/detect'  # save results to project/name
        name='exp'  # save results to project/name
        exist_ok=False  # existing project/name ok, do not increment
        line_thickness=3  # bounding box thickness (pixels)
        hide_labels=False  # hide labels
        hide_conf=False  # hide confidences
        half=False  # use FP16 half-precision inference 是否使用FP16精度推理
        dnn=False  # use OpenCV DNN for ONNX inference
        vid_stride=1  # video frame-rate stride

        source = str(source)
        # 是否保存图片
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        # 获取保存预测路径
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        # 获取设备
        device = select_device(device)
        # 加载Float32模型，确保用户设定的输入图片分辨率能整除32(如不能则调整为能整除并返回)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        # 获取步长、类别名称、
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        # 通过不同的输入源来设置不同的数据加载方式
        bs = 1  # batch_size
        if webcam:
            view_img = check_imshow(warn=True)
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        # 进行一次前向推理,测试程序是否正常
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        TTS("系统开始运行")
        """
        path 图片/视频路径
        im 进行resize+pad之后的图片
        im0s 原size图片
        vid_cap 当读取图片时为None，读取视频时为视频源
        """
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                # 图片也设置为Float16
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                # 没有batch_size的话则在最前面添加一个轴
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                """
                前向传播 返回pred的shape是(1, num_boxes, 5+num_class)
                h,w为传入网络图片的长和宽，注意dataset在检测时使用了矩形推理，所以这里h不一定等于w
                num_boxes = h/32 * w/32 + h/16 * w/16 + h/8 * w/8
                pred[..., 0:4]为预测框坐标
                预测框坐标为xywh(中心点+宽长)格式
                pred[..., 4]为objectness置信度
                pred[..., 5:-1]为分类结果
                """
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)

            # NMS
            """
            pred:前向传播的输出
            conf_thres:置信度阈值
            iou_thres:iou阈值
            classes:是否只保留特定的类别
            agnostic_nms:进行nms是否也去除不同类别之间的框
            max-det:保留的最大检测框数量
            经过nms之后，预测框格式：xywh-->xyxy(左上角右下角)
            pred是一个列表list[torch.tensor]，长度为batch_size
            每一个torch.tensor的shape为(num_boxes, 6),内容为box+conf+cls
            """
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                # 设置保存图片/视频的路径
                save_path = str(save_dir / p.name)  # im.jpg
                # 设置保存框坐标txt文件的路径
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                # 设置打印信息(图片长宽)
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    # 调整预测框的坐标：基于resize+pad的图片的坐标-->基于原size图片的坐标
                    # 此时坐标格式为xyxy
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    # 打印检测到的类别数量
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    # 保存预测结果
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            # 将xyxy(左上角+右下角)格式转为xywh(中心点+宽长)格式，并除上w，h做归一化，转化为列表再保存
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        # 在原图上画框
                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Stream results
                im0 = annotator.result()
                if view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    # cv2.imshow(str(p), im0)
                    yield cv2.imencode('.jpg', im0)[1].tobytes()
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)




