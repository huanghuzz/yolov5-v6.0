1# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
进行预测图片，传入指定的图片
python detect.py --source data\\image\\bus.jpg         data\\image\\bus.jpg 图片地址，\\是因为Windows下所有这样
"""
#导入头文件
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
#确定yolo文件路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
#模块查询路径列表，使后续导包正常
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
#导入数据建立的类
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync

'''
分六部分


    #第一部分sourse
    # Directories 新建保存文件文件夹
    # Load model 负责加载模型的权重
    # Dataloader 负责加载待预测图片
    # Print results 打印一些输出信息

'''
@torch.no_grad()
def run(weights=ROOT / 'weights/yolov5s.pt',  # model.pt path(s)
        source=ROOT / '0',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    #第一部分sourse  处理预测路径
    source = str(source)  #传入路径，并强制转义字符串
    #save_img:bool 判断是否要保存图片
    save_img = not nosave and not source.endswith('.txt')  # save inference images，保存预测图片
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))    #判断是否网络流

    # Directories 新建保存文件文件夹
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run 确认exp到多少
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir 新建文件夹，存储识别模型

    # Initialize
    set_logging() #设置日志
    device = select_device(device)  #设置设备
    half &= device.type != 'cpu'  # half precision only supported on CUDA 半精度推理

    # Load model 负责加载模型的权重
    w = str(weights[0] if isinstance(weights, list) else weights)
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        if dnn:
            # check_requirements(('opencv-python>=4.5.4',))
            net = cv2.dnn.readNetFromONNX(w)
        else:
            check_requirements(('onnx', 'onnxruntime'))
            import onnxruntime
            session = onnxruntime.InferenceSession(w, None)
    else:  # TensorFlow models
        check_requirements(('tensorflow>=2.4.1',))
        import tensorflow as tf
        if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
                return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                               tf.nest.map_structure(x.graph.as_graph_element, outputs))

            graph_def = tf.Graph().as_graph_def()
            graph_def.ParseFromString(open(w, 'rb').read())
            frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
        elif saved_model:
            model = tf.keras.models.load_model(w)
        elif tflite:
            interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader 负责加载待预测图片
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        #读取视频流
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        #读取照片
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference 执行推理的过程，
    if pt and device.type != 'cpu':
        '''
        将模型加入到设备，并为同一类型（因为是Ensemble（集成）的模型，每个模型的参数类型不一样，
        我们需要统一一下），输入torch.zeros（1，3，imgsz,imgsz）,是为声明，输入的形状，
        同时也可以判断模型是否正常运行
        '''
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, img, im0s, vid_cap in dataset:
        t1 = time_sync()
        if onnx:
            img = img.astype('float32')
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            if dnn:
                net.setInput(img)
                pred = torch.tensor(net.forward())
            else:
                pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
        else:  # tensorflow model (tflite, pb, saved_model)
            imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
            if pb:
                pred = frozen_func(x=tf.constant(imn)).numpy()
            elif saved_model:
                pred = model(imn, training=False).numpy()
            elif tflite:
                if int8:
                    scale, zero_point = input_details[0]['quantization']
                    imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
                interpreter.set_tensor(input_details[0]['index'], imn)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])
                if int8:
                    scale, zero_point = output_details[0]['quantization']
                    pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
            pred[..., 0] *= imgsz[1]  # x
            pred[..., 1] *= imgsz[0]  # y
            pred[..., 2] *= imgsz[1]  # w
            pred[..., 3] *= imgsz[0]  # h
            pred = torch.tensor(pred)
        t3 = time_sync()
        dt[1] += t3 - t2


        #进行非极大值抑制
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            #定义图片，txt等存储的地址
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            #因为pred结果的box是在加边（padding）后的图片上的坐标，所以要还原到原图的坐标
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                # 加边后图的坐标转为原图坐标
                # det(N,6) ，6代表x1,y1,x2,y2,conf,cls  ，img(1,3,H,W)
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class 类别数
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                #  xyxy:List[x1,y1,x2,y2]
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # 给推理的图片加box
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        #保存crop图
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            print(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            #保存结果，图片就保存图片，视频就保存视频片段
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
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results 打印一些输出信息
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)



def parse_opt():  #定义了很多参数
    parser = argparse.ArgumentParser()
    #修改训练好的权重文件路径----传入训练好的模型
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/detect/fan.pt', help='model path(s)')
    #传入识别视频，或调用摄像头
    parser.add_argument('--source', type=str, default=ROOT / 'E:/720/1.mp4', help='file/dir/URL/glob, 0 for webcam')
    #修改图片大小，imgsz默认640
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    #调整置信度，当检测出来的置信度大于该数值才能被检测出来，就是显示出来框框
    parser.add_argument('--conf-thres', type=float, default=0.55, help='confidence threshold')
    #NMS IOS 阈值----非极大抑制，具体不赘述，自行查阅，可不改，目的消除多余检测框---当某个框和当前最佳框的重叠程度大于阈值认为重复删除该框
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    #最大侦察的目标数
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    #GPU加速----选择是使用gpu还是cpu
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    #展示推理后图片
    parser.add_argument('--view-img', action='store_true', help='show results')
    #结果保存为txt
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    #在保存的txt里面，除了类别，再保存对应的置信度
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    #保存用为目标框crop的图片
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    #不保存图片/视频
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    #过滤得到为classes分类的图片
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    #不同类别间可以做NMS（不开启的话，每个类别单独做NMS）
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    #推理增强
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    #将模型中包含的优化器，ema等操作进行去除，减少模型大小（MB）
    parser.add_argument('--update', action='store_true', help='update all models')
    #推理保存的工程目录
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    #本次结果的保存文件夹名
    parser.add_argument('--name', default='exp', help='save results to project/name')
    #每次运行都会创建一个新的文件夹，相关内容保存在这个下面，如果为Ture，则会在之前文件下保存
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    #边界框厚度
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    #隐藏每个目标的标签
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    #隐藏每个目标的置信度
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    #FP16 半精度推理（增加推理速度） 上面默认false 故没有使用
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)#打印参数数据信息
    return opt #返回数据


def main(opt):
    # 检测依赖包安装，检测requirements.txt
    check_requirements(exclude=('tensorboard', 'thop'))
    #运行run
    run(**vars(opt))

#在头文件执行完，跳到这里运行
if __name__ == "__main__":
    opt = parse_opt() #解析之前传入的
    main(opt)#导入数据到main
