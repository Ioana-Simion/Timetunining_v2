import argparse
import os
import torch
import wandb
from data_loader import SamplingMode, VideoDataModule
from eval_metrics import PredsmIoU
from models import FeatureForwarder, FeatureExtractor
import video_transformations



project_name = "TimeTuning_v2"


def run(args):
    device = args.device
    batch_size = args.batch_size
    num_workers = args.num_workers
    input_size = args.input_size
    num_clips = args.num_clips
    num_workers = args.num_workers
    num_clip_frames = args.num_clip_frames
    regular_step = args.regular_step
    context_frames = args.context_frames
    context_window = args.context_window
    topk = args.topk
    uvos_flag = args.uvos_flag
    precision_based = args.precision_based
    many_to_one = args.many_to_one

    logger = wandb.init(project=project_name, group='exp_patch_correspondence', job_type='debug')
    rand_color_jitter = video_transformations.RandomApply([video_transformations.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.8)
    data_transform_list = [rand_color_jitter, video_transformations.RandomGrayscale(), video_transformations.RandomGaussianBlur()]
    data_transform = video_transformations.Compose(data_transform_list)
    video_transform_list = [video_transformations.Resize((input_size, input_size), 'bilinear'), video_transformations.ClipToTensor(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])] #video_transformations.RandomResizedCrop((224, 224))
    video_transform = video_transformations.Compose(video_transform_list)
    transformations_dict = {"data_transforms": None, "target_transforms": None, "shared_transforms": video_transform}
    prefix = "/ssdstore/ssalehi/dataset"
    data_path = os.path.join(prefix, "train1/JPEGImages/")
    annotation_path = os.path.join(prefix, "train1/Annotations/")
    meta_file_path = os.path.join(prefix, "train1/meta.json")
    path_dict = {"class_directory": data_path, "annotation_directory": annotation_path, "meta_file_path": meta_file_path}
    sampling_mode = SamplingMode.Full
    video_data_module = VideoDataModule("ytvos", path_dict, num_clips, num_clip_frames, sampling_mode, regular_step, batch_size, num_workers)
    video_data_module.setup(transformations_dict)
    data_loader = video_data_module.get_data_loader()
    vit_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    feature_extractor = FeatureExtractor(vit_model, 14, d_model=384)
    FF = FeatureForwarder(feature_extractor.eval_spatial_resolution, context_frames, context_window, topk, feature_head=None)
    FF = FF.to(device)
    FF.eval()
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    predictions = []
    for batch_idx, batch in enumerate(data_loader):
        inputs, annotations = batch
        inputs = inputs.to(device).squeeze()
        annotations = annotations.to(device).squeeze()
        if uvos_flag:
            idx = annotations > 0
            annotations[idx] = 1
        features, _ = feature_extractor.forward_features(inputs)
        first_frame_segmentation = annotations[0]
        n_dims = int(first_frame_segmentation.max()+ 1)
        one_hot_segmentation = torch.nn.functional.one_hot(first_frame_segmentation.long(), n_dims).permute(2, 0, 1).float()
        prediction = FF(features, one_hot_segmentation)
        prediction = torch.stack(prediction, dim=0)
        prediction = torch.nn.functional.interpolate(prediction, size=(inputs.size(-2), inputs.size(-1)), mode="nearest")
        _, prediction = torch.max(prediction, dim=1)
        prediction = adjust_max(prediction)
        annotations = adjust_max(annotations)
        num_classes = len(torch.unique(annotations))
        predictions.append(prediction)
        predsmIoU = PredsmIoU(num_classes, num_classes)
        predsmIoU.update(prediction.flatten(), annotations[1:].flatten())
        score, tp, fp, fn, reordered_preds, matched_bg_clusters = predsmIoU.compute(True, many_to_one, precision_based=precision_based)
        print(score)

def adjust_max(input):
    input = input
    unique = torch.unique(input)
    for i in range(len(unique)):
        input[input == unique[i]] = i
    return input


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--device", type=str, default="cuda:7")
    args.add_argument("--batch_size", type=int, default=1)
    args.add_argument("--num_workers", type=int, default=8)
    args.add_argument("--input_size", type=int, default=224)
    args.add_argument("--num_clips", type=int, default=1)
    args.add_argument("--num_clip_frames", type=int, default=4)
    args.add_argument("--regular_step", type=int, default=1)
    args.add_argument("--context_frames", type=int, default=4)
    args.add_argument("--context_window", type=int, default=2)
    args.add_argument("--topk", type=int, default=4)
    args.add_argument("--uvos_flag", type=bool, default=False)
    args.add_argument("--precision_based", type=bool, default=False)
    args.add_argument("--many_to_one", type=bool, default=False)
    args = args.parse_args()
    run(args)