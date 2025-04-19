# SAMRefiner++: SAMRefiner + IoU Adaption


## Installation

Create conda env same to SAMRefiner. Then, use the SAM mask decoder with LoRA as follows:

```
cd SAMRefiner_plus
cd segment-anything; pip install -e .; cd ..

```


## Usage
### SAM

First download a [model checkpoint](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints) to <path/to/checkpoint>. For example, download the default sam_vit_h:
```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O <path/to/checkpoint>
```

### DAVIS-585
Download DAVIS-585 datasets from [ClickSEG](https://drive.google.com/file/d/18AudWkq1IloV1PTInt1diBsxwp3-fFBY/view?usp=drive_link) and unzip it to <path/to/datasets>.

Then the IoU Adaption step can be performed as follows:

```
python iou_adaption_davis585.py --sam_checkpoint <local_path/sam_vit_h_4b8939.pth> --dataset_path <local_path/DAVIS585/data>
```

This script will perform both training and evaluation. The adapted checkpoint will be saved to <local_path/sam_vit_h_4b8939_iou_adaption.pth>