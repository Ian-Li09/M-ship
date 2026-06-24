# M-Ship

M-Ship is a PyTorch project for SAR ship image background generation and inpainting. It uses a diffusion model to preserve ship regions while regenerating masked background regions under different scene conditions such as offshore and inshore.


## Source Acknowledgements

This project reuses and adapts code from the following open-source projects. Add or update exact links here before release:

- Ship-Go / SAR image generation reference: [Ship-Go](https://github.com/XinZhangRadar/Ship-Go/blob/main/README.md)
- SSDD dataset reference: [Official-SSDD](https://github.com/TianwenZhang0825/Official-SSDD)
- RF-DETR downstream detector reference: [roboflow/rf-detr](https://github.com/roboflow/rf-detr)

## Weights


- Pretrained M-Ship weights:[Baidu Netdisk](https://pan.baidu.com/s/1Afh2asn-LiBgIIaPjqi9pQ)  #key==kkyy 
- Generated SAR images: TODO: add generated image archive or gallery link
- Downstream detector weights:[RF-DETR fine-tuned weight link](https://pan.baidu.com/s/1Afh2asn-LiBgIIaPjqi9pQ)#key==kkyy 



## Project Structure

```text
M-ship/
  config/
    sard.json                     # main train/test configuration
  core/
    base_model.py                 # training/test base class
    base_network.py               # network init utilities
    logger.py                     # log and image writer
    praser.py                     # config parser and experiment folder setup
    util.py                       # seed/device/helpers
  data/
    dataset.py                    # SSDD/SAR dataset, masks, env labels
    util/mask.py                  # mask helpers
  models/
    model.py                      # SARD trainer/test wrapper
    network.py                    # BCDif diffusion network
    loss.py                       # loss functions
    metric.py                     # MAE, IS, etc.
    guided_diffusion_modules/     # guided-diffusion style UNet
    sr3_modules/                  # SR3 style UNet
  preprocess/
    mirflickr25k_preprocess.py
  slurm/
    ssdd.slurm                    # cluster job example
  eval.py                         # FID / Inception Score evaluation
  run.py                          # train/test entry point
  requirements.txt
```

## Environment

Install dependencies in a Python environment with PyTorch and CUDA support:

```bash
pip install -r requirements.txt
```

Main dependencies:

- `torch`
- `torchvision`
- `numpy`
- `opencv-python`
- `tqdm`
- `tensorboardX`
- `clean-fid`

The default config uses CUDA. Set `gpu_ids` in `config/sard.json` or pass `-gpu` at runtime.

## Data Format

The dataset class is `I2IDataset` in `data/dataset.py`.

Expected data layout:

```text
datasets/sargen/
  flist/
    train.flist
    valid.flist
    test.flist
    train_offshore.flist
    train_inshore.flist
    test_offshore.flist
    test_inshore.flist
    offshore.txt
    inshore.txt
  image/
    000001.jpg
    ...
  Annotations_seg/
    000001.xml
    ...
```

`flist` files contain image paths or filenames used by the dataset loader. Environment labels are inferred by checking whether an image filename is listed in `offshore.txt` or `inshore.txt`.





## Configuration

The main configuration is [config/sard.json](config/sard.json).




## Training

Run training with:

```bash
python run.py -p train -c config/sard.json
```


The parser creates a timestamped experiment folder:

```text
experiments/train_i2i_sar_<timestamp>/
  code/             # snapshot of config/core/data/models/slurm and scripts
  tb_logger/        # TensorBoard logs
  results/          # validation images
  checkpoint/       # saved checkpoints
  config.json       # resolved runtime config
  train.log
```

## Testing / Generation

Run generation with:

```bash
python run.py -p test -c config/sard.json
```

Outputs are written under:

```text
experiments/test_i2i_sar_<timestamp>/results/
```

The test loop saves:

- `GT_*`: original image
- `Process_*`: intermediate diffusion samples
- `Out_*`: final generated output
- `Mask_*`: mask visualization


## RF-DETR Detection Usage




## Citation


```bibtex
@misc{mship,
  title = {M-Ship:employing multi-condition diffusion model to generate a SAR image ship target detection dataset},
  author = {Ian_li,WANG_xinyan},
  year = {2026},
  url = {https://github.com/Ian-Li09/M_ship}
}
```
