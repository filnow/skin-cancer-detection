# Skin-Cancer-Detection

Repo for training and experiments with skin-cancer dataset from kaggle

[Link to dataset](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign)

[Link to kaggle notebook](https://www.kaggle.com/code/filnow/acc-92-on-test-set-with-simple-pytorch)

## Experiments

I was able to experiments only with CNN that fits in 4GB of RAM on my GPU.

Diffrent architectures that was use: 

    * Custom CNN - my CNN with 3 conv layers and 2 fc layers
    * MobileNetV2 - pretrained wieghts from ImageNetV2, 3.5MLN PARM
    * RegNet X 400MF - pretrained weights from ImageNetV2, 5.5MLN PARM
    * EfficientNetB0 - pretrained weights from ImageNetV1, 5.3MLN PARM

The best was EfficientNetB0, with early stopping it achived 92% accuracy on test set,
bigger EfficientNet like B1 or B2 probably will give even better results that this.

All the experiments was documented using Weights&Biases.

[Link to Wandb project](https://wandb.ai/filnow42/skin-cancer)

After experiments I wrote training script using PyTorch Lightning.


