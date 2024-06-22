# KAN in Remote Sensing

In this study, we present the first approach for integrating the Kolmogorov-Arnold Network (KAN) with various pre-trained Convolutional Neural Network (CNN) models for remote sensing (RS) scene classification tasks using the EuroSAT dataset. Our novel methodology, named KCN, aims to replace traditional Multi-Layer Perceptrons (MLPs) with KAN to enhance classification performance. We employed multiple CNN-based models, including VGG16, MobileNetV2, EfficientNet, ConvNeXt, ResNet101, and Vision Transformer (ViT), and evaluated their performance when paired with KAN. Our experiments demonstrate that KAN could achieve high accuracy with fewer training epochs and parameters, significantly outperforming traditional MLPs. Specifically, ConvNeXt paired with KAN showed the best performance, achieving 94% accuracy in the first epoch, which increased to 96% and remained consistent across subsequent epochs. Utilizing the EuroSAT dataset, which involves satellite images across ten classes, provided a robust testbed for our methodology. These results suggest that KCN has the potential to significantly impact the RS field by utilizing the advanced capabilities of KAN and CNN models, offering a promising alternative for efficient image analysis.

## Citation
```bibtex
@misc{cheon2024kolmogorovarnold,
      title={Kolmogorov-Arnold Network for Satellite Image Classification in Remote Sensing}, 
      author={Minjong Cheon},
      year={2024},
      eprint={2406.00600},
      archivePrefix={arXiv},
      primaryClass={id='cs.CV' full_name='Computer Vision and Pattern Recognition' is_active=True alt_name=None in_archive='cs' is_general=False description='Covers image processing, computer vision, pattern recognition, and scene understanding. Roughly includes material in ACM Subject Classes I.2.10, I.4, and I.5.'}
}
