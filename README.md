### **Fill-UNet：Extended Composite Semantic Segmentation**
The code and datasets used in the paper "Fill-UNet：Extended Composite Semantic Segmentation" are provided here. Detailed code will be made available at this location once the paper is officially accepted for publication.
## **Contributions**
1. We reexamine the semantic segmentation problem from the perspective of U-Net's U-shaped encoder-decoder construction and propose an expanded composite semantic segmentation model, termed Fill-UNet.
2. We demonstrate that the deep semantic feature information of the U-Net model can be further leveraged. By introducing two mechanisms, SCFAM and MSFA, and incorporating the Transformer structure, we achieve more precise pixel segmentation in images.
3. Our model is compared with other leading semantic segmentation techniques on various segmentation benchmarks, demonstrating superior segmentation performance. These benchmarks include the Pascal VOC2012, Cityscapes, and CamVid datasets.
## **Dataset**
PASCAL VOC 2012 is a large-scale dataset containing various images and object categories. It comprises 20 foreground categories and 1 background category. The training set and validation set contained 10,341 and 1,149 images, respectively. The training set is utilized for model training, whereas the validation set is employed to assess the model's effectiveness. Our ablation experiments were conducted on its validation set. **Download link**:http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
Cityscapes comprises 5,000 high-resolution images from 50 cities, including finely annotated semantic segmentation labels. These images are taken under diverse weather and lighting conditions to guarantee algorithm robustness. The dataset comprises 5,000 meticulously annotated images split into training, validation, and test sets, containing 2,975, 500, and 1,525 images, respectively. The image resolution is 1024×2048. **Download link**:https://www.cityscapes-dataset.com/
The Cambridge-driving Labeled Video Database (CamVid) is a video dataset used for computer vision and scene understanding research. Primarily focused on semantic segmentation tasks, this dataset comprises short video sequences captured by onboard vehicle cameras, with detailed semantic segmentation annotations provided for each frame. The dataset consists of 701 images categorized into 11 classes. Among these, 367 images are allocated for training, 101 for validation, and 233 for testing, all at a resolution of 720×960 pixels. The CamVid dataset covers diverse scenes of urban and campus roads, including various road types, traffic conditions, and pedestrian activities. This diversity allows the dataset to comprehensively represent real-world scenarios, aiding in studying model generalization in complex environments. **Download link**:https://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
Pascal Context comprises 60 primary categories and 540 secondary categories, with a total of 10,103 images, including 4,998 images in the training set and 5,015 images in the validation set. Nearly every pixel in the dataset's images is assigned a category label, significantly enhancing the granularity of annotations. This level of detail makes the dataset particularly suitable for studying contextual relationships in complex scenes. Additionally, the dataset features numerous instances of occlusion, intricate backgrounds, and multi-object scenarios, providing a challenging benchmark for evaluating high-difficulty semantic segmentation algorithms. **Download link**:https://cs.stanford.edu/~roozbeh/pascal-context/
## **Evaluation metric**
For the three datasets mentioned above, this paper adheres to standard evaluation metrics, namely, the mIoU, aAcc, mAcc, and FPS.
## **Implementation details**
The comparative experiments in this paper utilize MMSegmentation for implementation. Unless specified otherwise, the following settings are used: (1) Training is conducted on a single NVIDIA 3080Ti GPU, with a batch size of 8 and a learning rate (lr) of 0.001. The optimizer used is Adaptive Moment Estimation (Adam), with the momentum parameter set to 0.9. (2) The experimental environment was standardized with the Ubuntu 18.04 system, CUDA v11.3, cuDNN v8.2.1, and PyTorch 1.12.1. (3) MMSegmentation employs mixed-precision training to reduce GPU memory usage, thereby accelerating training.
## **Code Description**
The detailed code will be updated after the paper is published.
