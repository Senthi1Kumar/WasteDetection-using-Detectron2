# <center><strong>WasteDetection-using-Detectron2</strong></center>

## Introduction
  Modern waste management is well known to all, but unfortunately neglected by many who practice waste segregation to address the problems caused by improper waste disposal. Illegal dumping is an ongoing problem in many urban communities around the world. Odours and pollutants caused by abandoned household items, dumped garbage and construction debris are ruining cities and threatening the well-being of citizens. To curb illegal dumping, some urban areas are planning network-based voluntary reporting systems and security camera-based monitoring systems. However, these methods require manual observation and detection, are vulnerable to false alarms, and are costly. Garbage is a global problem that affects all living things.
  
  To automate the recycling process, it is important to propose a smart framework that enables effective waste sorting. Due to the large number of objects detected in a limited amount of time, the use of object detection software in waste sorting is a worthwhile practice as opposed to traditional recycling strategies. Traditional approaches rely on human labour and tend to fail to separate waste for recycling. The Deep Learning techniques have been effectively applied in a variety of areas, including, medical imaging, autonomous driving, and many industrial environments. It gives amazing results in object identification problems. Applying these technologies to waste sorting can increase the amount of recycled materials, make everyday life more convenient for ordinary people and make industries more efficient.

Thus, this project is aimed at planning and developing up a framework with a deep learning approach that can be effectively used for waste segregation. The image will be recognized by utilizing the concept of a convolutional neural network and with the help of the state-of-the-art object detection algorithms that identifies wastes from their shape, colour, dimension, and size. This technique automatically will help the system to learn the pertinent features from the sample images of the trash and consequently recognize those features in new images. By using the strategy of convolutional neural networks, garbage will be classified into different classes. The strategy utilized for this characterization is with the assistance of PyTorch and Mask R-CNN technique. Through this technique, bounding boxes segmentation masks are made on the recyclable waste demonstrating which of the 60 different classes, the waste falls into. <strong>The main objective of this study is to develop software to detect types of recyclable materials in trash bins and check for possible contamination (non-recyclable materials), which would ultimately reduce human effort in waste segregation and expedite the entire process.</strong>

-----

## Process Flow 

![Mini Project_processFlow](https://user-images.githubusercontent.com/89689985/209521459-ae5debfd-2374-4c19-9323-293dce572794.jpg)

-----

## Result

- In this project, the outcomes of the created model will be examined. The created model works reasonably on the test data. Out of 150 images, 60 – 70  images were quite accurately got predicted by the model which concludes that the Average Precision rate of the model is near about 6.7%.
- The model quite precisely classifies the type of waste materials by detecting and segmenting the type of objects.

Model's prediction on unseen images
![bottle2op](https://user-images.githubusercontent.com/89689985/209522598-f2c8022f-8cc6-4f36-a49b-789337a01c88.png)
![bagsop](https://user-images.githubusercontent.com/89689985/209522643-e0ba7d60-26ee-49f6-8f66-03da03c36a49.png)
![op-foodcontainer](https://user-images.githubusercontent.com/89689985/209522797-2a8ec9de-19c0-4fe3-80c5-91e3c6675c3b.png)


-----

## Future Work

- To use multiple publicly available waste datasets and also try to build own dataset from home, streets, garbage bins, road sides, ponds, lakes, drainages systems, waste management centres 
- To implement with other object detection algorithms such as YOLO, Single Shot Detector, RetinaNet, and libraries such as ImageAI, GluonCV, Yolov3_TensorFlow
- And also to implement it with robotic arms like to classify the wastes in the Waste Management Centres or make surveillance system that identifies a person who throws a waste/trash in clean surroundings. 


-----

## Referencees

1. Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun: Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. arXiv:1506.01497
2. Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick; Mask R-CNN arXiv:1703.06870
3. Ying Liu, Zhishan Ge, Guoyun Lv, Shikai Wang; Research on Automatic Garbage Detection System Based on Deep Learning and Narrowband Internet of Things ResearchGate, DOI:10.1088/1742-6596/1069/1/012032
4. SylwiaMajchrowska, AgnieszkaMikołajczyk, MariaFerlin, ZuzannaKlawikowska, Marta A.Plantykow, ArkadiuszKwasigroch, KarolMajekd; Deep learning based waste detection in natural and urban environments ScienceDirect https://doi.org/10.1016/j.wasman.2021.12.001
5. FAIR’s blog on Detectron2- https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-/
6. Detectron2 - https://github.com/facebookresearch/detectron2
7. TACO - http://tacodataset.org/
8 TACO dataset - https://github.com/pedropro/TACO
9. Roboflow platform - https://roboflow.com/
10. R-CNNs - https://d2l.ai/chapter_computer-vision/rcnn.html#r-cnns
-----
