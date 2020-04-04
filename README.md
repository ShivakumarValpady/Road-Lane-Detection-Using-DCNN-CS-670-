# Computer Vision(CS670)
Project Repository

Abstract:

Lane detection is an important feature for autonomous vehicles and advanced driver support systems in driving scenes. For autonomous vehicles, lane identification is extremely important. Many sophisticated lane detection methods have been proposed in recent years. Many approaches use lane boundary information to locate the vehicle on the street. Convolutional Neural Networks (CNNs) are, like many other computer vision-based activities, the state-of-the-art software for defining lane boundaries. Some techniques, however, concentrate on detecting the lane from a single image and often lead to unsatisfactory results when coping with some extremely bad situations such as heavy shadow, severe mark degradation, serious vehicle occlusion, etc. 

As we know, lanes are continuous line structures on the road. The lanes that cannot be predicted accurately in one current frame may potentially be inferred out by gathering information from previous frames. Hence, lane detection by using multiple frames of a continuous driving scene is proposed by using a hybrid deep architecture of combination of CNN and Recurrent Neural Network (RNN). The idea is to extract features of continuous images using CNNs and these features of multiple frames, holding the properties of time-series, are then fed into RNN block for feature learning and lane prediction. To increase the accuracy of the obtained model, smoothing techniques are implemented. TuSimple lane detection dataset is used for training and testing.

    Index terms - Convolutional Neural Networks, LSTM, lane detection, semantic segmentation.

Dataset - https://github.com/TuSimple/tusimple-benchmark/issues/3

PyTorch = 1.3.1
