# Smart Wearables
This Smart Wearable project has been realized in the frame of the AI&Art exhibition at the University of Luxembourg. Throughout my job, I have worked in collaboration with the artist Mrs. Yolanda Spinola and under the supervision of Mrs. Sana Nouzri.

This project aims to generate a 3D avatar of a user placed in front of a webcam. This avatar should be dressed with various clothing items generated from a training set composed of haute couture garments. The final result (generated body with clothes) should be stored in the local storage.

# Table of contents
- [Smart Wearables](#smart-wearables)
- [Table of contents](#table-of-contents)
- [Abstract](#abstract)
- [How to run?](#how-to-run)
- [Final result & Future work](#final-result--future-work)
- [Development details](#development-details)
  * [*3D Generation*](#3d-generation)
  * [*Garment model*](#garment-model)
- [FrankMoCap Modifications](#frankmocap-modifications)
- [Models integration](#models-integration)
- [OpenCV Model](#opencv-model)
  * [*Overlaying garments*](#overlaying-garments)
    + [Body processing](#body-processing)
    + [Other garments](#other-garments)
    + [Final](#final)
  * [*Garments Dataset*](#garments-dataset)
  * [*Backgrounds Dataset*](#backgrounds-dataset)

# Abstract
This Smart Wearable project takes as input either a live webcam stream (a classic RGB camera, without a depth sensor), a local video file, or an image file. The program will then generate a 3D avatar of the user (in front of the webcam or on the video/images) and display it on screen. This generation is performed thanks to the [Frankmocap](https://www.notion.so/Concept-d08dc52bc69649b5ab1c062b8d59fa05) 3D motion capture system developed by Facebook. Frankmocap uses the SMPL model to generate the body shapes and joints. Those joints are sent to a custom OpenCV model in charge of overlaying 2D clothing items on top of the body. 

For each simulation, a random garment will be selected for the upper and lower body. A virtual button displayed in the top left of the image allows the user to be dressed with new random garments from the dataset. Backgrounds can also be displayed to give the illusion the user is in a real fitting room for example. After each simulation, the result of the dressed body is stored as a sequence of images in a custom directory.

The programming language used for this project is Python.
The main frameworks/models/libraries used are Apache MXNet, SMPL, and OpenCV.

# How to run?
First of all, please note that this project uses the CUDA Toolkit and therefore a CUDA-enabled GPU is required (more information on that [here](https://developer.nvidia.com/cuda-gpus)).

For the setup of this project, please go to the [INSTALL.md](https://github.com/gillet-thomas/smartwearables/blob/main/docs/INSTALL.md) in the docs folder and follow the steps for “Installing Body Module Only”.

It is recommended to run this program with an external webcam plugged in. Those usually offer better video quality and a wider angle. If you still want to use your default webcam, please change the VideoCapture argument from 1 to 0 in the mocap_utils/demo_utils.py file, line 110.

Once all the setup is done, run the project with the following command:

```latex
python -m demo.demo_bodymocap --input_path webcam --out_dir ./mocap_output
```

# Final result & Future work
<img src="https://github.com/gillet-thomas/smartwearables/blob/main/readme/demo.gif">

On [this drive](https://drive.google.com/drive/folders/1s9CqlsL0hWcfKQkC3LKzutG2Y8IMtTd_?usp=sharing), one can find 4 examples of the final result where both the input webcam video and the output of the model are combined. 
One can also find a demo version of this program implemented in **[Google Colab](https://colab.research.google.com/drive/1joV9aKNkk9aehK5z_6Kj869qYznOfkWb)**.


This project has been realized in only three weeks. Due to this small timeframe, a lot of features couldn’t be implemented. Here are some of the main next steps to be realized:

- First of all, the provided dataset is composed of over 1200 haute couture garments. My OpenCV model only has a dataset composed of 45 garments for both the upper and lower body. This dataset could be augmented to dress the avatars with a wider range of clothes.
- As said, the OpenCV model expects a very specific format for the different garments. An improvement would be to create a machine learning model that could recognize the different garment parts, segment them and name them automatically.
- Finally, the biggest upgrade would be to use the full haute couture dataset. This would give more realistic outputs. However, one must first find a way to generate 3D garments from these images. Another machine learning model could then be trained to generate new haute couture garments with different textures/shapes/colors.
- On a side note, it could be nice to deploy this project as a .exe file to have a standalone file that could be run anywhere.

---

# Development details
In order to create this project, 3 distinct phases have been defined.

1. *Generate an avatar*
2. *Dress it with clothes from the current dataset*
3. *Generate new clothes with new style and textures based on the artist’s collection*

Here were the different constraints I had to consider for the choice of my model. First, the final result should have real-time performance. It means that the whole body generation and avatar dressing should be done in a timely fashion. Then, the available input consists of a sequence of images coming from an RGB camera. No depth information is assumed. Finally, the available garments samples were classic JPG/PNG images. No 3D cloth segmentation data were given.

Only the two first phases could have been implemented in this current version of the project.

## *3D Generation*

I have first tried a dozen of different AI Models but a lot of problems have been encountered with most of them. Those were mainly due to deprecated python/libraries versions or to un-updated projects presenting mismatch between the current libraries updates and the older versions they were using. The other main problem encountered was missing datasets. When projects had an up-to-date setup, most of them failed to be tested due to unavailable dataset (broken link, unauthorized access, …). Finally, the very few models that were not subject to the two aforementioned issues were showing very slow performance. Either because of the nature of the model itself or because the cloth semantic segmentation was too slow.

The solution came rather randomly. I first found the SPMLicit project that could generate a 3D avatar and dress it with custom 3D clothes. The expected input is an image with the SMPL estimations and the clothes segmentation. Since I only had 2D garments images as input I knew this model would not be usable with my current data. However, I still installed the SMPL estimation model out of curiosity. This model called [FrankMoCap](https://github.com/facebookresearch/frankmocap) turned out to be the perfect solution. It generates 3D body shapes in real-time from a webcam stream, a video, or images.

Therefore, this model has been installed and used throughout the whole project. Frankmocap solved the 3D body generation problem with a single RGB image as input. Now the next step was to create a model that could dress the avatar.

## *Garment model*

There were two different kinds of models that could be used here. Either generate 3D clothes from the 2D sample data images and dress the avatar in real-time. This solution would be the most realistic one since the generated garment would perfectly fit the avatar even when it would move, turn, tilt, bend,… itself. However, no guarantee can be provided regarding the real-time performances of such models. The other solution would be to use the python library OpenCV to programmatically add the different garments as 2D layers on top of the generated avatar.

After having tried many machine learning models that could be used to convert the garments samples into 3D, I decided to use the 2D OpenCV alternative. Indeed, many models were again too slow or simply not working. That’s the reason why I didn’t continue with such 3D models.

I have first searched for projects performing body joints estimation with OpenCV. The best one I have found is called [VirtualDresser](https://github.com/MRishiK99/VirtualDresser) project and uses Apache MXNet and Gluon for the pose estimation.

I have used this project as the underlying base for my model which I have then improved to fit my needs. 


# FrankMoCap Modifications

Two functionalities have been added to the Frankmocap (FMC) project: a virtual button and a random background generation.

The virtual button uses a timer and whenever the user puts one of his hands for 2 seconds over this button, a parameter will be sent to the dress method of VirtualDresser (VD) to regenerate new clothes.

The backgrounds are taken at random from the local storage and resized to match the input stream shape. Please note that the backgrounds and avatars will always be resized to match the input frame height and width.

# Models integration

A package has been created from the OpenCV model to integrate it into the FMC project.

Technically, for each frame of the input (webcam/video), FML will estimate the 3D pose, create the body joints, and returns the input frame with the 3D avatar. This frame is then passed to my VD model.

I noticed FMC must render the resulting frame on screen, that’s the reason why two windows will be displayed: one with the bare avatar and one with the dressed avatar.

The integration of those two models resulted in a 3/4 fps output. This is unfortunately not enough to achieve real-time performance. In order to improve the output speed, I have removed all the unnecessary code from both projects (this improved the frame rate by 1/2 fps).

The biggest upgrade, however, has been to reuse the body joints from FMC instead of recomputing them using MXNet in VD. Now when the VD model is called inside FMC, two parameters are given: the frame with the generated avatar and the list of body joints found by SMPL. These improvements resulted in 10 to 15 fps on average.

# OpenCV Model

In the VirtualDresser folder, one can find the [embed.py](https://github.com/gillet-thomas/smartwearables/blob/main/VirtualDresser/embed.py) module which is the final version of the VD project used in the FMC model.

## *Overlaying garments*

In the intialization part of this module, some global variables are created and initialized, the upper and lower body garments are picked randomly, and finally, the body joints are set.

In order to select garments at random, a precise naming is expected. For the upper body garments, each of them has at least a “body” file. The algorithm will therefore take one random file containing “body” and then select all the other related cloth items (sleeves for a t-shirt, sleeves and forearm for a pullover). For the lower body, no files are segmented, they are all full cloth items. Therefore the algorithm just selects one random file with the tag “full”.

Then all those garments are preprocessed to be later applied as a 2D layer on top of the avatar. 

Overall, the preprocessing is the same for each garment. I will detail it in detail for the body and only explain the difference (if any) for the other cloth items.

### **Body processing**

First, a test is performed to ensure a file path has been found for the body and that all the body joints are within the boundaries of the frame. The image is then read from its path using OpenCV.

There are 3 main steps for each garment: resize, rotate, and get its new location.

#### 1. **Resize**

Two distances are computed: the distance between the right shoulder and the left shoulder (the body width) and the distance between the neck and the lower mid which is the middle point between the two hips (the body height). The scaling ratio is also set, it has a value of 1.2 by default (an arbitrary number that works well for all files).

However, some issues have been encountered with the dresses since those files can’t be cropped near the body part. Most of the time the lower part of the dress is wider than the upper part (the body width). To fix this I have implemented a function (getGarmentRatio) that will read the matrix representing the file and compute the number of colored pixels relative to the total width of the image. This will indicate the ratio between the body width and the total width of the image. One can notice that I compute this ratio on the line 10% lower than the image top. This is to prevent cropping errors where some gap would have been left between the top of the dress and the image border.

In the end, the scaling ratio is assigned the value returned by this function if the file is a dress, otherwise the default 1.2 value.

Both distances (body height and width) are scaled by that scaling ratio. Finally, the image is resized using those two adjusted distances. This ensures that the garment will always have the same proportions as the body of the avatar.

#### 2. **Rotation & new location**

Due to the different possible scaling ratios of the body part, the rotation of that cloth item has been very tricky (the body is the only garment that can have a varying scaling ratio).

First of all the rotation angle between the upper mid (middle point between the shoulders) and lower mid is computed. This gives an overall idea of how tilted the person is.
This angle is converted into degrees and a fixed value of 90 is added to have positive angle values.

The general idea is to rotate the image by a certain angle using the imutils.rotate_bound function.
This function allows not to crop the image if, after the rotation, it would go beyond the original edges. The drawback is that rotate_bound enlarges the image shape by an unknown value. It means that once the rotation has been performed I can no longer rely on the image corners to position it.

The workaround I found is to rotate the image, compute the center coordinate of the new rotated image and translate that point to the center of the body joints. This is exactly what the function getRotatedPoints do. The given image will be translated by (x,y) where x and y are the respective difference between the center coordinate of the rotated image and the rotation point I want my image to be at (usually the center of the body joints).

The problem for the body is that there is no way to get a precise center point once the user starts to tilt. The middle point defined by the half distance between the upper mid and lower mid results in the garment not fitting the shoulders correctly.

Here, the rotation center is computed before the garment rotation to rely on the original image dimensions. The final rotation Y point is computed based on the rotation angle and the final rotation X point is computed based on the scaling ratio and on whether the garment middle point is below the hips or not.

Please note that for all the other garments, this computation happens after the image has been rotated and the rotation point is always the middle point between the 2 body joints.

![body joints](https://github.com/gillet-thomas/smartwearables/blob/main/readme/body_joints.png)

### **Other garments**

For the sleeves, the only difference is that the adjusted width is not the product of the sleeve width by the scaling ratio but rather the adjusted height divided by 3. This is because the body joints can’t retrieve the arm width. Therefore, after some experiments, I found that setting a sleeve with a length 3 times bigger than its width produces a good result.

For the forearms, the height used in the resize function is the adjusted height of the sleeves.

Finally the sleeves and forearms always have the default 1.2 scaling ratio applied.

For the lower body, the same logic is applied except that no rotation is performed.
There is, however, one extra condition for the lower body garments to be set: the upper body must not be a dress. Indeed, it doesn’t make very much sense to wear both a dress and a pair of shorts/trousers.

The same getGarmentRatio function is used to find the scaling ratio to be used. This is because the length of the waist on those files is usually smaller than the width of the image.

<img src="https://github.com/gillet-thomas/smartwearables/blob/main/readme/shorts.png" width="280">

The adjusted height is the distance between the foot and the hip if the file is a pair of trousers, otherwise, the distance between the knee and the hip for skirts and 

### **Final**

Finally, all the garments are overlayed on the input frame using the overlay_image_alpha method.
The overlaying order is important here. First the pants, then the forearms, sleeves, and finally the body. This produces a better final result with this particular stacking of garments layers.

## *Garments Dataset*

The current dataset consists of t-shirts, pullovers, dresses, pants, shorts, and skirts. The dresses and lower body garments can be used as such, however, t-shirts and pullovers must be segmented into different files. Here is the list of all the different files expected for each garment type:

| Tshirt |  tshirt_body_[id], tshirt_rsleeve_[id], tshirt_lseleeve_[id] | Trousers | trousers_full_[id] |
| --- | --- | --- | --- |
| Pullover | pullover_body_[id],pullover_rsleeve_[id], pullover_lseleeve_[id], pullover_rforearm_[id], pullover_lforearm_[id] | Shorts | shorts_full_[id] |
| Dress | dress_body_[id] | Skirt | skirt_full_[id] |

If one wants to augment the dataset with new data here are the conventions to follow:

- All files must be in the PNG format (with a transparent background) and cropped as close as possible to the garment’s edges.
- The sleeves and forearms must be vertical rectangles.
- The id for all the segmented parts of a single garment must be the same.
- The garment’s parts must be named as shown in the table above.

## *Backgrounds Dataset*

One can easily add new backgrounds to the program by adding them to the data/backgrounds/ folder. No naming convention is assumed for the files. Please note that the background images are expected to be in JPG format.

