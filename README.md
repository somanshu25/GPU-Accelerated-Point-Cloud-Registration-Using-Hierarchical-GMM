**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Final Project: Point Cloud Registration Using Gaussian Mixure Models**

* SOMANSHU AGARWAL [LinkedIn](https://www.linkedin.com/in/somanshu25)
* SRINATH RAJAGOPALAN [LinkedIn](https://www.linkedin.com/in/srinath-rajagopalan-07a43155)
* DHRUV KARTHIK [LinkedIn](https://www.linkedin.com/in/dhruvkarthik/)

![](https://github.com/somanshu25/CIS565_Final_Project/blob/master/img_gmmreg/GMM_waymo.gif)

The Gaussian Mixure Models are implemented on Stanford Bunny for visualizing GMM in 3D point cloud data. The below two gifs show for 100 and 800 components respectively.

100 Components             |  800 Components
:-------------------------:|:-------------------------:
![](img_gmmreg/bunny_100_Components.gif)| 		![](img_gmmreg/bunny_800_Components.gif)

The performance analysis of CPU and GPU implmentations are shown below:

![](https://github.com/somanshu25/CIS565_Final_Project/blob/master/img_gmmreg/performance_analysis_1.png)
![](https://github.com/somanshu25/CIS565_Final_Project/blob/master/img_gmmreg/performance_analysis_2.png)

The above graphs mention that GPU performance improves with respet to our CPU implementation of point cloud registration.
