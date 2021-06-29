[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fgabbygab1233%2FHeart-Failure-Predictor-Application&count_bg=%23231920&title_bg=%23F76767&icon=&icon_color=%23E7E7E7&title=Heart+Failure+Predictor&edge_flat=false)](https://hits.seeyoufarm.com)

# Heart failure prediction 
According to [Wikepedia](https://en.wikipedia.org/wiki/Heart_failure) Heart failure (HF), also known as congestive heart failure (CHF), (congestive) cardiac failure (CCF), and decompensatio cordis, is when the heart is unable to pump sufficiently to maintain blood flow to meet the body tissues' needs for metabolism. Signs and symptoms of heart failure commonly include shortness of breath, excessive tiredness, and leg swelling. The shortness of breath is usually worse with exercise or while lying down, and may wake the person at night. A limited ability to exercise is also a common feature. Chest pain, including angina, does not typically occur due to heart failure.



<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Right_side_heart_failure.jpg/350px-Right_side_heart_failure.jpg" />
</p>


# Dataset
The “Cleveland heart disease dataset 2016” is used by various researchers and can be accessed from online data mining repository of the University of Cal-
ifornia, [Irvine](https://archive.ics.uci.edu/ml/datasets/heart+disease). The Cleveland heart disease dataset has a sample size of 303 patients, 76 features, and some missing values. During the analysis, 6 samples were removed due to missing values in feature columns and leftover samples size is 297 with 13 more appropriate independent input features, and target output label was extracted and used for diagnosing the heart disease. The target output label has two classes in order to represent a heart patient or a normal subject. Thus, the extracted dataset is of 297 ∗ 13 features matrix.

### Attributes information:
![](https://user-images.githubusercontent.com/26917380/123854856-78043300-d934-11eb-902a-4f3e8ee028c2.png)

# Demo
Live demo: https://heartfailure-predictor.herokuapp.com/


![](https://user-images.githubusercontent.com/26917380/123855128-c44f7300-d934-11eb-9517-80a7f8702f10.png)
![](https://user-images.githubusercontent.com/26917380/123855132-c6b1cd00-d934-11eb-97c6-157d9194e391.png)

# Clone project 
```shell
$ git clone https://github.com/YasinEnigma/Heart-Failure-Decision-Support-System/
$ cd Heart-Failure-Decision-Support-System
$ pip3 install -r requirements.txt
$ streamlit run app.py
```


# References
* https://www.hindawi.com/journals/misy/2018/3860146/
* https://www.sciencedirect.com/science/article/abs/pii/S0957417416305516
* https://github.com/gabbygab1233/Heart-Failure-Predictor-Application
* https://archive.ics.uci.edu/ml/datasets/heart+disease
