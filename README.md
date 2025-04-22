## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![Screenshot 2025-04-22 094655](https://github.com/user-attachments/assets/52fca36d-6162-43ed-ac27-4a6f7dbfc47f)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![Screenshot 2025-04-22 094759](https://github.com/user-attachments/assets/6628baf7-3de0-45ce-b44f-8a487d0c61d2)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![Screenshot 2025-04-22 094833](https://github.com/user-attachments/assets/a903acdf-68a6-4449-8768-3250ddd9024d)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![Screenshot 2025-04-22 094916](https://github.com/user-attachments/assets/588bde87-eeb3-45ef-af8b-bc7ba09d0751)

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![Screenshot 2025-04-22 095002](https://github.com/user-attachments/assets/1713f556-2a3c-4337-ae2f-f51778048d96)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![Screenshot 2025-04-22 095046](https://github.com/user-attachments/assets/1db573a5-e212-4701-a9e9-e00fe8693011)

```
pip install --upgrade category_encoders
```
![Screenshot 2025-04-22 095127](https://github.com/user-attachments/assets/b749a45e-8f7f-4b3a-9a53-c6ee474b64df)

```
from category_encoders import BinaryEncoder
be=BinaryEncoder()
df=pd.read_csv("/content/data.csv")
df
```
![Screenshot 2025-04-22 095207](https://github.com/user-attachments/assets/ab2bf15d-be65-4ba3-9b5c-0c2e63f45856)

```
be= BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb1
```
![Screenshot 2025-04-22 095238](https://github.com/user-attachments/assets/8f427638-fc34-48df-8f9a-5d14db22ddfa)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![Screenshot 2025-04-22 095311](https://github.com/user-attachments/assets/c3389d3c-b482-47fa-8d61-fc6803cc945e)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![Screenshot 2025-04-22 095359](https://github.com/user-attachments/assets/3ac406e7-2206-43c3-bf5e-eee334878d84)

```
df.skew()
```
![Screenshot 2025-04-22 095429](https://github.com/user-attachments/assets/ae505e91-61ae-4094-89f5-d3d41fb1f8c0)

```
np.log(df["Highly Positive Skew"])
```
![Screenshot 2025-04-22 095508](https://github.com/user-attachments/assets/fd77b68c-8f43-42de-b8c5-cbbdd837aaba)

```
np.reciprocal(df["Moderate Positive Skew"])
```
![Screenshot 2025-04-22 095538](https://github.com/user-attachments/assets/c683d1b0-46f7-4165-ac15-cdef23174273)

```
np.sqrt(df['Highly Positive Skew'])
```
![Screenshot 2025-04-22 095608](https://github.com/user-attachments/assets/766f2da1-ddcb-4979-b5c6-1808b376a8f1)

```
np.square(df["Highly Positive Skew"])
```
![Screenshot 2025-04-22 095700](https://github.com/user-attachments/assets/e8a3001e-64e5-43ac-861b-64105419aa9f)

```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df['Highly Positive Skew'])
df
```
![Screenshot 2025-04-22 095740](https://github.com/user-attachments/assets/927d810d-28d4-4445-a901-cb896e6bb4c4)

```
df.skew()
```
![Screenshot 2025-04-22 095815](https://github.com/user-attachments/assets/2e24c3b6-c9c8-4c06-a5a4-e9aafcb830d0)

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![Screenshot 2025-04-22 095854](https://github.com/user-attachments/assets/527dc31f-b0cb-47c6-ba75-1e2f79d5e441)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![Screenshot 2025-04-22 095959](https://github.com/user-attachments/assets/9f6a4573-8b28-481b-907e-116e91b47b3c)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2025-04-22 100038](https://github.com/user-attachments/assets/93cfbc3d-eca1-48fe-ba75-d802033cd729)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![Screenshot 2025-04-22 100107](https://github.com/user-attachments/assets/a1d40a8c-08aa-474a-88a6-43d03713950f)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2025-04-22 100148](https://github.com/user-attachments/assets/7a5cdec4-cf85-4efa-9f20-b917bf7a714c)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![Screenshot 2025-04-22 100246](https://github.com/user-attachments/assets/a202df8d-7ae2-4e37-a21f-485240f84297)

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![Screenshot 2025-04-22 100346](https://github.com/user-attachments/assets/16670c29-fb7c-4c35-9f0d-55acf8c27518)

```
dt=pd.read_csv("/content/titanic_dataset (2).csv")
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
```
![Screenshot 2025-04-22 100422](https://github.com/user-attachments/assets/73dfb3c4-06ac-462a-b87d-079ed53c14f6)

```
sm.qqplot(dt['Age_1'],line='45')
plt.show()
```
![Screenshot 2025-04-22 100449](https://github.com/user-attachments/assets/e047983f-ee10-4f10-9eba-12dd3a2f85c2)












# RESULT:
```
  Thus we performed Feature Encoding and Transformation process
```
       
