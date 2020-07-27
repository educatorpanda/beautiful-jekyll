---
layout: post
title: Everything you need to know about Linear Regression!
subtitle: Machine Learning Episode-1.1
gh-repo: educatorpanda/educatorpanda.github.io
gh-badge: [star, follow]
comments: true
readtime: true
tags: [Machine Learning]
comments: true
---

Linear Regression is a powerful statistical tool that is used to determine the relationship between two quantitative variables.

> "Somewhere between Simple Linear Regression and Deep Neural Networks we grow up to become a Data Scientist" 

1. The first variable (also denoted as '**X**') is called the independent variable.
2. The second variable (also denoted as '**Y**') is called the dependent variable.

The independent variable is the variable the experimenter changes or controls and is assumed to have a direct effect on the dependent variable. For example: Driving speed and gas mileage — as driving speed increases, you'd expect gas mileage to decrease. So, here the driving speed is the independent variable which is controlled by the driver (experimenter) and it is assumed to have a direct effect on the dependent variable which is the gas mileage.

### In this article, we will be covering the following topics: 
* [Introduction to Simple Linear Regression](#introduction-to-simple-linear-regression)
* [Residuals and Cost Function](#residuals-and-cost-function)
* [Least Square Method](#least-square-method)
* [Coefficient of Determination (R<sup>2</sup>)](#coefficient-of-determination)
* [Multiple Linear Regression](#multiple-linear-regression)
* [Coding (in python) from scratch](#python-code)
* When can you use Linear Regression model 


### Introduction to Simple Linear Regression 
As you may have studied in your high school, the relationship between the two different units of Temperature, degrees Fahrenheit (F) and degrees Celsius (C) is known to be:  

![\Large F= \frac{9}{5}C+32](https://latex.codecogs.com/gif.latex?F%3D%20%5Cfrac%7B9%7D%7B5%7DC&plus;32){: .mx-auto.d-block :}

![Temperature](/assets/img/CelciusFahrenheit.PNG){: .mx-auto.d-block :}

That is, if you know the temperature in degrees Celsius, you can use this equation to determine the temperature in degrees Fahrenheit **exactly**. Such a relationship is called a deterministic (or functional) relationship. In a deterministic relationship, the equation **exactly** describes the relationship between the two variables.

However, in the world of data science, we never talk about the deterministic relationship. Instead, we are always interested in statistical relationships. So, today let us understand in detail, one of the most basic method which is used to establish the statistical relationship between two variables (an independent variable and a dependent variable) known as **Simple Linear Regression**. 

**What is Simple Linear Regression?**

Simple Linear Regression is a basic regression analysis where we have just two variables (an independent variable and a dependent variable) and based on the changes made to the independent variable (**X**), we try to predict the outcome of the dependent variable (**Y**). Let us understand this through a fictitious example.

> **Example**: This table shows the investment of the company in its business over the years. Our task at hand is to predict the investment of the company when **Years = 3.5** using Simple Linear Regression model. So how will you do that?

| Years | Investment (in M) |
| :------: |:---: |
| 0 | 2 |
| 1 | 3 |
| 2 | 5 |
| 3 | 4 |
| 4 | 6 |

The very first step is to identify the independent and the dependent variables. When we analyse our data, we can clearly find that the variable **Years** is assumed to have a direct effect on the variable **Investment**. So **Years** is the independent variable (**X**) and **Investment** is the dependent variable **Y**.

{: .box-note}
**Note:** In general, it is a good practise to visualise your data (before proceeding towards the solution) in order to get a gist of the relationship that these variables behold. So, let us plot this data on a XY graph.

This is how the plot will look like:


![Data](/assets/img/datavisualization.png){: .mx-auto.d-block :}

The core idea in Simple Linear Regression is to obtain a line that best fits the data. This regression line is designed to provide the average of **Y** for any given value of **X**. Mathematically, the equation of such a line is of the form:


![\Large y=w_{0}+w_{1}X ](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5CLarge%20y%3Dw_%7B0%7D&plus;w_%7B1%7DX){: .mx-auto.d-block :}

where **y** represents the predicted output for a given input **X**. The terms **w<sub>0</sub>** and **w<sub>1</sub>** represents the *Y-intercept* of the line (i.e. the point where the given line intersects the Y-axis) and *Slope* of the given line respectively. However, for the case of one input variable and one output variable, it was relatively easy to label the coefficients **w<sub>0</sub>** and **w<sub>1</sub>** as the *Y-intercept* and the *Slope* of a line, because it is easier to visualize them. But, when we include multiple variables into the picture, things become more complex and simply designating the coefficients as *Y-intercept* and *Slope* is not a good idea. Therefore, we should come up with something more general. This problem can be resolved by collectively calling the terms **w<sub>0</sub>** and **w<sub>1</sub>** as the weights attached to the input variables. 

The above equation can also be rewritten in this fashion: 

![\Large y=w_{0}.1+w_{1}.X ](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5CLarge%20y%3Dw_%7B0%7D.1&plus;w_%7B1%7D.X){: .mx-auto.d-block :}

Therefore, now you can consider the above model having two input variables (**1** and **X**) and one output variable (**y**) and the weights associated with this input variables are **'w<sub>0</sub>'** and **'w<sub>1</sub>'** respectively.

{: .box-note}
**Note:** In machine learning terminology, the weight associated with the additional input variable (**1**) is also known as **Bias**.

Thus, the general representation of the predicted output (for any ML algorithm) can be represented as follows:

![Data](/assets/img/bias.PNG){: .mx-auto.d-block :}

In the figure given below, we find that there can be multiple lines with which we can fit the given data. But the best fit line is the one for which the total prediction error for all the data points is as small as possible, i.e. we find the optimum value of the weights and the bias such that the total error associated with our prediction is ***minimum***.


![Data2](/assets/img/datavisualization2.png){: .mx-auto.d-block :}

### Residuals and Cost Function

Taking about the errors, let us understand the concept of **Residual** and Total prediction error (or the **Cost Function**)

1. **Residual**

Out of the given multiple lines in the above figure, let us arbitrarily choose a line and call it as ***L*** such that ***Line L: y = w<sub>0</sub> + w<sub>1</sub>X*** becomes the regression line. So, a Residual is simply the vertical distance (denoted by the red line, in the figure below) between a data point and the regression line ***L***. Each data point has one residual. In our example, we have 5 data points, so there will be 5 residuals in total. They are negative if they are above the regression line and positive if they are below the regression line. If the regression line actually passes through the point, the residual at that point is zero.


![Residual](/assets/img/residual.PNG){: .mx-auto.d-block :}

We donote this residual term at the i<sup>th</sup> data point as **e<sub>i</sub>**. So, for the i<sup>th</sup> data point **(X<sub>i</sub>,Y<sub>i</sub>)**, the value of the predicted output will be **y<sub>i</sub>** such that ***y<sub>i</sub> = w<sub>0</sub> + w<sub>1</sub>X<sub>i</sub>*** and the value of the residual at this data point will be given as: 


![\Large e_{i}=y_{i}-Y_{i}](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5CLarge%20e_%7Bi%7D%3Dy_%7Bi%7D-Y_%7Bi%7D){: .mx-auto.d-block :}

2. **Cost Function**

**Cost function** in a machine learning terminology is simply a measure of how incorrect the model is in term of its ability to evaluate the relationship between **X** and **Y**. **Cost Function** quantifies the error between predicted values (**y**) and actual values (**Y**) and presents it in the form of a single real number. Thus Cost function, in layman terminology is nothing but the total prediction error. We denote this single real number by **C**. 

Now you can sense a relationship between the Residual and the Cost function. Residual denotes the error for a single data point, whereas the Cost function denotes the error for all the given data points. So, in order to develop a model which can accurately predict the output, we just need to minimize the Cost function (**C**). Depending on the problem, the Cost Function can be formed in many different ways. But for our case, it will be given as:


![\Large C=\sum_{i=1}^{N}\frac{1}{2N}\left ( y_{i}-Y_{i} \right )^{2}=\sum_{i=1}^{N}\frac{e_{i}^{2}}{2N}](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5CLarge%20C%3D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Cfrac%7B1%7D%7B2N%7D%5Cleft%20%28%20y_%7Bi%7D-Y_%7Bi%7D%20%5Cright%20%29%5E%7B2%7D%3D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Cfrac%7Be_%7Bi%7D%5E%7B2%7D%7D%7B2N%7D){: .mx-auto.d-block :}

Here **N** is the total number of data points or total number of available samples (**N** is 5 in our case). This function is same as Mean Square Error (MSE), where you first calculate the residual ***e<sub>i</sub> = y<sub>i</sub> - Y<sub>i</sub>*** for all the data points, then you square it and find the mean of all the residuals (you will have to multiply it by a factor of 0.5 also).

### Least Square Method

Now that we are familiar with the term Residuals and Cost function (or total prediction error) we can move forward towards our main objective (i.e. to find the optimum value of the weights such that the total error associated with our prediction is ***minimum***). The **Least Square Method** states that the best fit curve have a minimum sum of the squared residuals (or minimum Cost function) from the given data points. We see here that the word ***minimum*** is used. So what should be our approach now? Yes you have guessed it right! We will now be dealing with the derivatives of the function.

When we substitute the predicted output ***y<sub>i</sub> = w<sub>0</sub> + w<sub>1</sub>X<sub>i</sub>*** into the Cost function (**C**) we get:

![\large C= \frac{1}{2N}\sum_{i= 1}^{N}\left ( w_{0}+w_{1}X_{i}-Y_{i}\right )^{2}](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5Clarge%20C%3D%20%5Cfrac%7B1%7D%7B2N%7D%5Csum_%7Bi%3D%201%7D%5E%7BN%7D%5Cleft%20%28%20w_%7B0%7D&plus;w_%7B1%7DX_%7Bi%7D-Y_%7Bi%7D%5Cright%20%29%5E%7B2%7D){: .mx-auto.d-block :}

We see that our Cost function is now a function of the variable **w<sub>0</sub>** and **w<sub>1</sub>**, i.e. **C = C(w<sub>0</sub>, w<sub>1</sub>)**. So, in order to minimise this Cost function, we find the partial derivative of **C** with respect to **w<sub>0</sub>** and **w<sub>1</sub>** and equate it to zero (as shown in the figure below). 


![\large \frac{\partial C\left ( w_{0},w_{1} \right )}{\partial w_{0}} = 0 ;\hspace{0.3cm} \frac{\partial C\left ( w_{0},w_{1} \right )}{\partial w_{1}} = 0](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5Clarge%20%5Cfrac%7B%5Cpartial%20C%5Cleft%20%28%20w_%7B0%7D%2Cw_%7B1%7D%20%5Cright%20%29%7D%7B%5Cpartial%20w_%7B0%7D%7D%20%3D%200%20%3B%5Chspace%7B0.3cm%7D%20%5Cfrac%7B%5Cpartial%20C%5Cleft%20%28%20w_%7B0%7D%2Cw_%7B1%7D%20%5Cright%20%29%7D%7B%5Cpartial%20w_%7B1%7D%7D%20%3D%200){: .mx-auto.d-block :}

I will skip the derivation part and will directly provide the optimum value of the bias and weight **w<sub>0</sub>** and **w<sub>1</sub>** respectively (as given in the figure below)


![Data2](/assets/img/formulaweights.PNG){: .mx-auto.d-block :}

**Solution to the above Example**

I know the above formula looks scary. But don't be frightened, as I will show a step by step method (which can be easily followed) to calculate the optimum values of **w<sub>0</sub>** and **w<sub>1</sub>** using the above formula.

If you observe the formula carefully, all we need to calculate the weights and biases are the values of **ΣX**, **ΣY**, **ΣXY** and **ΣX<sup>2</sup>** and **N** (which is 5 in our case). That's it! Let us calculate these values with the help of the following table.  

| i | X<sub>i</sub> | Y<sub>i</sub> | X<sub>i</sub><sup>2</sup> | X<sub>i</sub>Y<sub>i</sub> |
| :------: |:---: | :------: |:---: | :------: |
| 1 | 0 | 2 | 0 | 0 |
| 2 | 1 | 3 | 1 | 3 |
| 3 | 2 | 5 | 4 | 10 |
| 4 | 3 | 4 | 9 | 12 |
| 5 | 4 | 6 | 16 | 24 |
| Sum | **ΣX=10** | **ΣY=20** | **ΣX<sup>2</sup>=30** | **ΣXY=49** |

On substituting the values of **ΣX**, **ΣY**, **ΣXY**, **ΣX<sup>2</sup>** and **N** in the given formula for **w<sub>0</sub>** and **w<sub>1</sub>**, we get

![Data2](/assets/img/weightsvalues.PNG){: .mx-auto.d-block :}

Once we know the value of the regression coefficients (or the weights and biases), the regression equation (or the equation of the predicted output) becomes: 

![\Large y=2.2+0.9X](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5CLarge%20y%3D2.2&plus;0.9X){: .mx-auto.d-block :}

Remember our goal was to predict the investment of the company when **Years = 3.5** using Simple Linear Regression model. Since you have the regression equation (**y**), using it is a snap. Just substitute the value of **X = 3.5** in the above equation of regression line to get the value of **y** (investment of the company) as 5.35M!

![\Large y=2.2+0.9 \times 3.5 = 5.35](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5CLarge%20y%3D2.2&plus;0.9%20%5Ctimes%203.5%20%3D%205.35){: .mx-auto.d-block :}

{: .box-warning}
**Warning:** When you use a regression equation, do not use values for the independent variable (**X**) that are outside the range of values used to create the equation. That is called extrapolation, and it can produce unreasonable estimates.

In our example, the **Years** (X) used to create the regression equation (**y = 2.2 + 0.9X**) ranged from 0 to 4. Therefore, only use values inside that range to estimate the investment of the company (**y**). Using values outside that range (less than 0 or greater than 4) is problematic.

### Coefficient of Determination

Since our scattered data points (**Y**) does not lie completely on the regression line ***y = w<sub>0</sub> + w<sub>1</sub>X*** (i.e., although the cost function is minimized, it is not zero), therefore a line is not a perfect explanation of the data or a perfect match to variation in **Y**. This is where the Coefficient of Determination (also known as R-squared value) comes into the picture. The Coefficient of Determination is comparing how much of true variation in **Y** is in fact explained by the best straight line provided by the regression model **y**. 

Before jumping to the mathematics, let us try to understand the concept more intuitively. As we have already seen before that the Cost function (or the Standard Error) measures the error that one commits with their estimation of the relation between **X** and **y** (regression line). Assume if we had no better tools for fitting lines to the data points, then what's the best thing that we could have done?  

We know one thing that **Mean(Y)** will stay the same, no matter what value of **X** is. It is completely independent of **X**. Therefore we would have just taken a horizontal line that goes through the **Mean(Y)**. So, in that case, ***y = Mean(Y)*** is the line (which is of the form ***y = constant***) that minimizes the Cost function (and in this case, your Cost function will simply become the **variance of Y**).

Still, we know that a constant line is the most basic model one could come up with, as a linear function, an exponential function, a quadratic function, etc. all can adapt better to data points (as they have more parameters to play with) than a line ***y = constant***. So, the **Cost function of y = Mean(Y)** can be seen as the error that is committed by fitting data points with the worst (or the most basic) model available.

For simplicity, let us denote 
* Cost function of the regression line: ***y = w<sub>0</sub> + w<sub>1</sub>X*** as **C<sub>L</sub>** 
* Cost function of the line ***y = Mean(Y)*** as **C<sub>M</sub>** 

Hence we can conclude that **C<sub>L</sub>** will never be higher than the biggest ever possible error **C<sub>M</sub>**. If we see things this way, then the ratio **C<sub>L</sub>**/**C<sub>M</sub>** tells us what part of the maximum possible error **C<sub>M</sub>** is the error of the regression line **C<sub>L</sub>** (or the bad-fit percentage). Therefore you can find how good your model is fitting the data points by subtracting the fraction from 1, and this is known as **Coefficient of Determination!**.
It is also denoted as **R<sup>2</sup>**.

So, if **R<sup>2</sup>** value is close to zero then it indicates you should consider models other than straight lines (as **C<sub>L</sub>** will be higher), and if it is close to one, then the straight line model is a good fit to the data points (as **C<sub>L</sub>** will be close to zero). Phew! That was difficult to explain, but I hope it is clear.

On simplifying the ratio, we get the value of **R<sup>2</sup>** as:

![Data2](/assets/img/rsquare.PNG){: .mx-auto.d-block :}

If you observe the formula carefully, all we need to calculate the value of **R<sup>2</sup>** are the values of **ΣX**, **ΣY**, **ΣXY**, **ΣX<sup>2</sup>**, **ΣY<sup>2</sup>** and **N** (which is 5 in our case). That's it! Let us calculate these values with the help of the following table.

| i | X<sub>i</sub> | Y<sub>i</sub> | X<sub>i</sub><sup>2</sup> | Y<sub>i</sub><sup>2</sup> | X<sub>i</sub>Y<sub>i</sub> |
| :------: |:---: | :------: |:---: | :------: | :------: |
| 1 | 0 | 2 | 0 | 4 | 0 |
| 2 | 1 | 3 | 1 | 9 | 3 |
| 3 | 2 | 5 | 4 | 25 | 10 |
| 4 | 3 | 4 | 9 | 16 | 12 |
| 5 | 4 | 6 | 16 | 36 | 24 |
| Sum | **ΣX=10** | **ΣY=20** | **ΣX<sup>2</sup>=30** | **ΣY<sup>2</sup>=90** | **ΣXY=49** |

On substituting the values of **ΣX**, **ΣY**, **ΣXY**, **ΣX<sup>2</sup>** and **N** in the given formula for **R<sup>2</sup>**, we get

![\large R^{2}= \frac{\left [5 \times 49-10 \times 20\right ]^{2}}{\left [ 5 \times 30-10^{2}\right ]\left [5 \times 90-20^{2}\right ]}=0.81](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5Clarge%20R%5E%7B2%7D%3D%20%5Cfrac%7B%5Cleft%20%5B5%20%5Ctimes%2049-10%20%5Ctimes%2020%5Cright%20%5D%5E%7B2%7D%7D%7B%5Cleft%20%5B%205%20%5Ctimes%2030-10%5E%7B2%7D%5Cright%20%5D%5Cleft%20%5B5%20%5Ctimes%2090-20%5E%7B2%7D%5Cright%20%5D%7D%3D0.81){: .mx-auto.d-block :}

The **R<sup>2</sup>** value of 0.81 (or 81%) tells that 81% of the data points should fall within the regression line ***y = 2.2 + 0.9X***. Therefore it is a good-fit! 

### Multiple Linear Regression

Till now we have been dealing with only a single input variable (**X**). Now consider another example where the mileage of a car depends upon say the maximum speed and the lenght of the car. So in this case, you have 2 independent variables namely maximum speed and the length of the car. Let us denote them by **X<sup>1</sup>** and **X<sup>2</sup>** respectively. Let the dependent variable (the mileage of car) be represented as **Y**. If we assume a linear relationship between independent and dependent variable(s), then this is the case of **Multiple Linear Regression** as more than one variables are involved.

Multiple Linear Regression model is just an extension of Simple Linear Regression, where our predicted output **y** can be thought of as the weighted sum of the inputs plus a bias term. So, if your actual output **Y** depends upon M different input variables (**X<sub>1</sub>**, **X<sub>2</sub>**, ..., **X<sub>M</sub>**), then the predicted output **y** is given as

![\large y = w_{0}+w_{1}X_{1}+w_{2}X_{2}+\cdots +w_{M}X_{M}](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5Clarge%20y%20%3D%20w_%7B0%7D&plus;w_%7B1%7DX_%7B1%7D&plus;w_%7B2%7DX_%7B2%7D&plus;%5Ccdots%20&plus;w_%7BM%7DX_%7BM%7D){: .mx-auto.d-block :}

where **w<sub>1</sub>**, **w<sub>2</sub>**, ..., **w<sub>M</sub>** are the weights attached to the input variables and **w<sub>0</sub>** is the bias term. Now suppose we have N data points (or samples), then let us denote **X<sub>ij</sub>** as the i<sup>th</sup> input variable and j<sup>th</sup> data point and **Y<sub>j</sub>** as the j<sup>th</sup> output where (i will take any integer value from 1,2,...,M) and (j will take any integer value from 1,2,...,N). For more clarity, let us draw a table.

| X<sub>1</sub> | X<sub>2</sub> | ... | X<sub>i</sub> | ... | X<sub>M</sub> | Y |
| :------: |:---: | :------: |:---: | :------: | :------: | :------: |
| X<sub>11</sub> |  X<sub>21</sub> |  ... |  X<sub>i1</sub> | ... |  X<sub>M1</sub> | Y<sub>1</sub> |
| X<sub>12</sub> |  X<sub>22</sub> |  ... |  X<sub>i2</sub> | ... |  X<sub>M2</sub> | Y<sub>2</sub> |
| ... | ... |  ... | ... | ... | ... | ... |
|  X<sub>1j</sub> |  X<sub>2j</sub> |  ... |  X<sub>ij</sub> | ... |  X<sub>Mj</sub> | Y<sub>j</sub> |
| ... | ... |  ... | ... | ... | ... | ... |
|  X<sub>1N</sub> |  X<sub>2N</sub> |  ... |  X<sub>iN</sub> | ... |  X<sub>MN</sub> | Y<sub>N</sub> |

Therefore we can write our predicted output **y<sub>j</sub>** (where j goes from 1,2,...,N) as

![Data2](/assets/img/mlr.PNG){: .mx-auto.d-block :}

If you are confused with too many equations (as we see here), don't worry as we are about to convert these messy equations into a beautiful matrix form! We can also rewrite above equations as:

![mlr](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5Cbegin%7Bbmatrix%7D%20y_%7B1%7D%5C%5C%20y_%7B2%7D%5C%5C%20%5Cvdots%20%5C%5C%20y_%7Bj%7D%5C%5C%20%5Cvdots%20%5C%5C%20y_%7BN%7D%20%5Cend%7Bbmatrix%7D%20%3D%20w_%7B0%7D%5Cbegin%7Bbmatrix%7D%201%5C%5C%201%5C%5C%20%5Cvdots%20%5C%5C%201%5C%5C%20%5Cvdots%20%5C%5C%201%20%5Cend%7Bbmatrix%7D%20&plus;%20w_%7B1%7D%5Cbegin%7Bbmatrix%7D%20X_%7B11%7D%5C%5C%20X_%7B12%7D%5C%5C%20%5Cvdots%20%5C%5C%20X_%7B1j%7D%5C%5C%20%5Cvdots%20%5C%5C%20X_%7B1N%7D%20%5Cend%7Bbmatrix%7D%20&plus;%20w_%7B2%7D%5Cbegin%7Bbmatrix%7D%20X_%7B21%7D%5C%5C%20X_%7B22%7D%5C%5C%20%5Cvdots%20%5C%5C%20X_%7B2j%7D%5C%5C%20%5Cvdots%20%5C%5C%20X_%7B2N%7D%20%5Cend%7Bbmatrix%7D%20&plus;%20%5Ccdots%20&plus;%20w_%7BM%7D%5Cbegin%7Bbmatrix%7D%20X_%7BM1%7D%5C%5C%20X_%7BM2%7D%5C%5C%20%5Cvdots%20%5C%5C%20X_%7BMj%7D%5C%5C%20%5Cvdots%20%5C%5C%20X_%7BMN%7D%20%5Cend%7Bbmatrix%7D
){: .mx-auto.d-block :}

which can further be rewritten as

![mlr](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5Cbegin%7Bbmatrix%7D%20y_%7B1%7D%5C%5C%20y_%7B2%7D%5C%5C%20%5Cvdots%20%5C%5C%20y_%7BN%7D%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%201%20%26%20X_%7B11%7D%20%26%20X_%7B21%7D%20%26%20%5Ccdots%20%26%20X_%7BM1%7D%5C%5C%201%20%26%20X_%7B12%7D%20%26%20X_%7B22%7D%20%26%20%5Ccdots%20%26%20X_%7BM2%7D%5C%5C%20%5Ccdots%20%26%20%5Ccdots%20%26%20%5Ccdots%20%26%20%5Ccdots%20%26%20%5Ccdots%5C%5C%201%20%26%20X_%7B1N%7D%20%26%20X_%7B2N%7D%20%26%20%5Ccdots%20%26%20X_%7BMN%7D%20%5Cend%7Bbmatrix%7D%5Cbegin%7Bbmatrix%7D%20w_%7B0%7D%5C%5C%20w_%7B1%7D%5C%5C%20w_%7B2%7D%5C%5C%20%5Cvdots%20%5C%5C%20w_%7BM%7D%20%5C%5C%20%5Cend%7Bbmatrix%7D
){: .mx-auto.d-block :}

Now let us define matrices **X**, **w** and **y** as follows

![mlr2](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5Clarge%20X%3D%20%5Cbegin%7Bbmatrix%7D%201%20%26%20X_%7B11%7D%20%26%20X_%7B21%7D%20%26%20%5Ccdots%20%26%20X_%7BM1%7D%5C%5C%201%20%26%20X_%7B12%7D%20%26%20X_%7B22%7D%20%26%20%5Ccdots%20%26%20X_%7BM2%7D%5C%5C%20%5Ccdots%20%26%20%5Ccdots%20%26%20%5Ccdots%20%26%20%5Ccdots%20%26%20%5Ccdots%5C%5C%201%20%26%20X_%7B1N%7D%20%26%20X_%7B2N%7D%20%26%20%5Ccdots%20%26%20X_%7BMN%7D%20%5Cend%7Bbmatrix%7D%3B%20%5Chspace%7B0.3cm%7D%20y%3D%20%5Cbegin%7Bbmatrix%7D%20y_%7B1%7D%5C%5C%20y_%7B2%7D%5C%5C%20%5Cvdots%20%5C%5C%20y_%7BN%7D%20%5Cend%7Bbmatrix%7D%3B%20%5Chspace%7B0.3cm%7D%20w%3D%20%5Cbegin%7Bbmatrix%7D%20w_%7B0%7D%5C%5C%20w_%7B1%7D%5C%5C%20w_%7B2%7D%5C%5C%20%5Cvdots%20%5C%5C%20w_%7BM%7D%20%5C%5C%20%5Cend%7Bbmatrix%7D){: .mx-auto.d-block :}

So, finally we can rewrite our messy equations into a beautiful matrix form as follows:

![matrixform](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5Clarge%20Xw%3Dy){: .mx-auto.d-block :}

{: .box-note}
**Note:** Since we have only one output variable (**Y**), so the cost function for Multiple Linear Regression will be similar to that of Simple Linear Regression! 

Our **Objective** for Multiple Linear Regression will be the same as Simple Linear Regression i.e., finding the optimum values for weights and bias (clubbed together by matrix **w**) by ***minimizing*** the Cost function. So, using the same method as described for Simple Linear Regression, we will use **Least Square Method** and a bit of **Calculus** to derive the optimum values as follows: 

![wformula](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5Clarge%20w%3D%5Cleft%20%28%20X%5E%7B%5Ctop%7DX%20%5Cright%20%29%5E%7B-1%7DX%5E%7B%5Ctop%7DY){: .mx-auto.d-block :}

where **X<sup>T</sup>** is the transpose of Matrix **X** which is obtained by interchanging the values of rows and columns of matrix **X**. 

> So, now we have all the tools available to solve any question related to Linear Regression! Finally, let us get our hands dirty by writing a python code for Multiple Linear Regression model from stratch

### Python Code

* **Data Preparation**

We will use a real life data which is built for multiple linear regression and multivariate analysis, known as the Fish Market Dataset that contains information about common fish species in market sales. The dataset includes the fish species, weight, length (of 3 types), height, and width. You can download this dataset from ![here](https://www.kaggle.com/aungpyaeap/fish-market/data#)

Let us first load this data using Pandas library and preview the first 5 lines of the loaded data
 


Here's a code chunk:

~~~
var foo = function(x) {
  return(x + 5);
}
foo(3)
~~~

And here is the same code with syntax highlighting:

```javascript
var foo = function(x) {
  return(x + 5);
}
foo(3)
```

And here is the same code yet again but with line numbers:

{% highlight python linenos %}
import pandas as pd 
data = pd.read_csv("Fish.csv") 
data.head()
{% endhighlight %}

## Boxes
You can add notification, warning and error boxes like this:

### Notification

{: .box-note}
**Note:** This is a notification box.

### Warning

{: .box-warning}
**Warning:** This is a warning box.

### Error

{: .box-error}
**Error:** This is an error box.
