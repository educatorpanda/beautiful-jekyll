---
layout: post
title: Everything you need to know about Simple Linear Regression!
subtitle: Machine Learning Episode-1.1
gh-repo: daattali/beautiful-jekyll
gh-badge: [star, fork, follow]
tags: [test]
comments: true
---

Simple Linear Regression is a powerful statistical tool that is used to determine the relationship between two quantitative variables.

> "Somewhere between Simple Linear Regression and Deep Neural Networks we grow up to become a Data Scientist" 

1. The first variable (also denoted as '**X**') is called the independent variable.
2. The second variable (also denoted as '**Y**') is called the dependent variable.

The independent variable is the variable the experimenter changes or controls and is assumed to have a direct effect on the dependent variable. For example: Driving speed and gas mileage — as driving speed increases, you'd expect gas mileage to decrease. So, here the driving speed is the independent variable which is controlled by the driver (experimenter) and it is assumed to have a direct effect on the dependent variable which is the gas mileage.

### In this article, we will be covering the following topics: 
* [Introduction to Simple Linear Regression](#introduction-to-simple-linear-regression)
* [Residuals and Cost Function](#residuals-and-cost-function)
* Simple Linear Regression model
* Mathematical Derivation using Least Square Method
* Find the Coefficient of Determination (R<sup>2</sup>)
* Coding (in python) from scratch
* Coding (in python) using libraries
* Applications and Usage 


### Introduction to Simple Linear Regression 
As you may have studied in your high school, the relationship between the two different units of Temperature, degrees Fahrenheit (F) and degrees Celsius (C) is known to be:  

![\Large F= \frac{9}{5}C+32](https://latex.codecogs.com/gif.latex?F%3D%20%5Cfrac%7B9%7D%7B5%7DC&plus;32){: .mx-auto.d-block :}

![Temperature](/assets/img/CelciusFahrenheit.PNG){: .mx-auto.d-block :}

That is, if you know the temperature in degrees Celsius, you can use this equation to determine the temperature in degrees Fahrenheit **exactly**. Such a relationship is called a deterministic (or functional) relationship. In a deterministic relationship, the equation **exactly** describes the relationship between the two variables.

However, in the world of data science, we never talk about the deterministic relationship. Instead, we are always interested in statistical relationships. So, today let us understand in detail, one of the most basic method which is used to establish the statistical relationship between two variables (an independent variable and a dependent variable) known as **Simple Linear Regression**. 

**What is Simple Linear Regression?**

Simple Linear Regression is a basic regression analysis where we have just two variables (an independent variable and a dependent variable) and based on the changes made to the independent variable (**X**), we try to predict the outcome of the dependent variable (**Y**). Let us understand this through a fictitious example.

This table shows the investment of the company in its business over the years. Our task at hand is to predict the investment of the company in the 6<sup>th</sup> year using Simple Linear Regression model. So how will you do that?

| Years | Investment (in M) |
| :------ |:--- |
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

The core idea in Simple Linear Regression is to obtain a line that best fits the data. Mathematically, the equation of such a line is of the form:

![\Large y=w_{0}+w_{1}X ](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5CLarge%20y%3Dw_%7B0%7D&plus;w_%7B1%7DX){: .mx-auto.d-block :}

where **'y'** represents the predicted output for a given input **'X'**. The terms **'w<sub>0</sub>'** and **'w<sub>1</sub>'** represents the *Y-intercept* of the line (i.e. the point where the given line intersects the Y-axis) and *Slope* of the given line respectively. Let us collectively call the terms **'w<sub>0</sub>'** and **'w<sub>1</sub>'** as  the weights attached to the input (or simply **'weights'**)

In the figure given below, we find that there can be multiple lines with which we can fit the given data. But the best fit line is the one for which the total prediction error for all the data points is as small as possible, i.e. we find the optimum value of the weights such that the total error associated with our prediction is ***minimum***.

![Data2](/assets/img/datavisualization2.png){: .mx-auto.d-block :}

### Residuals and Cost Function

Taking about the errors, let us understand the concept of **Residual** and Total prediction error (or the **Cost Function**)

1. **Residual**

Out of the given multiple lines in the above figure, let us arbitrarily choose a line and call it as ***L*** such that ***Line L: y = w<sub>0</sub> + w<sub>1</sub>X*** becomes the regression line. So, a Residual is simply the vertical distance (denoted by the red line, in the figure below) between a data point and the regression line ***L***. Each data point has one residual. In our example, we have 5 data points, so there will be 5 residuals in total. They are negative if they are above the regression line and positive if they are below the regression line. If the regression line actually passes through the point, the residual at that point is zero.

![Residual](/assets/img/residual.PNG){: .mx-auto.d-block :}

We donote this residual term at the i<sup>th</sup> data point as **e<sub>i</sub>**. So, for the i<sup>th</sup> data point **(X<sub>i</sub>,Y<sub>i</sub>)**, the value of the predicted output will be **y<sub>i</sub>** such that ***y<sub>i</sub> = w<sub>0</sub> + w<sub>1</sub>X<sub>i</sub>*** and the value of the residual at this data point will be given as: 

![\Large e_{i}=y_{i}-Y_{i}](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5CLarge%20e_%7Bi%7D%3Dy_%7Bi%7D-Y_%7Bi%7D){: .mx-auto.d-block :}

2. **Cost Function**

**Cost function** is a Machine Learning Terminology which is simply a measure of how wrong the model is in term of its ability to estimate the relationship between **X** and **Y**. **Cost Function** quantifies the error between predicted values (**y**) and actual values (**Y**) and presents it in the form of a single real number. Thus Cost function, in layman terminology is nothing but the total prediction error. We denote this single real number by **C**. 

Now you can sense a relationship between Residual and Cost function. Residual denotes the error for a single data point, whereas the Cost function denotes the error for all the given data points. Now, in order to develop a model which can accurately predict the output, we just need to minimize the Cost function (**C**). Depending on the problem, the Cost Function can be formed in many different ways. But for our case, it will be given as:

![\Large C=\sum_{i=1}^{N}\frac{1}{2}\left ( y_{i}-Y_{i} \right )^{2}=\sum_{i=1}^{N}\frac{1}{2}\left ( e_{i}\right )^{2}](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5CLarge%20C%3D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Cfrac%7B1%7D%7B2%7D%5Cleft%20%28%20y_%7Bi%7D-Y_%7Bi%7D%20%5Cright%20%29%5E%7B2%7D%3D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Cfrac%7B1%7D%7B2%7D%5Cleft%20%28%20e_%7Bi%7D%5Cright%20%29%5E%7B2%7D){: .mx-auto.d-block :}

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

{% highlight javascript linenos %}
var foo = function(x) {
  return(x + 5);
}
foo(3)
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
