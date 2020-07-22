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
* Introduction to Simple Linear Regression (Best fitting line)
* Residuals and Cost Function
* Simple Linear Regression model
* Mathematical Derivation to find the optimum value of the weights
* Solved Example
* Find the Coefficient of Determination (R<sup>2</sup>)
* Coding (in python) from scratch
* Coding (in python) using libraries
* Fun Plotting

## 1. Introduction to Simple Linear Regression
As you may have studied in your high school, the relationship between the two different units of Temperature, degrees Fahrenheit (F) and degrees Celsius (C) is known to be:  


That is, if you know the temperature in degrees Celsius, you can use this equation to determine the temperature in degrees Fahrenheit **exactly**. Such a relationship is called a deterministic (or functional) relationship. In a deterministic relationship, the equation **exactly** describes the relationship between the two variables.

![\Large F= \frac{9}{5}C+32](https://latex.codecogs.com/gif.latex?F%3D%20%5Cfrac%7B9%7D%7B5%7DC&plus;32){: .mx-auto.d-block :}

![Temperature]({{ /blob/master }}/assets/img/CelciusFahrenheit.PNG){: .mx-auto.d-block :}

*Deterministic relationship between Fahrenheit and Celsius*

However, in the world of data science, we never talk about the deterministic relationship. Instead, we are always interested in statistical relationships.







Here's a useless table:

| Number | Next number | Previous number |
| :------ |:--- | :--- |
| Five | Six | Four |
| Ten | Eleven | Nine |
| Seven | Eight | Six |
| Two | Three | One |




It can also be centered!



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
