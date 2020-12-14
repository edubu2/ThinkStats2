## CoVariance

* the measure of the tendency of two variables to vary together
* difficult to interpret on its own because its units are the product of X and Y (for example, kilogram-centimeters, whatever that means).
    * because of this, it's better to use Pearson's correlation
* formula:
  $Cov(X, Y) = \frac{1}{n} Σ dx_i dy_i$
* in python:
  ``cov = np.dot(xs-meanx, ys-meany) / len(xs)``

## Pearson's Correlation

* a value in the range(-1:1) that determines strength of the correlation between *X* and *Y*
* it is the variance of *X* and *Y* over the standard deviation of *X* and *Y*
  * can be written as:
  $p=\frac{Cov(X,Y)}{S_X S_Y}$
* or, in python:
  ``corr = Cov(xs, ys, meanx, meany) / math.sqrt(varx * vary)``
* side note: you can square $p$ to find the [coefficient of determination](#coefficient-of-determination-r2) of Determination ($R^2$)

## Spearman's Rank Correlation

* works well if the relationship between variables is linear and if the variables are roughly normal
* not robust in the presence of outliers
* to compute Spearman's correlation, we have to compute the rank of each value, which is its index in the sorted sample.
* in python:
```
# option 1, more manual

def SpearmanCorr(xs, ys):
    xranks = pandas.Series(xs).rank()
    yranks = pandas.Series(ys).rank()
    return Corr(xranks, yranks)

# option 2, using pandas built-in .corr method

def SpearmanCorr(xs, ys):
    xs = pandas.Series(xs)
    ys = pandas.Series(ys)
    return xs.corr(ys, method='spearman')
```

## Standard Error / RMSE

* also known as **RMSE** or Root Mean Squared Error
* in python:
    ```
    def RMSE(estimates, actual):
        e2 = [(estimate-actual)**2 for estimate in estimates]
        mse = np.mean(e2)
        return math.sqrt(mse)
    ```

## Coefficient of Determination ($R^2$)

* measures goodness of fit between two variables, in terms of minimizing [MSE/RMSE](#standard-error--rmse)
  * **Not easily interpreted in terms of predictive power**
    * standard deviation of the residuals is a better indicator (Downey's opinion)
* denoted $R^2$ and called "R-squared"
* it is equal to (Pearson's Correlation)$^2$
* example:
  * for birth weight and mother's age, $R^2$ is 0.0047, which means that mother's age predicts about half of 1% of variance in birth weight.







