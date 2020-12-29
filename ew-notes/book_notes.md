## CoVariance

* the measure of the tendency of two variables to vary together
* difficult to interpret on its own because its units are the product of X and Y (for example, kilogram-centimeters, whatever that means).
    * because of this, it's better to use Pearson's correlation
* formula:
  $Cov(X, Y) = \frac{1}{n} Σ dx_i dy_i$
* in python:
  ``cov = np.dot(xs-meanx, ys-meany) / len(xs)``

## Pearson's Correlation (ρ)

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

* also known as **RMSE** or Root Mean Squared Error, used to measure 'Goodness of Fit'
* in python:
    ```
    def RMSE(estimates, actual):
        e2 = [(estimate-actual)**2 for estimate in estimates]
        mse = np.mean(e2)
        return math.sqrt(mse)
    ```

* RMSE is equal to the standard deviation of the y values if there are no independent/explanatory variables to base your guesses off of 

## Power

* the "correct positive rate" is called the **power** of the test, or sometimes, **sensitivity**
* To get the power, first need to find the **false negative rate**\*:
  * \* if the effect is real, the chance that the hypothesis test will fail
  * more difficult to find because it depends on the actual effect size, and we normally don't know that
  * we can compute this rate using a hypothetical effect size
    * if the observed difference between groups (for example, difference in birth weight for firsts vs. others) is accurate, we can use the observed samples as a model of the population and run hypothesis tests with simulated data
  * formula for false neg rate:
    ```
    def FalseNegRate(data, num_runs=100):
        group1, group2 = data
        count = 0
        for i in range(num_runs):
            sample1 = thinkstats2.Resample(group1)
            sample2 = thinkstats2.Resample(group2)

            ht = DiffMeansPermute((sample1, sample2))
            pvalue = ht.PValue(iters=101)

            if pvalue > 0.05:
                count += 1

        return count / num_runs
    ```
    * takes data in the form of a tuple(example: ``firsts.prglngth, others.prglngth``)
    * each time thru the loop, it draws a random sample & runs the hypothesis test. Then it checks result & counts for false negatives
* now, we can get the **power**, or **correct positive rate** by subtracting the false neg rate by one 
  * ``power = 1 - false_neg_rate``

## Resampling

* takes a sequence and draws a sample with the same length, with replacement
* two ways:
  1. give each row the same probability of being resampled
  * formula:
    ```
    def Resample(xs):
        return np.random.choice(xs, len(xs), replace=True)
    ```
  2. give certain rows different weights than others
       * offsets **oversampling**
          * oversampling example: NSFG data is not a representative sample. The survey deliberately over-samples several groups in order to improve the chance of getting stastically significant results
          * in other words, to improve the **power** of tests involving these groups


## Coefficient of Determination ($R^2$)

* measures goodness of fit between two variables, in terms of minimizing [MSE/RMSE](#standard-error--rmse)
  * **Not easily interpreted in terms of predictive power**
    * standard deviation of the residuals is a better indicator (Downey's opinion)
* denoted $R^2$ and called "R-squared"
* it is equal to (Pearson's Correlation)$^2$
* it has its own formula, though:
    ```
    def CoefDetermination(ys, res):
        return 1 - Var(res) / Var(ys)
    ```
* example:
  * for birth weight and mother's age, $R^2$ is 0.0047, which means that mother's age predicts about half of 1% of variance in birth weight.

## Measuring Goodness of Fit (Methods):

1. Best method: standard deviation of the residuals, especially in comparison to ``stdev(ys)``
   * ``stdev(ys)`` is the same as RMSE if you have to guess $y$s without having explanatory/independent variables to use
2. [**$R^2$**](#coefficient-of-determination-r2)

## Testing Linear Models (Methods):

* test whether an apparent reduction in MSE is due to chance
    *  simulate the Null Hypothesis by permutation
    *  find RMSE of null hypothesis vs sample
* test whether an apparent slope is due to chance
     *  null hypothesis: the slope is equal to zero.
     *  compute the probability that the slope in the sampling distribution falls below 0. 
        * if the estimated slope were negative, we would compute the probability that the slope in the sampling distribution exceeds 0
        * should be easy because because we normally want to compute the sampling distribution of the parameters anyway. 
        * and it is a good approximation unless the sample size is small and the distribution of residuals is skewed. 
          * even then, it is usually good enough, because p-values don't have to be precise.
  
## Regression

* the goal of regression analysis is to describe the relationship between one set of **dependent variables**, and another set of variables called **independent**, or **explanatory** variables
  * dependent variables are typically plotted on the ``y-axis``
  * **ex**planatory on the ***ex**-axis* ;)
* types of regression:
  * **simple regression**: 1 dependent, 1 explanatory variable
  * **multiple regression**: 1 dependent, multiple explanatory variables
  * **multivariate regression**: multiple dependent & explanatory variables

### Linear Regression

* when the relationship between the dependent and explanatory variables is linear
* formula:
$$y = β_0 + β_1 x_1 + β_2 x_2 + ε$$
* where:
  * $β_0$ is the ``intercept``,
  * $β_1$ is the parameter associated with $x_1$,
  * $β_2$ is the parameter associated with $x_2$, and
  * $ε$ is the residual due to random variation or other unknown factors
* given values for $y$ and sequences for $x_1$ and $x_2$, we can find parameters $β_0$, $β_1$, and $β_2$ that **minimize the sum of $ε^2$**
  * this process is known as [**ordinary least squares**](#ordinary-Least-Squares-(#ordinary-least-squares-ols))

#### Ordinary Least Squares (OLS)
* given values for $y$ and sequences for $x_1$ and $x_2$, we can find parameters $β_0$, $β_1$, and $β_2$ that **minimize the sum of $ε^2$** using **Ordinary Least Squares**
* We will use python's ``StatsModels`` (conda built-in) library.
* example (notes below codeblock):
    ```
    import statsmodels.formula.api as smf
    
    live, firsts, others = first.MakeFrames()
    formula = 'totalwgt_lb ~ agepreg'
    model = smf.ols(formula, data=live)
    results = model.fit()
    ```
  * the "formula" API uses strings to identify the dependent and explanatory variables. It uses a syntax called patsy; in this example, the ``~`` operator separates the dependent variable on the left from the explanatory variables on the right.

  * ``smf.ols`` takes the formula string and the DataFrame, live, and returns an OLS ('ordinary least squares') object that represents the model.

  * the **fit** method fits the model to the data and returns a ``RegressionResults`` object that contains the results.

* we can then run ``results.summary()``, which prints a summary of the test results, like so:

![OLS Results Summary](https://github.com/edubu2/ThinkStats2/blob/master/ew-notes/book_notes-snips/11-OLSresultsSummary.jpg)

* the results are also available as attributes. params is a Series that maps from variable names to their parameters, so we can get the intercept and slope like this:
    ```
    inter = results.params['Intercept']
    slope = results.params['agepreg']
    ```

* ``pvalues`` is a Series that maps from variable names to the associated p-values, so we can check whether the estimated slope is statistically significant
    ```
    results.pvalues['agepreg']
    ```

#### Logistic vs Poisson Regression

* **Logistic Regression**
  * when the dependent variable is boolean
  * solves this issue (example):
    * if gender is dependent variable (predicting gender of baby), getting a result of -1.1 or 0.6 does not really help us
    * Logistic Regression solves this by converting the result from probabilities to **odds**
  * formula:
$$log_o = β_0 + β_1x_1 + β_2x_2 + ε$$
* **Poisson Regression**
  * when the dependent variable is an **integer count**
  







