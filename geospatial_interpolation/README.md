# Geospatial-Interpolation
#### This project compares the performance of Kriging & Custom Interpolation implementation

#### <u>Interpolation Description</u>
Let's assume we have a dataset with n observations ($x_{k}, y_{k}, z_{k}$), where $k = 1,....,n. Here $z_{k}$ is the response variable and $x_{k}, y_{k}$ are the two features

$w(x_{a},y_{a},x_{b},y_{b})$ is the distance function for points ($x_{a},y_{a}$) & ($x_{b}, y_{b}$)

Then predicted vector $z = f(x, y) = \sum_{k=1}^{n} \gamma_{k} z_{k}$, with $\gamma_{k} = \underset {i\neq k}{\prod} \frac{w(x, y; x_{i}, y_{i})}{w(x_{k}, y_{k};x_{i}, y_{i})}$

Following updates, improves the predictions
* Replacing $\gamma_{k}$ by $\gamma^{'}_{k} = \gamma_{k}/(1 + \gamma_{k})$. It guarantees that these coefficients lie between 0 and 1.
* Replacing $\gamma^{'}_{k} $ by $\frac{\gamma^{*}_{k}}{w^{\kappa}_{k}(x,y)}$ where $\kappa \geq 0$ is a hyperparameter. This reduces the impact of the
point ($x_{k}, y_{k}$) if it is too far away from $(x, y)$.
* Normalizing $\gamma^{*}_{k}$ so that their sum is equal to 1. This eliminates additive bias outside the training set.

These transformations make the technique somewhat hybrid: a combination of multiplicative, additive, and nearest neighbor methods. Further improvement is obtained by completely ignoring a point ($x_{k}, y_{k}$) when
interpolating $f(x, y)$, if $w_{k}(x, y) > δ$. Here $δ > 0$ is a hyperparameter. It may result in the inability to make a prediction for a point $(x, y)$ far away from all training set points: this is actually a desirable feature, not a defect.
