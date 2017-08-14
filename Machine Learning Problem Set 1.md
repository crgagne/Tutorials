# Machine Learning Problem Set 1
@(Notes: HWs)[Class_MachineLearning_Ng]

Solutions: https://see.stanford.edu/materials/aimlcs229/ps1_solution.pdf


#### 1. Newton's Method for Solving Least Squares

We are showing that by using Newton's method, we can find the minimum in one step. 

a) Find the hessian of the cost function: $ J(\theta) = 1/2 \sum_{i=0}^m (\theta^T x^{(i)} - y^{(i)})^2 $. 
- First, I'll consider the vector of first derivatives. 
- applying chain rule gives you
-  $ \frac{\partial J(\theta)}{\partial \theta_i}  = \sum_{i=0}^m 2 x_j^{(i)} ( \theta^Tx^{(i)} - y^{(i)}) $ 
- just to show ... $  = (2x_i x_i \theta_i + 2x_j x_i \theta_j .. - 2y x_i)  $
- In matrix form, 
- $ \frac{\partial J(\theta)}{\partial \theta}  =  2 X^T ( \theta^TX- y) $ or $ 2 X^T X \theta- X^T y $ But i'm not 100% sure why this is legal. Transpose facts? 
- Hessian is the matrix of second derivatives. 
- $ \frac{\partial J^2}{\partial \theta_j \ \theta_k} = \sum_{i=0}^m 2x_k^{(i)} x_j^{(i)} $
- $ \frac{\partial J^2}{\partial \theta_j^2}  = \sum_{i=0}^m 2 (x_j^{(i)})^2 $ 
- Or $ H(J(\theta)) = X^T X $ where X is example (row )by feature (column), and vice versa for $X^T$. So the first entry into the H is $x_j^T x_j$


b) Show that the first iteration of Newton's method gives us $ \theta^* = (X^T X)^{-1} X^T y $
- To apply Newton's optimization method in one dimension we are using a second order taylor expansion of f(x). $ f(x) ~= f(x_n + \Delta x) + \Delta x f'(x) + 1/2 \Delta x f''(x) $ and to find the minimum of f(x), we use $ x_{n+1} = x_n -  f'(x)/f''(x) $ 
- In multiple dimensions $ \theta += \theta - H(J(\theta))^{-1} \nabla J(\theta) $. Notice the inverse replaces the division by the second derivative. 

- $\theta += \theta - (X^T X )^{-1}  (X^T X \theta - X^T y)$
- $ \theta += \theta - \theta*1 + (X^T X )^{-1}X^T y $
- $ \theta +=  (X^T X )^{-1}X^T y $ which is one step. 
- alternatively, you could have set the gradient = 0 and solved, and you'd get the same answer. 

####  2. Optional Derivations for problem 2

Derivation of the gradient of the cost function: 
$ l(\theta) = -\lambda/2 \theta^T \theta + \sum_{i=0}^m w^{(i)} [ y^{(i)} log (h_{\theta} (x^{(i)})) + (1-y^{(i)}) log(1-h_\theta (x^{(i)})))] $

The derivative of the penalty term is:
$ .. - \theta \lambda $ 

The derivative of everything else: 
$  \frac{d}{d\theta} w^{(i)}[ y^{(i)} log (h_{\theta} (x^{(i)})) + (1-y^{(i)}) (log(1-h_\theta (x^{(i)})))] $

Get rid of example indicators and w for notational simplicity, and sub in h(x). The Second expansion comes from 1 minus a fraction, then reducing the log. 
$   [ -y log(1+exp(-\theta T x)) + (1-y) (-\theta^T x - log(1+exp(-\theta^T x)) )]$
$  [-y log(1+exp(-\theta T x)) + (1-y) (-\theta^T x - log(1+exp(-\theta^T x))] $
$ [-y log(1+exp(-\theta T x))  - \theta^T x - log(1+exp(-\theta^T x)) + y\theta^T x + ylog(1+exp(-\theta^T x))] $
$ [y\theta^T x-\theta^T x - log(1+exp(-\theta^T x))]  $

Let's take derivatives now
$ \frac{d}{d\theta}[y\theta^T x-\theta^T x - log(1+exp(-\theta^T x))]  $
$  [yx - x -\frac{1}{1+exp(-\theta^T x))} *x* exp(-\theta^T  x)] $
$ x(y-1-\frac{exp(-\theta^T x)}{1+exp(-\theta^T x)})$
$ x(y-\frac{1}{1+exp(-\theta^T x)})$
$ [x(y-h(x))] $

consider w again
$ wx(y-h(x)) $

consider examples i again 
$ \sum_{i=1}^m w^{(i)}x^{(i)}(y^{(i)}-h(x^{(i)})) $

For each training example, w, y, h(x) are a single numbers so we can stack them into vectors. 
$ X^T (\bf{w}(\bf{y}-\bf{h(x)})$

$ \nabla l(\theta) = X^T (\bf{w}(\bf{y}-\bf{h(x)}) - \lambda \theta $

Which means the gradient is again the difference between predicted $ h(\bf{x^{(i)}}) $ and actual $y^{(i)}$, weighted by locally weighted $\bf{x^{(i)}}$, times $\bf{x^{(i)}}$, and then summed over all training examples. 

... Skipping the Hessian 



