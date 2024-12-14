r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**
1. False. \
The test set is not part of the training process. It can help us estimate the performance on unseen data but not on the seen- training data. \
Maybe if the test error is very small, we can assume that the model learned something useful and would perform at least equally as good on the training data, but it is not guaranteed.\
\
\
2. False.\
There could be many problems with the split. For example, the data could be split in a way that the test set is not representative of the training set.\
Another example is that the test set could be too small to give a good estimate of the model's performance, or the training set is too small to learn from.\
\
\
3. True.\
The test set should only be used to test the generalization of the model and not to train it (tuning the hyper parameters in this case).\
\
\
4. True.\
Validation set performance can be used to estimate the generalization error of the model, but it cannot replace the check with the test set.\
during the cross-validation process, the validation set is used to tune the hyperparameters so the validation set is not truly unseen data, so unfit to truly estimate the generalization error.\

"""

part1_q2 = r"""
**Your answer:**
It may seems logical but it is bad practice and actually wrong to use the test set to tune the hyperparameters.\
If we use the test set to tune the hyperparameters they cease being truly unseen, hence we wouldn't get a good estimate of the generalization error.\
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**

The $\Delta$ parameter is a hyperparameter the control the minimum amount we want the right answer's score to be bigger than the rest.\
can be thought of as the amount of certainty we want our model to predict with.\
If we allow $\Delta < 0$ we will not incur loss when the prediction is wrong, i.e not representing our goal correctly.\
If we set $\Delta > 0$ we are implying we want the correct label to get a score that is at least higher than the rest by $\Delta$

"""

part2_q2 = r"""
**Your answer:**

It seems that each classifier learns the shape of the digit it is supposed to classify.\
That way when the multiplication happens, we get the highest score for the correct digit.\
Many of the classification errors happened on samples which are not well defined or look like another digit. So with the explanation above, the misclassification is understood.
"""

part2_q3 = r"""
**Your answer:**

1. I would say that the learning rate is good-lowish, because it's not clear whether we reach a loss plateau. \
- If it was too low we would see that the process was halted without reaching a plateau loss wise.\
- If it was too high we would see that the loss probably sharply decrease and then oscillates around the minimum.\
- With a good learning rate we see that the loss is decreasing steadily and plateauing.\
\
2. The model is slightly overfitting, because the training loss is lower than the validation and test loss.\

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
The ideal pattern to see in a residual plot is a random scatter of points as close as possible to the horizontal axis.\
when we have spread out y values we would get a line.\
In the last graph we can see the residuals are closer to the horizontal line compared the the five feature graph.\

"""

part3_q2 = r"""
**Your answer:**

1. It's still a linear model, because the model is still a linear combination of the features, but now the features themselves (the space) are not linear.\

2. Yes, because we can 'move' to any feature space we want, getting unlimited expressivity.\
specifically, the polynomial feature space is very expressive.\

3. For the same reasons above, adding non-linear features to the classification model would not turn the classifier itself to non linear.\
the classifier would be a hyperplane in the new feature space.\
Worth noting that the projection of the hyperplane to the original feature space would be non-linear, hence the added expressivity.\

"""

part3_q3 = r"""
**Your answer:**
1. The logspace will give wider range of values to test with fewer values.\
So the cross validation process will be fast while covering a wide range of values.\

2. the data was fitted |degree_range $\times$ lambda_range| * k_folds times.\

"""

# ==============

# ==============
