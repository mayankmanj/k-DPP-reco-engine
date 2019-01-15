#!/usr/bin/python3

# Author: Mayank Manjrekar

# A program to get basket recommendation of a new item based on the
# current list of items in the cart. the learning phase of program is
# based on maximizing the (regulaized) log-likelihood function of
# k-DPP ensembles over the trainin dataset. DPPs ensembles provide
# interesting properties for dealing with such problems. We use the
# Belgian basket dataset for training in the code below. This is
# anonymized set of shopping lists at a supermarket. To improve
# efficiency, we work with low rank matrices of rank K, where K
# intuitively denotes the number of traits of an item. The performance
# of the recommendation engine is computed in terms of the
# Mean-Percentile-Ranking.

# For details of these properties, and for analysis of the
# computational complexity and performance of the current
# implementation, see https://arxiv.org/abs/1602.05436.

# Details of Belgian market dataset:
# author = ”Tom Brijs and Gilbert Swinnen and Koen Vanhoof and Geert Wets”,
# title = ”Using Association Rules for Product Assortment Decisions: A Case Study”,
# year = ”1999”

# In the following, nesterov_descent(...) is a stochastic gradient
# descent algorithm. grad_ll(...) computes the gradient of the
# log-likelihood function.

# import matplotlib
import numpy as np
import random
import scipy
import scipy.linalg as spl

# matplotlib.use('Qt5Agg')
randint = np.random.randint

USE_SAVED_V = True  # Flag to run the learning algorithm over the training data
K = 76  # number of traits


class vectorized(object):
    def __init__(self, excluded = None, otypes = None):
        self.excluded = excluded
        self.otypes = otypes

    def __call__(self, func):
        return np.vectorize(func, excluded=self.excluded, otypes=self.otypes)

@vectorized(excluded=[1,2], otypes=[np.ndarray])
def grad_ll_helper(elems: np.ndarray, Vl, fterm):
    Vn = np.array([Vl[e] for e in elems])
    if type(Vn) == float:
        Vn = np.array([[Vn]])
        print("exit")
    An = 2 * spl.solve(Vn @ Vn.T, Vn)
    # Fl = np.zeros(Vl.shape)
    for i, e in enumerate(elems):
        fterm[e] += An[i]


@vectorized(excluded=[1])
def regularization_helper(elems, lmbda):
    for i in elems:
        lmbda[i] += 1

def grad_ll(Vl: np.array, tdf):  # Likelihood of the gradient function
    Ml, Kl = Vl.shape
    Nl = len(tdf)

    B = spl.solve(np.eye(Kl) + (Vl.T @ Vl), Vl.T).T  # V-V(I_k+V^tV)^{-1}V^tV


    secondTerm = 2 * Nl * B

    # firstTerm = np.sum(vgrad_ll_helper(tdf, Vl))

    firstTerm = np.zeros(Vl.shape)
    # for slist in tdf:
    #     grad_ll_helper(slist, Vl, firstTerm)
    grad_ll_helper(tdf, Vl, firstTerm)

    alpha = 0.1

    llambda = np.ones((Ml, 1))
    regularization_helper(tdf, llambda)

    llambda = 1 / llambda
    thirdTerm = alpha * llambda * Vl  # regularization

    gradll = firstTerm - secondTerm - thirdTerm
    return gradll

@vectorized(excluded=[1])
def loglikelihood_helper(elems: np.ndarray, Vl):
    Vn = np.array([Vl[e] for e in elems])
    return np.log(spl.det((Vn @ Vn.T)))


def loglikelihood(Vl, tdf):
    Ml, Kl = Vl.shape

    firstterm = np.sum(loglikelihood_helper(tdf, Vl))

    llambda = np.ones((Ml, 1))
    regularization_helper(tdf, llambda)

    llambda = 1 / llambda
    secondTerm = len(tdf) * np.log(spl.det(np.eye(Kl) + Vl.T @ Vl))
    alpha = 0.1
    thirdTerm = alpha * (np.array([[spl.norm(Vl[i]) ** 2 for i in range(Ml)]]) @ llambda)[0][0] / 2
    return firstterm - secondTerm - thirdTerm


def nesterov_descent(Vl, tdf, epsilon=1e-5, beta=0.95, T=100, mbatchsize=10000, func=loglikelihood, delta=1e-5, t=0,
                     tmax=10000):
    Ml, Kl = Vl.shape
    Nl = len(tdf)
    Vt = Vl
    Wt = np.float_(np.zeros((Ml, Kl)))
    val0 = 1
    np.random.shuffle(tdf)
    l_index = 0
    h_index = mbatchsize
    while t < tmax:
        if l_index >= Nl:
            np.random.shuffle(tdf)
            l_index = 0
            h_index = mbatchsize
        mbatch_tdf = tdf[l_index:h_index]
        epsilon_t = epsilon / (1 + t / T)
        Wt = (beta * Wt) + (1 - beta) * epsilon_t * grad_ll(Vt + beta * Wt, mbatch_tdf)
        if t % 100 == 0:
            print(t, spl.norm(Wt), spl.norm(Vt))
        if t % 1000 == 0:
            val1 = func(Vt, tdf)
            rel_error = abs((val1 - val0) / val0)
            print("Relative error = ", rel_error)
            if val1 < val0:
                print("nonsense")
            if rel_error < delta:
                break
            val0 = val1
        Vt = Vt + Wt
        t += 1
        l_index += mbatchsize
        h_index += mbatchsize
    return Vt


def pred_func(slist: set, Vl: np.array):
    Ml, Kl = Vl.shape
    Vslist = np.array([Vl[e] for e in slist])
    VslistT = Vslist.T
    imapl = np.array([i for i in range(Ml) if i not in slist])
    Vnotslist = np.array([Vl[e] for e in range(Ml) if e not in slist])
    Zslist = VslistT @ (spl.solve(Vslist @ VslistT, Vslist))
    Vnew = Vnotslist - (Vnotslist @ Zslist)
    Ml_new = Ml - len(slist)
    prob_l = np.array([np.dot(Vnew[i], Vnew[i]) for i in range(Ml_new)])
    prob_l = prob_l / sum(prob_l)
    return prob_l, imapl  # return probability vector and item-map


with open("retail.dat", "r") as file:
    lines = file.readlines()
    lines = [l.strip('\n') for l in lines]

df = np.array([np.array([int(i) for i in l.split()]) for l in lines])
items = set()
for i in df:
    items |= set(i)
M = len(items)
tot_N = len(df)
N = int(np.floor(0.7 * tot_N))

# for i in range(N):
#     j = randint(i, tot_N)
#     temp = df[i]
#     df[i] = df[j]  # swap
#     df[j] = temp

training_df = df[:N]  # Training data
test_df = df[N:]  # Testing data

if not USE_SAVED_V:
    V = np.float_(np.random.uniform(-1, 1, (M, K)))
    V = nesterov_descent(V, training_df, tmax=10000, mbatchsize=10000, epsilon=1e-4, beta=0.9, T=100)
    np.savetxt("Vfile.txt", V)
else:
    V = np.loadtxt("Vfile.txt")


def discard_random_elem(slist: np.ndarray):
    i = np.random.choice(slist)
    return i, slist[slist != i]


# Discard a random element from each shopping list in the test data
test_df = [discard_random_elem(slist) for slist in test_df if len(slist) > 1]

MPR = 0
iter = 0
for val, slist in test_df:
    # if val != pred_func(slist, V):
    prob, imap = pred_func(slist, V)
    Mnew = M - len(slist)
    # error += 1
    MPR += sum(prob <= prob[np.where(imap == val)]) / Mnew
    iter += 1
    print(iter, MPR)

MPR = MPR / len(test_df)

print("Mean Percentile Ranking =", MPR)
