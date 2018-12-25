#!/usr/bin/python3

# Author: Mayank Manjrekar A program to get basket recommendation of a
# new item based on the current list of items in the cart. the
# learning phase of program is based on maximizing the (regulaized)
# log-likelihood function of k-DPP ensembles over the trainin
# dataset. DPPs ensembles provide interesting properties for dealing
# with such problems. We use the Belgian basket dataset for training
# in the code below. This is anonymized set of shopping lists at a
# supermarket. To improve efficiency, we work with low rank matrices
# of rank K, where K intuitively denotes the number of traits of an
# item. The performance of the recommendation engine is computed in
# terms of the Mean-Percentile-Ranking.

# For details of these properties, and for analysis of the
# computational complexity and performance of the current
# implementation, see https://arxiv.org/abs/1602.05436.

# Details of Belgian market dataset:
# author = ”Tom Brijs and Gilbert Swinnen and Koen Vanhoof and Geert Wets”,
# title = ”Using Association Rules for Product Assortment Decisions: A Case Study”,
# year = ”1999”

# In the following, nesterov_descent(...) is a hand-written stochastic
# gradient descent algorithm. grad_ll(...) computes the gradient of
# the log-likelihood function.

# import matplotlib
import numpy as np
import random
import scipy.linalg as spl

# matplotlib.use('Qt5Agg')
randint = np.random.randint

USE_SAVED_V = False  # Flag to run the learning algorithm over the training data
K = 76  # number of traits


def grad_ll(Vl: np.array, tdf):  # Likelihood of the gradient function
    Ml, Kl = Vl.shape
    Nl = len(tdf)
    B = Vl @ spl.inv(np.eye(Kl) + (Vl.T @ Vl))  # V-V(I_k+V^tV)^{-1}V^tV
    secondTerm = 2 * Nl * B
    elems = [list(l) for l in tdf]
    elems = [{i: elems[n][i] for i in range(len(elems[n]))} for n in
             range(Nl)]  # map between items and the locations in the list
    Vn = [np.array([Vl[e] for e in elems[n]]) for n in range(Nl)]
    An = [2 * spl.solve(Vn[n] @ Vn[n].T, Vn[n]) for n in range(Nl)]
    firstTerm = np.zeros((Ml, Kl))
    for n in range(Nl):
        for i in elems[n].keys():
            firstTerm[elems[n][i]] += An[n][i]

    alpha = 1
    llambda = np.ones((Ml, 1))
    for shoplist in tdf:
        for i in shoplist:
            llambda[i] += 1
    llambda = 1 / llambda
    thirdTerm = alpha * llambda * Vl  # regulaization
    gradll = firstTerm - secondTerm - thirdTerm
    return gradll


def loglikelihood(Vl, tdf):
    Ml, Kl = Vl.shape
    llambda = np.ones((Ml, 1))
    Vslist = [np.array([Vl[e] for e in slist]) for slist in tdf]
    firstterm = np.sum([np.log(spl.det(Vslist[n] @ Vslist[n].T)) for n in range(len(Vslist))])
    for slist in tdf:
        for i in slist:
            llambda[i] += 1
    llambda = 1 / llambda
    secondTerm = len(tdf) * np.log(spl.det(np.eye(Kl) + Vl.T @ Vl))
    alpha = 1
    thirdTerm = alpha * (np.array([[spl.norm(Vl[i]) ** 2 for i in range(Ml)]]) @ llambda)[0][0] / 2
    return firstterm - secondTerm - thirdTerm


def nesterov_descent(Vl, tdf, epsilon=1e-4, beta=0.95, T=1000, mbatchsize=1000, func=loglikelihood, delta=1e-5, t=0,
                     tmax=10000):
    Ml, Kl = Vl.shape
    Nl = len(tdf)
    Vt = Vl
    Wt = np.zeros((Ml, Kl))
    while t < tmax:
        mbatch_tdf = np.random.choice(tdf, size=mbatchsize)
        epsilon_t = epsilon / (1 + t / T)
        Wt = (beta * Wt) + (1 - beta) * epsilon_t * grad_ll(Vl, mbatch_tdf)
        print(t, spl.norm(Wt))
        Vtemp = Vt
        Vt = Vt + Wt
        t += 1
        if t % 1000 == 0:
            val1 = func(Vt, tdf)
            val0 = func(Vtemp, tdf)
            rel_error = abs((val1 - val0) / val0)
            print("Relative error = ", rel_error)
            if val1 < val0:
                print("nonsense")
            if rel_error < delta:
                break
    return Vt


def pred_func(slist: set, Vl: np.array):
    Ml, Kl = Vl.shape
    Vslist = np.array([Vl[e] for e in slist])
    VslistT = Vslist.T
    imapl = np.array([i for i in range(Ml) if i not in slist])
    Vnotslist = np.array([Vl[e] for e in range(Ml) if e not in slist])
    Zslist = VslistT @ (spl.solve(Vslist @ VslistT, Vslist))
    Vnew = Vnotslist - (Vnotslist @ Zslist)
    # Lnew = Vnew.dot(Vnew.T)
    Ml_new = Ml - len(slist)
    prob_l = np.array([np.dot(Vnew[i], Vnew[i]) for i in range(Ml_new)])
    prob_l = prob_l / sum(prob_l)
    return prob_l, imapl  # return probability vector and item-map


with open("retail.dat", "r") as file:
    lines = file.readlines()
    lines = [l.strip('\n') for l in lines]

df = [{int(i) for i in l.split()} for l in lines]
items = set()
for i in df:
    items |= i
M = len(items)
tot_N = len(df)
N = int(np.floor(0.7 * tot_N))

for i in range(N):
    j = randint(i, tot_N)
    temp = df[i]
    df[i] = df[j]  # swap
    df[j] = temp

training_df = df[:N]  # Training data
test_df = df[N:]  # Testing data

if not USE_SAVED_V:
    try:
        V = np.loadtxt("Vfile.txt")
    except:
        V = np.random.uniform(0, 1, (M, K))
    V = nesterov_descent(V, training_df) #, t=10000, tmax=20000)
    np.savetxt("Vfile.txt", V)
else:
    V = np.loadtxt("Vfile.txt")


def discard_random_elem(slist):
    i = random.choice(list(slist))
    return slist.difference({i}), i


# Discard a random element from each shopping list in the test data
test_df = [discard_random_elem(slist) for slist in test_df if len(slist) > 1]

MPR = 0
for slist, val in test_df:
    # if val != pred_func(slist, V):
    prob, imap = pred_func(slist, V)
    Mnew = M - len(slist)
    #    error += 1
    MPR += sum(prob <= prob[np.where(imap == val)]) / Mnew
    print(MPR)

MPR = MPR / len(test_df)

print("Mean Percentile Ranking = ", MPR)
