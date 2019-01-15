# k-DPP Basket Recommendation Engine

This is an implementation of a learning mechanism for obtaining the kernel of a Determinantal point process (DPPs). DPPs offer and model for set diversity in a variety of subset selection tasks. For example, in the basket recommendation problem, the task is to reccommend to a user a subset of diverse as well as relevant products, so that at least one of the product catches the attention of the user.

In this example, we train a low-rank DPP kernel using the Belgian Basket data [2], given in [retail.dat](retail.dat). This is an implementation of the formalism in [1]. The low-rank condition improves the efficiency in computation of the gradients in the optimization step. Once the DPP kernel has been trained, it is stored in [Vfile.txt](Vfile.txt). The prediction algorithm gives a probability distribution over the next item to recommend. The performance of of the prediction algorithm is measured over the test-data, using a mean-percentile-ranking (MPR). We see that an MPR of ~80% is achievable for this large-scale problem.

[1]: Gartrell, Mike, Ulrich Paquet, and Noam Koenigstein. "Low-rank factorization of determinantal point processes for recommendation." arXiv preprint arXiv:1602.05436 (2016).

[2]: Tom Brijs and Gilbert Swinnen and Koen Vanhoof and Geert Wets, ”Using Association Rules for Product Assortment Decisions: A Case Study”, 1999

# Usage
To run the learning mechanism, set the variable `USE_SAVED_V` to `False`, and run `python3 k-DPP-reco-engine.py`. This will print the MPR obtained over the test data.

To simply run the prediction rule over the test data and obtain the MPR, set `USE-SAVED-V` to `True`, and run `python3 k-DPP-reco-engine.py`.

