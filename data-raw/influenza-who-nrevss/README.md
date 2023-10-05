# WHO-NREVSS Virology Testing Data for the US

We downloaded virology testing data on March 30, 2022 using the `cdcfluview` package for R. The code for this task is in the file `download-who-nrevss.R`.

We then calculated smoothed estimates of the proportion of influenza cases that were due to each of A/H1, A/H3, and B using a latent Gaussian process logistic regression model. The code for this task is in the file `smooth-who-nrevss.py`. Note that this script is fairly intensive: it took about 8 hours to run on a high powered desktop.

The methods for data preprocessing are described in more detail in the document `smoothing-methods.html`, which is produced by `smoothing-methods.Rmd`.
