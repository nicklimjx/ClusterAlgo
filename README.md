# UCLB Quant Projects

This is an implementation of a paper from [*Expert Systems with Applications*](https://www.sciencedirect.com/science/article/abs/pii/S0957417422004353) tweaks for the clustering, completed as part of a research project for the UCL Blockchain society's Algo Trading Projects

The approach uses supervised learning with a RandomForest classifier to identify overbought and oversold trends within the market, thus generating buy and sell signals. It has an accuracy of 0.978.

## Authors

[@nicklimjx](https://github.com/nicklimjx)
[@pbht](https://github.com/pbht)

## Technical breakdown

This approach uses overlapping subseries of 4 hour close data to train the classifier.

Data is first clustered using K-means and Hierarchical clustering according to a Dynamic Time Warping (DTW) distance metric. By not using a Euclidean distance matrix, we hope to better cluster the time series according to trends. This allows us to remove outliers by visual inspection of the clusters. An image is provided to better visualise the problem (see thoughts on the paper for potential improvements).

A featureset is then built using price indicators. The the data is then labelled as 'upwards', 'neutral' or 'downwards' according to the percentage change between the first and last close prices in the subseries. 

## Thoughts on the paper

The paper has a couple of limitations which we believe could be improved upon. At the root of the issue is the fact that the paper utilises machine learning for prediction. Due to the inherent 'blackbox' nature of machine learning, it is impossible for this to be a truly reliable system. Moreover, the authors did not justify many of the decisions they took, leading to a lack of rigor when choosing clusters and finetuning parameters etc. A suggestion for improvement is choosing a certain height up the dendrogram for hierarchical clustering to be able to have outlier clusters.

One of the best features we thought of the paper was the generation of a correlation matrix for the price indicators used  when building the featureset. Doing so means that certain indicators that would measure similar things such as momentum would not be 'double counted' when generating the featureset.

## Extensions

We are currently investigating pump and dump data for trading strategies with [Honglin Fu](https://profiles.ucl.ac.uk/95638-honglin-fu) of the [UCL Centre for Blockchain Technologies](http://blockchain.cs.ucl.ac.uk/). Do contact us if you would like to work on something!

## License

What's a license?
