## UCLB Quant Projects

This is an implementation of a paper from [*Expert Systems with Applications*](https://www.sciencedirect.com/science/article/abs/pii/S0957417422004353) tweaks for the clustering

# Authors

[@nicklimjx](https://github.com/nicklimjx)
[@pbht](https://github.com/pbht)

# Thoughts on the paper

The paper has a couple of limitations which we believe could be improved upon. At the root of the issue is the fact that the paper utilises machine learning for prediction. Due to the inherent 'blackbox' nature of machine learning, it is impossible for this to be a truly reliable system. Moreover, the authors did not justify many of the decisions they took, leading to a lack of rigor when choosing clusters and finetuning parameters etc.

One of the best features we thought of the paper was the generation of a correlation matrix for the technical indicators when building the featureset. Doing so means that certain indicators that would measure similar things such as momentum would not be 'double counted' when generating the featureset

# License

What's a license?