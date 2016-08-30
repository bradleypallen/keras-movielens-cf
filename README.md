# keras-movielens-cf

A set of Jupyter notebooks demonstrating a simple Keras implmentation
of matrix factorization for collaborative filtering. The notebooks use
the MovieLens 1M Dataset [[1]] to show the effectiveness of the
architecture. Using 120-dimensional embeddings for users and movies,
we achieve an RMSE of 0.862 on a held-out validation set after 15
epochs, taking under 18 minutes on an AWS EC2 g2.2xlarge instance with an
NVIDIA GRID K520 GPU.

The notebooks provide the following workflows:
* MovieLens 1M ETL: loads and processes user, movie and ratings data to prepare them for input into the Keras model.
* MovieLens Training: trains an instance of CFModel using the prepared MovieLens data.
* MovieLens Recommendations: shows recommendations generated using the trained model for a given test user.

## Requirements

* Python 2.7
* A copy of the MovieLens 1M dataset, downloaded from [[2]].

## Dependencies

* pandas (0.18.1)
* matplotlib (1.5.2)
* keras (1.0.5)
* numpy (1.11.1)
* h5py (2.6.0)  

## License

MIT. See the LICENSE file for the copyright notice.

## References

[[1]] F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Trans. Interact. Intell. Syst. 5, 4, Article 19 (December 2015).

[[2]] MovieLens 1M Dataset. http://grouplens.org/datasets/movielens/1m/. Last downloaded 2016-08-14.

[1]: http://dx.doi.org/10.1145/2827872
[2]: http://grouplens.org/datasets/movielens/1m/
