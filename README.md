# keras-movielens-cf
An implementation expanding on the code fragment in [Alkahest](http://www.fenris.org/)'s blog post
[Collaborative Filtering in Keras](http://www.fenris.org/2016/03/07/collaborative-filtering-in-keras),
tweaked to work in keras 1.0.6, and using the [MovieLens 1M Dataset](http://grouplens.org/datasets/movielens/1m/)
as training data to demonstrate the effectiveness of the architecture. We see an MSE of around 0.85
after about 15 epochs.

## Dependencies

* keras 1.0.6
* numpy 1.11.1
* h5py 2.6.0  
* hdf5 1.8.17

## License
MIT.
