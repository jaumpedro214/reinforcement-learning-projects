from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class TileCoding():
    def __init__(self, n_bins=[10], limits=[[0,1]], n_tiles=1, tile_shift=[1] ):
        """
        Discretize a envrioment using tiles.

        Parameters
        ----------
        n_bins : int orlist of int, default=[10]
            Number of bins in each original dimension
        limits : list, default=[[0,1]]
            Inferior and superior limits for each original dimension
        n_tiles : int, optional
            Total number of tiles created. Default is 1.
        tile_shift : None, optional
            Space between tiles
        """
        self._n_bins = n_bins
        self._limits = np.array(limits)
        self._n_tiles = n_tiles
        self._tile_shift = tile_shift

    def __spacing(self, inferior_lim, superior_lim, n_points):
        """
        TODO Function to provide support for multiple spacing functions.

        Parameters
        ----------
        inferior_lim : number
            Inferior limit of the spacing
        superior_lim : number
            Superior limit of the spacing
        n_points : non-negative int
            Number of points in the spacing

        Returns
        -------
        ndarray
            Spaced numbers over a specified interval.
        """
        spacing = np.linspace(inferior_lim, superior_lim, n_points)
        return spacing

    def fit(self, X=None, y=None):
        """
        Create the needed variables to perform the transform operation.

        Parameters
        ----------
        X : Any, optional
            Not used, just to provide .fit(X,y) support
        y : Any, optional
            Not used, just to provide .fit(X,y) support

        Returns
        -------
        self : object
            Fitted TileCoding
        """
        self._bins = [
                       [
                        # Bins for each tile
                        self.__spacing(self._limits[dim][0] + self._tile_shift[dim]*tile, 
                                       self._limits[dim][1] + self._tile_shift[dim]*tile, 
                                       self._n_bins[dim]
                                       )
                        for dim in range( len(self._n_bins) )
                       ]
                       for tile in range( self._n_tiles )
                     ]
                     
        return self

    def __transform_X_column(self, X_dim, bins):
        """
        Transform a column of an array X using a set of bins.
        This function is used discretize a feature/dimension of X.

        Parameters
        ----------
        X_dim : array-like, 1D
            Column of X
        bins : array-like, 1D
            bins used to transform

        Returns
        -------
        numpy.array
            Discretized array
        """
        X_dim_transformed = np.digitize( X_dim, bins, right=True )
        X_dim_transformed = np.where(X_dim_transformed>=len(bins), 
                                     len(bins)-1, 
                                     X_dim_transformed
                                    )
        X_dim_transformed
        
        return X_dim_transformed

    def __transform_tile(self, X, tile):
        """
        Discretize all the features from an array X using the bins from a tile

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            array to be discretized
        tile : int
            Int representing the tile to be used

        Returns
        -------
        array-like, shape (n_samples, product self._n_bins[:])
            One Hot Encoded array indicating the region in the current tile.
        """

        # Discretizing each feature
        tile_bins = self._bins[tile]
        X_tile_transformed = np.array( 
                                      [
                                        self.__transform_X_column(X[:,dim], tile_bins[dim])
                                        for dim in range( X.shape[1] ) 
                                      ] 
                                     ).T

        # Calculating the global id considering each discretized feature
        n_dummies = np.prod(self._n_bins)

        before_weight = self._n_bins.copy()
        before_weight[0] = 1
        before_weight = np.roll( before_weight, -1 )
        before_weight = np.flip( before_weight )
        before_weight = np.cumprod( before_weight )
        before_weight = np.flip( before_weight )

        X_tile_ids = np.sum(X_tile_transformed*before_weight, axis=1)
        X_tile_ids

        # One Hot Encoding the global id
        X_tile_ohe_transformed = np.zeros( (len(X), n_dummies) )
        X_tile_ohe_transformed[ np.arange(len(X)), X_tile_ids ] = 1

        return X_tile_ohe_transformed

    def transform(self, X, y=None):
        """
        Transform X in its discretized version

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Array to be transformed
        y : Any, optional
            Not used, just to provide .transform(X,y) support

        Returns
        -------
        X_transformed: ndarray, shape (n_samples, total_dimensions)
            X discretized array. total_dimensions = :math:`n\_tiles\prod_{i} {n\_bins[i]}`
        """
        X = np.array(X)
        X_transformed = self.__transform_tile( X, 0 )
        for tile in range(1, self._n_tiles ):
            X_transformed = np.hstack( (X_transformed, self.__transform_tile(X.copy(), tile)) )
        
        return X_transformed