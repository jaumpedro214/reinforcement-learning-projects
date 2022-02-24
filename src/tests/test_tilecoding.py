import os
import sys
import unittest

sys.path.append( os.path.join(os.path.dirname(__file__), '..') )

from RLearning.feature_extraction import TileCoding
import numpy as np

class TestTileCoding(unittest.TestCase):
    def test_output_dimensions_1_dim(self):
        nbins = [100]
        n_tiles = 1
        tile_coding = TileCoding(n_bins=nbins, 
                                 limits=[[0,1]], 
                                 n_tiles=n_tiles,
                                ).fit()
        X = np.array( [ [-20], [0.1], [0.2], [0.3], [0.5], [20]] )

        self.assertEqual(
                         tile_coding.transform(X).shape, 
                         ( len(X), np.prod(nbins)*n_tiles )
                        )

    def test_output_dimensions_2_dim(self):
        nbins = [10, 4]
        n_tiles = 1
        tile_coding = TileCoding(n_bins=nbins, 
                                 limits=[[0,1], [1,2]], 
                                 n_tiles=n_tiles, 
                                 tile_shift=[0.1, 0.5]
                                ).fit()
        
        X = np.array( [[0.5, 1.1], 
                       [0.1, 20 ],
                       [0.2, 0.3],] ) 

        self.assertEqual(
                         tile_coding.transform(X).shape, 
                         ( X.shape[0], np.prod(nbins)*n_tiles )
                        )
    
    def test_output_dimensions_3_dim(self):
        nbins = [2, 10, 4]

        n_tiles = 1
        tile_coding = TileCoding(n_bins=nbins, 
                                 limits=[[0,1], [0,1], [0,1]], 
                                 n_tiles=n_tiles, 
                                 tile_shift=[0.1, 0.5, -0.1]
                                ).fit()
        
        X = np.array( [[0.2, 0.5, 1.1], 
                       [0.1, 20, 5.0 ],
                       [0.2, 0.3, 0.8],] ) 

        self.assertEqual(
                         tile_coding.transform(X).shape, 
                         ( X.shape[0],  np.prod(nbins)*n_tiles )
                        )
    
    def test_output_dimensions_1_dim_multitiles(self):
        nbins = [100]
        n_tiles = 10
        tile_coding = TileCoding(n_bins=nbins, 
                                 limits=[[0,1]], 
                                 n_tiles=n_tiles,
                                ).fit()
        X = np.array( [ [-20], [0.1], [0.2], [0.3], [0.5], [20]] )

        self.assertEqual(
                         tile_coding.transform(X).shape, 
                         ( len(X), np.prod(nbins)*n_tiles )
                        )

    def test_output_dimensions_2_dim_multitiles(self):
        nbins = [10, 4]
        n_tiles = 10
        tile_coding = TileCoding(n_bins=nbins, 
                                 limits=[[0,1], [1,2]], 
                                 n_tiles=n_tiles, 
                                 tile_shift=[0.1, 0.5]
                                ).fit()
        
        X = np.array( [[0.5, 1.1], 
                       [0.1, 20 ],
                       [0.2, 0.3],] ) 

        self.assertEqual(
                         tile_coding.transform(X).shape, 
                         ( X.shape[0], np.prod(nbins)*n_tiles )
                        )
    
    def test_output_dimensions_3_dim_multitiles(self):
        nbins = [2, 10, 4]

        n_tiles = 10
        tile_coding = TileCoding(n_bins=nbins, 
                                 limits=[[0,1], [0,1], [0,1]], 
                                 n_tiles=n_tiles, 
                                 tile_shift=[0.1, 0.5, -0.1]
                                ).fit()
        
        X = np.array( [[0.2, 0.5, 1.1], 
                       [0.1, 20, 5.0 ],
                       [0.2, 0.3, 0.8],] ) 

        self.assertEqual(
                         tile_coding.transform(X).shape, 
                         ( X.shape[0],  np.prod(nbins)*n_tiles )
                        )

if __name__ == '__main__':
    unittest.main()