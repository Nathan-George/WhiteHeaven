import numpy as np
import cv2

import puzzletools as tools

# data structures for solving the puzzle


class PuzzlePiece:
    """A puzzle piece. Used for comparing edges of a puzzle piece to other puzzle pieces
    """
    def __init__(self, contour=None):
        if contour is None:
            return
        
        """
        self.contour
        self.edges
        self.edge_types
        self.expanded_edges
        self.normalized_edges
        """
        
        self.contour = contour
        
        # split the contour into 4 edges
        corner_size = 10
        corners = tools.find_corners(contour)
        self.edges = []
        for i in range(4):
            self.edges.append(tools.get_slice_circular(contour, corners[i-1]+corner_size, corners[i]-corner_size))
        
        # determine what type of edge each edge is ("flat", "inner", "outer")
        self.edge_types = []
        for i in range(4):
            self.edge_types.append(tools.get_edge_type(self.edges[i]))
        
        # expand the edges by 2 pixels so they line up better
        self.expanded_edges = []
        for i in range(4):
            self.expanded_edges.append(tools.expand_edge(self.edges[i], pixels=2))
        
        self.normalized_edges = []
        for i in range(4):
            self.normalized_edges.append(tools.normalize_edge(self.expanded_edges[i], self.edge_types[i]))
    
    def compare_edges(self, edges, types, skipping=1):
        """Compares the edges of this piece to the edges of another piece.
        Returns the total difference between the edges.
        """
        
        # first compare the edge types
        for i in range(4):
            if types[i] is None:
                continue
            if types[i] == 'inner' and self.edge_types[i] != 'outer':
                return np.inf
            if types[i] == 'outer' and self.edge_types[i] != 'inner':
                return np.inf
            if types[i] == 'flat' and self.edge_types[i] != 'flat':
                return np.inf
        
        
        difference = 0
        for i in range(4):
            if types[i] is None:
                continue
            difference += tools.compare_edges_DTW(self.normalized_edges[i][::skipping], edges[i][::skipping])
        return difference