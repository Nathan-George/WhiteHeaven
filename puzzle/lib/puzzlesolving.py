import numpy as np
import cv2

from scipy.spatial import KDTree

import puzzle.lib.puzzletools as tools

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
        corner_size = 1
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
            self.expanded_edges.append(tools.expand_edge(self.edges[i], pixels=2.7))
        
        self.normalized_edges = []
        for i in range(4):
            self.normalized_edges.append(tools.normalize_edge(self.expanded_edges[i], self.edge_types[i]))
        
        self.edge_kdtrees = []
        for i in range(4):
            self.edge_kdtrees.append(KDTree(self.normalized_edges[i]))
    
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
            if edges[i] is None:
                continue
            difference += tools.compare_edges_DTW(self.normalized_edges[i][::skipping], edges[i][::skipping])
        return difference
    
class Puzzle:
    """The set of all solved pieces in the puzzle
    """
    
    EMPTY = -1
    
    def __init__(self, pieces, shape):
        self.pieces = pieces
        self.shape = shape
        
        # array of piece indices
        self.board = np.full(shape, Puzzle.EMPTY, dtype=np.int32)
        self.rotations = np.zeros(shape, dtype=np.int32)
        
        self.pieces_left = list(range(len(pieces)))
    
    # TODO: implement
    def solve(self):
        pass
    
    def place_piece(self, x, y, piece, i):
        """Places a piece in the board so it can be used in comparisons

        Args:
            x (int): location of the piece
            y (int): location of the piece
            piece (int): index of the piece
            i (int): index of the edge which should be on the top
        """
        
        assert self.point_in_bounds(x, y)
        
        # piece is already there
        if self.board[y, x] != Puzzle.EMPTY:
            self.pieces_left.append(self.board[y, x])
            self.pieces_left.sort()
        
        self.pieces_left.remove(piece)
        self.board[y, x] = piece
        self.rotations[y, x] = i
        
    
    def solve_piece(self, x, y):
        edges, edge_types = self.get_pocket(x, y)
        
        # setup the indices
        indices = np.zeros((len(self.pieces_left) * 4, 2), dtype=np.int32)
        i = 0
        for piece in self.pieces_left:
            for j in range(4):
                indices[i] = [piece, j]
                i += 1
        
        # speeds up solving drastically
        indices, _ = self.sort_indices(indices, edges, edge_types, 5)
        indices = indices[:100]
        indices, _ = self.sort_indices(indices, edges, edge_types, 2)
        indices = indices[:10]
        indices, differences = self.sort_indices(indices, edges, edge_types, 1)
        
        return indices, differences
    
    def sort_indices(self, indices, edges, edge_types, skipping=1):
        """sorts the indices by the difference between the edges

        Args:
            indices (np.array): pairs of piece index and edge index
            edges (list): edges to compare against
            edge_types (list): list of the types of edges
            skipping (int, optional): how many points to skip in the comparison. Defaults to 1.
        """
        
        differences = np.zeros(len(indices))
        i = 0
        for piece, j in indices:
            edges_rot = edges[-j:] + edges[:-j]
            edge_types_rot = edge_types[-j:] + edge_types[:-j]
            differences[i] = self.pieces[piece].compare_edges(edges_rot, edge_types_rot, skipping)
            i += 1
        
        sort = np.argsort(differences)
        return indices[sort], differences[sort]
        
    
    def get_pocket(self, x, y):
        """Returns the edges around the location (x, y).
        """
        
        edges = [None for i in range(4)]
        edge_types = [None for i in range(4)]
        
        order = [[0, 1], [-1, 0], [0, -1], [1, 0]]
        for i, direction in enumerate(order):
            edge, type = self.get_edge(x+direction[0], y+direction[1], i)
            edges[(i + 2) % 4] = edge
            edge_types[(i + 2) % 4] = type
            
        return edges, edge_types
    
    def get_edge(self, x, y, edge):
        """Returns the edge of the piece at (x, y), and the type of edge it is.
        
        Args:
            x (int): x position of the piece
            y (int): y position of the piece
            edge (int): edge to get (0=top, 1=right, 2=bottom, 3=left)
        """
        
        if y == self.shape[0] and edge == 0:
            return None, "flat"
        if x == -1 and edge == 1:
            return None, "flat"
        if y == -1 and edge == 2:
            return None, "flat"
        if x == self.shape[1] and edge == 3:
            return None, "flat"
        
        if not self.point_in_bounds(x, y):
            return None, None
        
        index = self.board[y, x]
        if index == Puzzle.EMPTY:
            return None, None
        piece = self.pieces[index]
        
        rot_edge = (edge + self.rotations[y, x]) % 4
        return piece.normalized_edges[rot_edge], piece.edge_types[rot_edge]
        
    def point_in_bounds(self, x, y):
        """Returns true if the point is in bounds of the puzzle
        """
        return x >= 0 and x < self.shape[1] and y >= 0 and y < self.shape[0]
        
        
        