import numpy as np
import cv2

from scipy.spatial import KDTree

import lib.puzzletools as tools

# data structures for solving the puzzle
class Edge:
    """An edge of a puzzle piece. Used for comparing edges of a puzzle piece to other puzzle pieces
    """
    
    def __init__(self, points=None, type=None):
        if points is None:
            self.points = None
            self.type = type
            return
        
        self.points = points
        self.type = tools.get_edge_type(points)
        self.expanded = tools.expand_edge(points, 2.3)
        self.normalized = tools.normalize_edge(self.expanded, self.type)
        self.tangents = tools.get_tangents(self.normalized)
        self.kd_tree = KDTree(self.normalized)
    
    def compare_kd_trees(self, other : 'Edge'):
        """Compares the edge to another edge using a KDTree
        """
        
        #TODO: check all points among the two edges
        distances, indices = self.kd_tree.query(other.kd_tree.data)
            
        return np.sum(distances) / len(distances)
        #return np.sum(np.abs(np.cross(other.normalized - self.normalized[indices], other.tangents))) / len(distances)
            
            
        
        

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
            points = tools.get_slice_circular(contour, corners[i-1]+corner_size, corners[i]-corner_size)
            self.edges.append(Edge(points))
    
    def compare_edges(self, edges):
        """Compares the edges of this piece to the edges of another piece.
        Returns the total difference between the edges.
        """
        
        # first compare the edge types
        for i in range(4):
            if edges[i].type is None:
                continue
            if edges[i].type == 'inner' and self.edges[i].type != 'outer':
                return np.inf
            if edges[i].type == 'outer' and self.edges[i].type != 'inner':
                return np.inf
            if edges[i].type == 'flat' and self.edges[i].type != 'flat':
                return np.inf
        
        
        difference = 0
        num_edges = 0
        for i in range(4):
            if edges[i].points is None:
                continue
            difference += self.edges[i].compare_kd_trees(edges[i])
            num_edges += 1
            
        if num_edges == 0:
            return 0
        return difference / num_edges
    
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
    
    def remove_piece(self, x, y):
        assert self.point_in_bounds(x, y)
        
        if self.board[y, x] == Puzzle.EMPTY:
            return
        
        self.pieces_left.append(self.board[y, x])
        self.board[y, x] = Puzzle.EMPTY
        self.rotations[y, x] = 0
    
    def solve_piece(self, x, y):
        edges = self.get_pocket(x, y)
        
        # setup the indices
        indices = np.zeros((len(self.pieces_left) * 4, 2), dtype=np.int32)
        i = 0
        for piece in self.pieces_left:
            for j in range(4):
                indices[i] = [piece, j]
                i += 1
        
        # speeds up solving drastically
        indices, differences = self.sort_indices(indices, edges)
        
        return indices, differences
    
    def sort_indices(self, indices, edges):
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
            differences[i] = self.pieces[piece].compare_edges(edges_rot)
            i += 1
        
        sort = np.argsort(differences)
        return indices[sort], differences[sort]
        
    
    def get_pocket(self, x, y):
        """Returns the edges around the location (x, y).
        """
        
        edges = [None for i in range(4)]
        
        order = [[0, 1], [-1, 0], [0, -1], [1, 0]]
        for i, direction in enumerate(order):
            edges[(i + 2) % 4] = self.get_edge(x+direction[0], y+direction[1], i)
            
        return edges
    
    def get_edge(self, x, y, edge):
        """Returns the edge of the piece at (x, y), and the type of edge it is.
        
        Args:
            x (int): x position of the piece
            y (int): y position of the piece
            edge (int): edge to get (0=top, 1=right, 2=bottom, 3=left)
        """
        
        if y == self.shape[0] and edge == 0:
            return Edge(None, "flat")
        if x == -1 and edge == 1:
            return Edge(None, "flat")
        if y == -1 and edge == 2:
            return Edge(None, "flat")
        if x == self.shape[1] and edge == 3:
            return Edge(None, "flat")
        
        if not self.point_in_bounds(x, y):
            return Edge(None, None)
        
        index = self.board[y, x]
        if index == Puzzle.EMPTY:
            return Edge(None, None)
        piece = self.pieces[index]
        
        rot_edge = (edge + self.rotations[y, x]) % 4
        return piece.edges[rot_edge]
        
    def point_in_bounds(self, x, y):
        """Returns true if the point is in bounds of the puzzle
        """
        return x >= 0 and x < self.shape[1] and y >= 0 and y < self.shape[0]
        
        
        