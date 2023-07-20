import cv2
import numpy as np

import os
os.chdir('C:\\Users\\nbg05\\Local Documents\\Projects\WhiteHeaven')

import lib.puzzletools as tools
from lib.puzzlesolving import PuzzlePiece

from tqdm import tqdm

# create a list of all the images in the folder
# Gets the puzzle piece image
def get_piece_image(path):
    # images are of the format piece#.png
    img = cv2.imread(path)
    return img

def main():
    """generates all edge comparisons for all pieces in the dataset and saves them so
    they can be loaded later for faster comparisons to solve the puzzle
    """
    
    piece_dir = 'dataset/pieces-clean'
    image_files = os.listdir(piece_dir)
    image_files = sorted(image_files, key=lambda x: int(x[5:-4]))

    # load all off the images
    piece_images = []
    for i in tqdm(range(len(image_files)), 'images'):
        piece_images.append( get_piece_image(os.path.join(piece_dir, image_files[i])) )

    piece_contours = []
    for i in tqdm(range(len(piece_images)), 'contours'):
        piece_contours.append(tools.get_contour(piece_images[i], 226))
        
    pieces = []
    for i in tqdm(range(len(piece_contours)), 'pieces'):
        pieces.append(PuzzlePiece(piece_contours[i]))
        
    # main dataset for storing the comparisons of all edges
    comparison_scores_dataset = np.zeros((len(pieces), 4, len(pieces), 4), dtype=np.float32)
    
    indices = [(piece, i) for i in range(4) for piece in range(len(pieces))]
    
    for piece1, i1 in tqdm(indices, 'comparisons'):
        edge1 = pieces[piece1].edges[i1]
        if edge1.type == 'flat':
            comparison_scores_dataset[piece1, i1, :, :] = np.inf
            continue
        
        for piece2, i2 in indices:
            edge2 = pieces[piece2].edges[i2]
            if edge2.type == 'flat':
                comparison_scores_dataset[piece1, i1, piece2, i2] = np.inf
                continue
            
            if piece1 == piece2:
                continue
            
            # do not compare edges of the same type (inner to inner, outer to outer, etc)
            if edge1.type == edge2.type:
                comparison_scores_dataset[piece1, i1, piece2, i2] = np.inf
            
            # compare the edges
            comparison_scores_dataset[piece1, i1, piece2, i2] = edge1.compare_kd_trees(edge2, threads=1)
    
    # sub dataset of sorted comparisons
    num_comparisons = 64
    # store the 64 best comparisons for each edge
    sorted_scores_dataset = np.zeros((len(pieces), 4, num_comparisons), dtype=np.float32)
    sorted_pieces_dataset = np.zeros((len(pieces), 4, num_comparisons), dtype=np.int32)
    sorted_indices_dataset = np.zeros((len(pieces), 4, num_comparisons), dtype=np.int32)
    
    for piece, i in tqdm(indices, 'sorting'):
        comparison_scores = comparison_scores_dataset[piece, i, :, :]
        flattened_scores = comparison_scores.flatten()
        sort = np.argsort(flattened_scores)[:num_comparisons]
        
        sorted_pieces_dataset, sorted_indices_dataset = np.unravel_index(sort, comparison_scores.shape)
        sorted_scores_dataset[piece, i, :] = flattened_scores[sort]
    
    print('saving')
    np.save('puzzle/data/comparison_scores.npy', comparison_scores_dataset)
    np.savez('puzzle/data/sorted_scores.npz', scores=sorted_scores_dataset, pieces=sorted_pieces_dataset, indices=sorted_indices_dataset)

if __name__ == '__main__':
    main()