import numpy as np
import cv2
from helper import choose_piece, rescale, rotate

def updateCanvas(screen, locations, pointA, pointB, cornerA, cornerB, B):
    
    # push pieces on "working surface" aka canvas
    for N, pos in enumerate(locations):
      if N in screen:
        new_center = (pos[0] + 700 - pointA[0], pos[1] + 700 - pointA[1])
        new_center = rotate(new_center, cornerA)
        new_corner = pos[2] + cornerA
        locations[N] = [*new_center, new_corner]

    screen.append(B)
    center = rotate((700 + 700 - pointB[0], 700 + 700 - pointB[1]), cornerB)
    locations[B] = [*center, cornerB]

    return screen, locations


def assemble(pieces, screen_pieces, matches, piece_centers, piece_number):
  # put pieces together
  solution = screen_pieces[0].copy()
  locations = [[0,0,0]]*len(pieces)
  locations[0] = [700,700,0]
  screen = [0]
  attempts = 0

  while (len(screen) < piece_number) & (attempts < 15):
    for n in range(len(matches)):
          
      (A, B), _, pointA, pointB, angleB, _, _, _, lock = matches[n]
      pointA = rescale(pointA, locations[A])
      pointB = rescale(pointB, (700,700,0))

      if A in screen:
        cornerA = - locations[A][2]
        pre_assembly = choose_piece(solution.copy(), pointA, cornerA)
        
        if B not in screen:
          new_piece = choose_piece(screen_pieces[B], pointB, angleB)

          loss = (np.sum(pre_assembly[:,:,3]>0) + np.sum(new_piece[:,:,3]>0) - 
                  np.sum((pre_assembly+new_piece)[:,:,3]>0)
                  ) / np.sum(new_piece[:,:,3]>0)
          if loss < 0.1: 
            matches[n][-1] = 1
            solution = pre_assembly.copy() + new_piece.copy()
            screen, locations = updateCanvas(screen, locations, 
                                            pointA, pointB, cornerA, angleB, B)
    
    attempts += 1

  return solution