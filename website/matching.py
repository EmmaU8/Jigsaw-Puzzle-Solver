# imports
import numpy as np
import cv2
from fastdtw import fastdtw
from helper import get_colors, choose_piece, rescale, rotate


def match_tiles(pieces, screen_pieces, A, B):

    max_contour = 0.015
    max_color = 8000
    max_pixel = 0.03
    max_fit = 0.7
    length = 160
    precision = 8
    step_A = 20
    stap_B = 7
    center = round(length/2)

    pieceA, pieceB = pieces[A], pieces[B]
    contourA, _ = cv2.findContours(pieceA[:,:,3], 0, 1)
    contourB, _ = cv2.findContours(pieceB[:,:,3], 0, 1)
    contourA, contourB = contourA[0].reshape(-1,2), contourB[0].reshape(-1,2)
    length_sum = contourA.shape[0] + contourB.shape[0]

    # contour matching of pieces
    contour_matches = []
    for i in range(0, contourA.shape[0], step_A):

      # piece side contour of pieceA and type of contours
      subcontourA = np.roll(contourA, -i, 0)[:length]
      pointA = tuple(np.flip(subcontourA[center]))
      cA, (hA,wA), aA = cv2.minAreaRect(subcontourA)
      typespotA = np.int0(np.flip(subcontourA[0] + subcontourA[-1] - cA))
      typeA = pieceA[:,:,3][tuple(typespotA)]
      a = cv2.drawContours(np.zeros((300,300),'uint8'), subcontourA.reshape(-1,1,2), -1, 255, 1)

      for j in range(0, contourB.shape[0], stap_B):
        
        # piece side contour of pieceB and its type
        subcontourB = np.roll(contourB, -j, 0)[:length]
        pointB = tuple(np.flip(subcontourB[center]))
        cB, (hB,wB), aB = cv2.minAreaRect(subcontourB)
        typespotB = np.int0(np.flip(subcontourB[0] + subcontourB[-1] - cB))
        typeB = pieceB[:,:,3][tuple(typespotB)]

        # compute best matches via precision
        if typeB != typeA:
          if ((abs(hA-hB) < precision) & (abs(wA-wB) < precision)) or ((abs(hA-wB) < precision) & (abs(wA-hB) < precision)):
            b = cv2.drawContours(np.zeros((300,300),'uint8'), subcontourB.reshape(-1,1,2), -1, 255, 1)
            fmatch = cv2.matchShapes(a,b,1,0)
            if fmatch < max_contour: 
              colinear = True if np.sign(hA-wA) == np.sign(hB-wB) else False
              if colinear:
                codirect = True if (np.sign(typespotA - np.flip(cA)) ==  np.sign(typespotB - np.flip(cB))).all() else False
              else:
                c = np.concatenate([np.sign(typespotA - np.flip(cA)), np.sign(typespotB - np.flip(cB))])
                codirect = True if (abs(np.sum(c[:3])) + abs(np.sum(c[-3:]))) == 4 else False
              if not colinear: aB = aB + 90
              if not codirect: aB = aB + 180  
              contour_matches.append([(i, j), pointA, pointB, round(aB-aA,4), round(fmatch,4)])
  
    # color matching along piece edges
    color_matches = []
    for n in range(len(contour_matches)):
      (i, j), pointA, pointB, angle, fmatch = contour_matches[n]
      subcontourA = np.roll(contourA, -i, 0)[:length] 
      subcontourB = np.roll(contourB, -j, 0)[:length]
      colorsA = get_colors(pieceA, subcontourA)
      colorsB = get_colors(pieceB, subcontourB)
      cmatch = fastdtw(colorsA, np.flip(colorsB, axis=0))[0]
      if cmatch < max_color: 
        color_matches.append([(i, j), pointA, pointB, angle, fmatch, round(cmatch)])

    # pre fitting of puzzle pieces
    fit_matches = []
    for n in range(len(color_matches)):
      (i, j), pointA, pointB, angle, fmatch, cmatch = color_matches[n]
      a = choose_piece(screen_pieces[A][:,:,3], rescale(pointA, [700,700,0]), 0)
      b = choose_piece(screen_pieces[B][:,:,3], rescale(pointB, [700,700,0]), angle)
      loss = 1 - (np.sum((a+b)>0) / (np.sum(a>0) + np.sum(b>0)))
      contours, _ = cv2.findContours((a+b), 0, 1)
      fit = contours[0].shape[0] / length_sum
      if (loss < max_pixel) & (fit < max_fit): 
        fit_matches.append([(A, B), (i, j), pointA, pointB, angle, fmatch, cmatch, round(loss+fit,4), 0])
    fit_matches.sort(key=lambda n: n[-1])

    return fit_matches

def get_matches(pieces, screen_pieces):

    all_matches = []
    for a in range(len(pieces)-1):
        for b in range(a+1,len(pieces)):
            all_matches.extend(match_tiles(pieces, screen_pieces, a,b))

    for n in range(len(all_matches)):
        pair, ij, pointa, pointb, angle, fmatch, cmatch, fit, lock = all_matches[n]
        all_matches.extend([[(pair[1],pair[0]), ij, pointb, pointa, -angle, fmatch, cmatch, fit, lock]])
    all_matches.sort(key=lambda m: (m[0], m[-2]))

    return all_matches