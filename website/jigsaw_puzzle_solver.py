# imports
from matching import get_matches
from preprocessing import preprocessing
from assemble import assemble


def puzzle_solver(puzzle_filename, pieces_number):
  
  _, pieces, piece_centers, screen_pieces = preprocessing(puzzle_filename, pieces_number)

  matches = get_matches(pieces, screen_pieces)

  assembly = assemble(pieces, screen_pieces, matches, piece_centers, pieces_number)

  return assembly


