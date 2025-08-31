export const chessPiecesLookupNames = {
  blackPawn: 0,
  blackRook: 1,
  blackKnight: 2,
  blackBishop: 3,
  blackQueen: 4,
  blackKing: 5,
  //
  whitePawn: 6,
  whiteRook: 7,
  whiteKnight: 8,
  whiteBishop: 9,
  whiteQueen: 10,
  whiteKing: 11,
  //
  empty: 12,
} as const;

export const chessPiecesLookup = {
  // black pieces
  p: 0,
  r: 1,
  n: 2,
  b: 3,
  q: 4,
  k: 5,
  // white pieces
  P: 6,
  R: 7,
  N: 8,
  B: 9,
  Q: 10,
  K: 11,
  //
  s: 12,
} as const;
