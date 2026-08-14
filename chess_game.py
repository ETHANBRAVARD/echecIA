"""
chess_game.py — Classe de jeu d'échecs complète, sans dépendance externe.

Fournit tout ce qu'il faut pour faire jouer des IA dessus :
- validation complète des règles (roque, prise en passant, promotion,
  échec, mat, pat, règle des 50 coups, triple répétition, matériel insuffisant)
- coups acceptés en notation UCI ("e2e4", "e7e8q") ou SAN ("e4", "Cf3" -> "Nf3",
  "O-O", "exd6", "e8=Q+")
- export FEN et PGN (très utile pour construire les prompts de tes IA)
- perft() pour vérifier la génération de coups

Utilisation minimale :

    game = ChessGame()
    print(game)                     # plateau ASCII
    game.play("e4")                 # SAN ou UCI, au choix
    game.legal_moves_san()          # ['a6', 'a5', 'Nc6', ...]
    game.is_game_over()             # bool
    game.result()                   # "1-0", "0-1", "1/2-1/2" ou None

Convention interne : le plateau est une liste 8x8, board[0][0] = a8
(comme l'ordre d'une FEN). Les pièces blanches sont en MAJUSCULES
('P','N','B','R','Q','K'), les noires en minuscules. Case vide = None.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

WHITE = "w"
BLACK = "b"

FILES = "abcdefgh"

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

KNIGHT_OFFSETS = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                  (1, -2), (1, 2), (2, -1), (2, 1)]
KING_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                (0, 1), (1, -1), (1, 0), (1, 1)]
ROOK_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BISHOP_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

UNICODE_PIECES = {
    "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
    "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
}

# Droits de roque perdus quand une case est quittée ou capturée
# (e1/e8 = roi, coins = tours).
CASTLING_RIGHTS_LOST = {
    (7, 4): {"K", "Q"}, (7, 7): {"K"}, (7, 0): {"Q"},
    (0, 4): {"k", "q"}, (0, 7): {"k"}, (0, 0): {"q"},
}


def square_name(row: int, col: int) -> str:
    """(row, col) internes -> nom algébrique, ex. (4, 4) -> 'e4'."""
    return FILES[col] + str(8 - row)


def parse_square(name: str) -> tuple[int, int]:
    """Nom algébrique -> (row, col) internes, ex. 'e4' -> (4, 4)."""
    return 8 - int(name[1]), FILES.index(name[0])


@dataclass(frozen=True)
class Move:
    """Un coup : cases de départ/arrivée + éventuelle promotion ('q','r','b','n')."""
    fr: int  # from row
    fc: int  # from col
    tr: int  # to row
    tc: int  # to col
    promotion: Optional[str] = None

    def uci(self) -> str:
        return (square_name(self.fr, self.fc)
                + square_name(self.tr, self.tc)
                + (self.promotion or ""))


class ChessGame:
    """Partie d'échecs : état du plateau, règles, validation et fin de partie."""

    def __init__(self, fen: str = START_FEN):
        self.set_fen(fen)
        # Historique lisible de la partie : liste de (uci, san).
        self.moves_played: list[tuple[str, str]] = []

    # ------------------------------------------------------------------ FEN

    def set_fen(self, fen: str) -> None:
        """Charge une position depuis une chaîne FEN (réinitialise l'historique)."""
        parts = fen.split()
        self.board: list[list[Optional[str]]] = []
        for row_str in parts[0].split("/"):
            row: list[Optional[str]] = []
            for ch in row_str:
                if ch.isdigit():
                    row.extend([None] * int(ch))
                else:
                    row.append(ch)
            if len(row) != 8:
                raise ValueError(f"Rangée FEN invalide : {row_str!r}")
            self.board.append(row)
        if len(self.board) != 8:
            raise ValueError("FEN invalide : il faut 8 rangées")
        self.turn = parts[1]
        self.castling: set[str] = set(parts[2]) if parts[2] != "-" else set()
        self.ep_square = parse_square(parts[3]) if parts[3] != "-" else None
        self.halfmove_clock = int(parts[4]) if len(parts) > 4 else 0
        self.fullmove_number = int(parts[5]) if len(parts) > 5 else 1
        self._history: list[tuple] = []
        self._position_counts: dict = {self._position_key(): 1}

    def fen(self) -> str:
        """Position courante au format FEN."""
        rows = []
        for row in self.board:
            s, empty = "", 0
            for piece in row:
                if piece is None:
                    empty += 1
                else:
                    if empty:
                        s += str(empty)
                        empty = 0
                    s += piece
            if empty:
                s += str(empty)
            rows.append(s)
        castling = "".join(c for c in "KQkq" if c in self.castling) or "-"
        ep = square_name(*self.ep_square) if self.ep_square else "-"
        return (f"{'/'.join(rows)} {self.turn} {castling} {ep} "
                f"{self.halfmove_clock} {self.fullmove_number}")

    # ------------------------------------------------------- petits helpers

    @staticmethod
    def _color_of(piece: str) -> str:
        return WHITE if piece.isupper() else BLACK

    @staticmethod
    def _enemy(color: str) -> str:
        return BLACK if color == WHITE else WHITE

    def _position_key(self) -> tuple:
        # Clé de position pour la triple répétition. Note : l'en passant est
        # inclus tel quel (léger excès de prudence : certaines répétitions
        # théoriques avec un e.p. "fantôme" ne seront pas détectées).
        return (tuple(tuple(r) for r in self.board), self.turn,
                frozenset(self.castling), self.ep_square)

    def _find_king(self, color: str) -> tuple[int, int]:
        king = "K" if color == WHITE else "k"
        for r in range(8):
            for c in range(8):
                if self.board[r][c] == king:
                    return r, c
        raise ValueError(f"Roi {color} introuvable (position invalide)")

    # -------------------------------------------------------------- attaques

    def is_attacked(self, row: int, col: int, by: str) -> bool:
        """La case (row, col) est-elle attaquée par le camp `by` ?"""
        board = self.board
        # Pions : un pion blanc attaque vers le haut (row - 1).
        pawn, pawn_row = ("P", row + 1) if by == WHITE else ("p", row - 1)
        if 0 <= pawn_row < 8:
            for dc in (-1, 1):
                c = col + dc
                if 0 <= c < 8 and board[pawn_row][c] == pawn:
                    return True
        knight = "N" if by == WHITE else "n"
        for dr, dc in KNIGHT_OFFSETS:
            r, c = row + dr, col + dc
            if 0 <= r < 8 and 0 <= c < 8 and board[r][c] == knight:
                return True
        king = "K" if by == WHITE else "k"
        for dr, dc in KING_OFFSETS:
            r, c = row + dr, col + dc
            if 0 <= r < 8 and 0 <= c < 8 and board[r][c] == king:
                return True
        # Pièces à longue portée.
        rook_like = ("R", "Q") if by == WHITE else ("r", "q")
        bishop_like = ("B", "Q") if by == WHITE else ("b", "q")
        for dirs, attackers in ((ROOK_DIRS, rook_like), (BISHOP_DIRS, bishop_like)):
            for dr, dc in dirs:
                r, c = row + dr, col + dc
                while 0 <= r < 8 and 0 <= c < 8:
                    piece = board[r][c]
                    if piece is not None:
                        if piece in attackers:
                            return True
                        break
                    r, c = r + dr, c + dc
        return False

    def in_check(self, color: Optional[str] = None) -> bool:
        """Le roi du camp donné (par défaut le camp au trait) est-il en échec ?"""
        color = color or self.turn
        kr, kc = self._find_king(color)
        return self.is_attacked(kr, kc, self._enemy(color))

    # -------------------------------------------------- génération des coups

    def _pseudo_legal_moves(self, color: str) -> list[Move]:
        """Coups possibles sans vérifier si le roi reste en échec."""
        moves: list[Move] = []
        board = self.board
        for r in range(8):
            for c in range(8):
                piece = board[r][c]
                if piece is None or self._color_of(piece) != color:
                    continue
                kind = piece.upper()
                if kind == "P":
                    self._pawn_moves(r, c, color, moves)
                elif kind == "N":
                    for dr, dc in KNIGHT_OFFSETS:
                        self._step_move(r, c, r + dr, c + dc, color, moves)
                elif kind == "K":
                    for dr, dc in KING_OFFSETS:
                        self._step_move(r, c, r + dr, c + dc, color, moves)
                    self._castling_moves(color, moves)
                else:
                    dirs = {"R": ROOK_DIRS, "B": BISHOP_DIRS,
                            "Q": ROOK_DIRS + BISHOP_DIRS}[kind]
                    for dr, dc in dirs:
                        tr, tc = r + dr, c + dc
                        while 0 <= tr < 8 and 0 <= tc < 8:
                            target = board[tr][tc]
                            if target is None:
                                moves.append(Move(r, c, tr, tc))
                            else:
                                if self._color_of(target) != color:
                                    moves.append(Move(r, c, tr, tc))
                                break
                            tr, tc = tr + dr, tc + dc
        return moves

    def _step_move(self, r, c, tr, tc, color, moves) -> None:
        if 0 <= tr < 8 and 0 <= tc < 8:
            target = self.board[tr][tc]
            if target is None or self._color_of(target) != color:
                moves.append(Move(r, c, tr, tc))

    def _pawn_moves(self, r, c, color, moves) -> None:
        direction = -1 if color == WHITE else 1
        start_row = 6 if color == WHITE else 1
        promo_row = 0 if color == WHITE else 7

        def add(fr, fc, tr, tc):
            if tr == promo_row:
                for promo in ("q", "r", "b", "n"):
                    moves.append(Move(fr, fc, tr, tc, promo))
            else:
                moves.append(Move(fr, fc, tr, tc))

        one = r + direction
        if 0 <= one < 8 and self.board[one][c] is None:
            add(r, c, one, c)
            two = r + 2 * direction
            if r == start_row and self.board[two][c] is None:
                moves.append(Move(r, c, two, c))
        for dc in (-1, 1):
            tc = c + dc
            if not (0 <= one < 8 and 0 <= tc < 8):
                continue
            target = self.board[one][tc]
            if target is not None and self._color_of(target) != color:
                add(r, c, one, tc)
            elif (one, tc) == self.ep_square:
                moves.append(Move(r, c, one, tc))

    def _castling_moves(self, color: str, moves) -> None:
        row = 7 if color == WHITE else 0
        king = "K" if color == WHITE else "k"
        rook = "R" if color == WHITE else "r"
        if self.board[row][4] != king:
            return
        enemy = self._enemy(color)
        # Pas de roque si le roi est en échec.
        if self.is_attacked(row, 4, enemy):
            return
        kingside = "K" if color == WHITE else "k"
        queenside = "Q" if color == WHITE else "q"
        if (kingside in self.castling and self.board[row][7] == rook
                and self.board[row][5] is None and self.board[row][6] is None
                and not self.is_attacked(row, 5, enemy)
                and not self.is_attacked(row, 6, enemy)):
            moves.append(Move(row, 4, row, 6))
        if (queenside in self.castling and self.board[row][0] == rook
                and self.board[row][1] is None and self.board[row][2] is None
                and self.board[row][3] is None
                and not self.is_attacked(row, 2, enemy)
                and not self.is_attacked(row, 3, enemy)):
            moves.append(Move(row, 4, row, 2))

    def legal_moves(self) -> list[Move]:
        """Tous les coups légaux pour le camp au trait."""
        legal = []
        for move in self._pseudo_legal_moves(self.turn):
            self.make_move(move)
            # make_move a changé le trait : on vérifie le roi du camp qui vient de jouer.
            if not self.in_check(self._enemy(self.turn)):
                legal.append(move)
            self.undo_move()
        return legal

    def legal_moves_uci(self) -> list[str]:
        return [m.uci() for m in self.legal_moves()]

    def legal_moves_san(self) -> list[str]:
        legal = self.legal_moves()
        return [self.san(m, _legal=legal) for m in legal]

    # ---------------------------------------------------- exécution des coups

    def make_move(self, move: Move) -> None:
        """Applique un coup SANS vérifier sa légalité (usage interne / recherche).

        Pour jouer un coup validé, utiliser play(). Chaque make_move doit être
        annulable par undo_move().
        """
        board = self.board
        piece = board[move.fr][move.fc]
        snapshot = ([row[:] for row in board], self.turn, set(self.castling),
                    self.ep_square, self.halfmove_clock, self.fullmove_number)

        captured = board[move.tr][move.tc]
        is_pawn = piece.upper() == "P"

        # Prise en passant : le pion capturé n'est pas sur la case d'arrivée.
        if is_pawn and (move.tr, move.tc) == self.ep_square and captured is None:
            board[move.fr][move.tc] = None
            captured = "ep"

        board[move.fr][move.fc] = None
        if move.promotion:
            board[move.tr][move.tc] = (move.promotion.upper()
                                       if self._color_of(piece) == WHITE
                                       else move.promotion)
        else:
            board[move.tr][move.tc] = piece

        # Roque : déplacer aussi la tour.
        if piece.upper() == "K" and abs(move.tc - move.fc) == 2:
            row = move.fr
            if move.tc == 6:      # petit roque
                board[row][5], board[row][7] = board[row][7], None
            else:                 # grand roque
                board[row][3], board[row][0] = board[row][0], None

        # Mise à jour des droits de roque (case quittée ou case capturée).
        for sq in ((move.fr, move.fc), (move.tr, move.tc)):
            self.castling -= CASTLING_RIGHTS_LOST.get(sq, set())

        # Case en passant pour le coup suivant.
        if is_pawn and abs(move.tr - move.fr) == 2:
            self.ep_square = ((move.fr + move.tr) // 2, move.fc)
        else:
            self.ep_square = None

        self.halfmove_clock = 0 if (is_pawn or captured is not None) \
            else self.halfmove_clock + 1
        if self.turn == BLACK:
            self.fullmove_number += 1
        self.turn = self._enemy(self.turn)

        key = self._position_key()
        self._position_counts[key] = self._position_counts.get(key, 0) + 1
        self._history.append(snapshot + (key,))

    def undo_move(self) -> None:
        """Annule le dernier make_move()."""
        (board, turn, castling, ep, halfmove, fullmove, key) = self._history.pop()
        self.board = board
        self.turn = turn
        self.castling = castling
        self.ep_square = ep
        self.halfmove_clock = halfmove
        self.fullmove_number = fullmove
        self._position_counts[key] -= 1
        if self._position_counts[key] == 0:
            del self._position_counts[key]

    def play(self, move_str: str) -> str:
        """Joue un coup donné en UCI ('e2e4') ou SAN ('e4', 'Nf3', 'O-O').

        Lève ValueError si le coup est illégal ou la partie terminée.
        Retourne le coup en notation SAN.
        """
        if self.is_game_over():
            raise ValueError(f"La partie est terminée ({self.result()})")
        move = self.parse_move(move_str)
        san = self.san(move)
        self.make_move(move)
        self.moves_played.append((move.uci(), san))
        return san

    def undo_last(self) -> None:
        """Annule le dernier coup joué via play()."""
        if not self.moves_played:
            raise ValueError("Aucun coup à annuler")
        self.undo_move()
        self.moves_played.pop()

    def parse_move(self, move_str: str) -> Move:
        """Convertit une chaîne UCI ou SAN en Move légal (ValueError sinon).

        Le SAN est décomposé (pièce / désambiguïsation / arrivée / promotion)
        au lieu d'être comparé au SAN canonique : on accepte donc aussi les
        notations sur-spécifiées ('Rfg1' ou 'Rf1g1' là où 'Rg1' suffit), très
        répandues dans les PGN réels.
        """
        s = move_str.strip().replace("e.p.", "").strip().rstrip("+#!?")
        legal = self.legal_moves()

        # 1) Essai UCI : "e2e4", "e7e8q"
        if (len(s) in (4, 5) and s[0] in FILES and s[2] in FILES
                and s[1].isdigit() and s[3].isdigit()):
            uci = s.lower()
            for m in legal:
                if m.uci() == uci:
                    return m

        # 2) Roque, avec des O ou des zéros.
        nu = s.replace("0", "O").replace("-", "")
        if nu in ("OO", "OOO"):
            arrivee = 6 if nu == "OO" else 2
            for m in legal:
                if (self.board[m.fr][m.fc].upper() == "K"
                        and abs(m.tc - m.fc) == 2 and m.tc == arrivee):
                    return m
            raise ValueError(f"Roque illégal : {move_str!r}")

        # 3) SAN : promotion, puis pièce, puis case d'arrivée, le reste
        #    étant la désambiguïsation (colonne, rangée, ou les deux).
        corps, promotion = s, None
        if "=" in corps:
            corps, suffixe = corps.split("=", 1)
            promotion = suffixe[:1].lower()
        elif len(corps) > 2 and corps[-1] in "QRBN" and corps[-2].isdigit():
            promotion = corps[-1].lower()          # forme "e8Q", sans le '='
            corps = corps[:-1]

        if corps[:1] in ("K", "Q", "R", "B", "N"):
            piece, corps = corps[0], corps[1:]
        else:
            piece = "P"
        corps = corps.replace("x", "")

        arrivee = corps[-2:]
        if (len(arrivee) != 2 or arrivee[0] not in FILES
                or not arrivee[1].isdigit()):
            raise ValueError(f"Coup illisible : {move_str!r}")
        tr, tc = parse_square(arrivee)
        indices = corps[:-2]

        candidats = []
        for m in legal:
            if ((m.tr, m.tc) != (tr, tc)
                    or self.board[m.fr][m.fc].upper() != piece
                    or m.promotion != promotion):
                continue
            if all((c in FILES and m.fc == FILES.index(c))
                   or (c.isdigit() and m.fr == 8 - int(c))
                   for c in indices):
                candidats.append(m)

        if len(candidats) == 1:
            return candidats[0]
        if candidats:
            raise ValueError(f"Coup ambigu : {move_str!r}")
        raise ValueError(
            f"Coup illégal : {move_str!r}. "
            f"Coups légaux : {', '.join(self.san(m, _legal=legal) for m in legal)}")

    # ------------------------------------------------------------------- SAN

    def san(self, move: Move, _legal: Optional[list[Move]] = None) -> str:
        """Notation SAN d'un coup légal (avant de le jouer), suffixe +/# inclus."""
        piece = self.board[move.fr][move.fc]
        target = self.board[move.tr][move.tc]
        is_pawn = piece.upper() == "P"
        is_capture = target is not None or (is_pawn and (move.tr, move.tc) == self.ep_square)

        if piece.upper() == "K" and abs(move.tc - move.fc) == 2:
            base = "O-O" if move.tc == 6 else "O-O-O"
        elif is_pawn:
            base = (FILES[move.fc] + "x" if is_capture else "") \
                + square_name(move.tr, move.tc)
            if move.promotion:
                base += "=" + move.promotion.upper()
        else:
            legal = _legal if _legal is not None else self.legal_moves()
            same = [m for m in legal
                    if (m.tr, m.tc) == (move.tr, move.tc)
                    and (m.fr, m.fc) != (move.fr, move.fc)
                    and self.board[m.fr][m.fc] == piece]
            dis = ""
            if same:
                if all(m.fc != move.fc for m in same):
                    dis = FILES[move.fc]
                elif all(m.fr != move.fr for m in same):
                    dis = str(8 - move.fr)
                else:
                    dis = square_name(move.fr, move.fc)
            base = piece.upper() + dis + ("x" if is_capture else "") \
                + square_name(move.tr, move.tc)

        # Suffixe échec / mat.
        self.make_move(move)
        if self.in_check(self.turn):
            base += "#" if not self.legal_moves() else "+"
        self.undo_move()
        return base

    # --------------------------------------------------------- fin de partie

    def is_checkmate(self) -> bool:
        return self.in_check() and not self.legal_moves()

    def is_stalemate(self) -> bool:
        return not self.in_check() and not self.legal_moves()

    def is_insufficient_material(self) -> bool:
        """Roi seul, roi + 1 pièce mineure, ou fous tous sur la même couleur."""
        pieces = []
        for r in range(8):
            for c in range(8):
                p = self.board[r][c]
                if p is not None and p.upper() != "K":
                    pieces.append((p.upper(), (r + c) % 2))
        if not pieces:
            return True
        if any(kind in "PRQ" for kind, _ in pieces):
            return False
        if len(pieces) == 1:
            return True  # un seul cavalier ou fou
        if all(kind == "B" for kind, _ in pieces):
            return len({color for _, color in pieces}) == 1
        return False

    def is_fifty_moves(self) -> bool:
        return self.halfmove_clock >= 100

    def is_threefold_repetition(self) -> bool:
        return self._position_counts.get(self._position_key(), 0) >= 3

    def is_game_over(self) -> bool:
        return self.result() is not None

    def result(self) -> Optional[str]:
        """'1-0', '0-1', '1/2-1/2' si la partie est finie, None sinon."""
        if not self.legal_moves():
            if self.in_check():
                return "1-0" if self.turn == BLACK else "0-1"
            return "1/2-1/2"  # pat
        if (self.is_insufficient_material() or self.is_fifty_moves()
                or self.is_threefold_repetition()):
            return "1/2-1/2"
        return None

    def winner(self) -> Optional[str]:
        """'w', 'b' ou None (nulle ou partie en cours)."""
        return {"1-0": WHITE, "0-1": BLACK}.get(self.result())

    # ------------------------------------------------------------- affichage

    def __str__(self) -> str:
        lines = []
        for r in range(8):
            cells = " ".join(self.board[r][c] or "." for c in range(8))
            lines.append(f"{8 - r} {cells}")
        lines.append("  a b c d e f g h")
        trait = "blancs" if self.turn == WHITE else "noirs"
        lines.append(f"Trait aux {trait}")
        return "\n".join(lines)

    def unicode_board(self) -> str:
        """Plateau avec les symboles Unicode des pièces."""
        lines = []
        for r in range(8):
            cells = " ".join(
                UNICODE_PIECES.get(self.board[r][c], "·") if self.board[r][c] else "·"
                for c in range(8))
            lines.append(f"{8 - r} {cells}")
        lines.append("  a b c d e f g h")
        return "\n".join(lines)

    def pgn(self) -> str:
        """Liste des coups au format PGN, ex. '1. e4 e5 2. Nf3 Nc6'."""
        parts = []
        for i, (_, san) in enumerate(self.moves_played):
            if i % 2 == 0:
                parts.append(f"{i // 2 + 1}. {san}")
            else:
                parts.append(san)
        return " ".join(parts)

    def history_san(self) -> list[str]:
        return [san for _, san in self.moves_played]

    # ----------------------------------------------------------------- perft

    def perft(self, depth: int) -> int:
        """Compte les positions atteignables à `depth` demi-coups (test de règles)."""
        if depth == 0:
            return 1
        total = 0
        for move in self.legal_moves():
            self.make_move(move)
            total += self.perft(depth - 1)
            self.undo_move()
        return total


# --------------------------------------------------------------- auto-tests

if __name__ == "__main__":
    import time

    def check(label, got, expected):
        status = "OK " if got == expected else "ECHEC"
        print(f"[{status}] {label}: obtenu={got} attendu={expected}")
        assert got == expected, label

    t0 = time.time()

    # Perft depuis la position initiale (valeurs de référence connues).
    g = ChessGame()
    check("perft(1) initial", g.perft(1), 20)
    check("perft(2) initial", g.perft(2), 400)
    check("perft(3) initial", g.perft(3), 8902)

    # 'Kiwipete' : position de référence riche en roques, clouages, e.p.
    kiwi = ChessGame("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    check("perft(1) kiwipete", kiwi.perft(1), 48)
    check("perft(2) kiwipete", kiwi.perft(2), 2039)

    # Position de référence riche en promotions.
    promo = ChessGame("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
    check("perft(1) promotions", promo.perft(1), 6)
    check("perft(2) promotions", promo.perft(2), 264)

    # Prise en passant.
    g = ChessGame()
    for mv in ("e4", "a6", "e5", "d5"):
        g.play(mv)
    check("e.p. disponible", "exd6" in g.legal_moves_san(), True)
    g.play("exd6")
    check("e.p. : pion d5 retiré", g.board[parse_square("d5")[0]][parse_square("d5")[1]], None)

    # Mat du berger (accepte SAN et UCI).
    g = ChessGame()
    for mv in ("e2e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#"):
        g.play(mv)
    check("mat du berger", g.is_checkmate(), True)
    check("résultat 1-0", g.result(), "1-0")

    # Pat.
    g = ChessGame("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    check("pat détecté", g.is_stalemate(), True)
    check("résultat pat", g.result(), "1/2-1/2")

    # Roques des deux côtés.
    g = ChessGame("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    sans = g.legal_moves_san()
    check("petit roque possible", "O-O" in sans, True)
    check("grand roque possible", "O-O-O" in sans, True)
    g.play("O-O")
    check("tour en f1 après O-O", g.board[7][5], "R")

    # Triple répétition (navette de cavaliers).
    g = ChessGame()
    for mv in ("Nf3", "Nf6", "Ng1", "Ng8") * 2:
        g.play(mv)
    check("triple répétition", g.is_threefold_repetition(), True)

    # Règle des 50 coups et matériel insuffisant.
    check("50 coups", ChessGame("k7/8/8/8/8/8/8/7K w - - 100 1").is_fifty_moves(), True)
    check("K vs K", ChessGame("k7/8/8/8/8/8/8/7K w - - 0 1").is_insufficient_material(), True)
    check("K+B vs K", ChessGame("k7/8/8/8/8/8/8/5B1K w - - 0 1").is_insufficient_material(), True)
    check("K+Q vs K", ChessGame("k7/8/8/8/8/8/8/5Q1K w - - 0 1").is_insufficient_material(), False)

    # Aller-retour FEN.
    g = ChessGame()
    g.play("e4")
    check("FEN après 1.e4", g.fen(),
          "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")

    # Coup illégal rejeté proprement.
    g = ChessGame()
    try:
        g.play("Qh5")
        raise AssertionError("Qh5 aurait dû être rejeté au 1er coup")
    except ValueError:
        print("[OK ] coup illégal rejeté avec ValueError")

    print(f"\nTous les tests passent en {time.time() - t0:.1f}s")
