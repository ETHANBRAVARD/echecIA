"""
elo.py — Système de classement ELO pour le tournoi d'IA.

Fournit :
- le calcul ELO classique (score attendu + mise à jour avec facteur K)
- une classe EloSystem qui gère les joueurs, enregistre les parties,
  tient le classement et se sauvegarde en JSON

Utilisation minimale :

    elo = EloSystem()
    elo.add_player("MonIA")
    elo.add_player("IA_de_Paul")
    elo.record_game("MonIA", "IA_de_Paul", "1-0")   # blancs, noirs, résultat
    print(elo.table())
    elo.save("classement.json")

Le résultat d'une partie est exprimé du point de vue des BLANCS :
"1-0" (ou 1), "0-1" (ou 0), "1/2-1/2" (ou 0.5). C'est exactement ce que
retourne ChessGame.result(), donc les deux briques se branchent directement :

    elo.record_game(nom_blanc, nom_noir, game.result())

Facteur K (inspiré des règles FIDE) :
- 40 tant qu'un joueur a moins de 30 parties (classement provisoire)
- 20 ensuite
- 10 dès qu'un joueur a atteint 2400
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

DEFAULT_RATING = 1200

RESULT_SCORES = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probabilité de gain attendue du joueur A contre le joueur B.

    Formule ELO : E = 1 / (1 + 10^((Rb - Ra) / 400)).
    A 0 d'écart -> 0.5 ; à +400 d'écart -> ~0.91.
    """
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


class Player:
    """Un participant du tournoi : nom, classement et statistiques."""

    def __init__(self, name: str, rating: float = DEFAULT_RATING):
        self.name = name
        self.rating = float(rating)
        self.games = 0
        self.wins = 0
        self.draws = 0
        self.losses = 0
        # Classement après chaque partie (index 0 = classement initial).
        self.rating_history: list[float] = [float(rating)]

    @property
    def score(self) -> float:
        """Points marqués (1 par victoire, 0.5 par nulle)."""
        return self.wins + 0.5 * self.draws

    def to_dict(self) -> dict:
        return {
            "name": self.name, "rating": self.rating, "games": self.games,
            "wins": self.wins, "draws": self.draws, "losses": self.losses,
            "rating_history": self.rating_history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Player":
        player = cls(data["name"], data["rating"])
        player.games = data["games"]
        player.wins = data["wins"]
        player.draws = data["draws"]
        player.losses = data["losses"]
        player.rating_history = data["rating_history"]
        return player

    def __repr__(self) -> str:
        return f"Player({self.name!r}, elo={self.rating:.0f}, parties={self.games})"


class EloSystem:
    """Gère les classements ELO d'un groupe de joueurs (le tournoi)."""

    def __init__(self, default_rating: float = DEFAULT_RATING,
                 k_new: float = 40, k_standard: float = 20, k_master: float = 10,
                 provisional_games: int = 30, master_threshold: float = 2400):
        self.default_rating = default_rating
        self.k_new = k_new
        self.k_standard = k_standard
        self.k_master = k_master
        self.provisional_games = provisional_games
        self.master_threshold = master_threshold
        self.players: dict[str, Player] = {}
        # Journal des parties : dicts {white, black, result, delta_white, ...}
        self.games_log: list[dict] = []

    # ------------------------------------------------------------- joueurs

    def add_player(self, name: str, rating: Optional[float] = None) -> Player:
        """Inscrit un joueur (ValueError si le nom existe déjà)."""
        if name in self.players:
            raise ValueError(f"Le joueur {name!r} existe déjà")
        player = Player(name, rating if rating is not None else self.default_rating)
        self.players[name] = player
        return player

    def get(self, name: str) -> Player:
        if name not in self.players:
            raise ValueError(f"Joueur inconnu : {name!r}. "
                             f"Inscrits : {', '.join(self.players) or '(aucun)'}")
        return self.players[name]

    def k_factor(self, player: Player) -> float:
        """Facteur K : plus il est grand, plus le classement bouge vite."""
        if player.games < self.provisional_games:
            return self.k_new
        if player.rating >= self.master_threshold:
            return self.k_master
        return self.k_standard

    # ------------------------------------------------------------- parties

    def record_game(self, white: str, black: str, result) -> tuple[float, float]:
        """Enregistre une partie et met à jour les deux classements.

        `result` : "1-0", "0-1", "1/2-1/2" (ou 1, 0, 0.5 = score des blancs).
        Retourne (delta_blancs, delta_noirs).
        """
        if white == black:
            raise ValueError("Un joueur ne peut pas jouer contre lui-même")
        if isinstance(result, str):
            if result not in RESULT_SCORES:
                raise ValueError(f"Résultat invalide : {result!r} "
                                 f"(attendu : 1-0, 0-1 ou 1/2-1/2)")
            score_white = RESULT_SCORES[result]
        else:
            score_white = float(result)
            if score_white not in (0.0, 0.5, 1.0):
                raise ValueError(f"Score invalide : {result!r} (attendu 0, 0.5 ou 1)")

        p_white, p_black = self.get(white), self.get(black)

        # Les deux mises à jour utilisent les classements d'AVANT la partie.
        exp_white = expected_score(p_white.rating, p_black.rating)
        exp_black = 1.0 - exp_white
        delta_white = self.k_factor(p_white) * (score_white - exp_white)
        delta_black = self.k_factor(p_black) * ((1.0 - score_white) - exp_black)

        p_white.rating += delta_white
        p_black.rating += delta_black
        for player, score in ((p_white, score_white), (p_black, 1.0 - score_white)):
            player.games += 1
            if score == 1.0:
                player.wins += 1
            elif score == 0.5:
                player.draws += 1
            else:
                player.losses += 1
            player.rating_history.append(player.rating)

        self.games_log.append({
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "white": white, "black": black, "score_white": score_white,
            "delta_white": round(delta_white, 2), "delta_black": round(delta_black, 2),
            "rating_white": round(p_white.rating, 2),
            "rating_black": round(p_black.rating, 2),
        })
        return delta_white, delta_black

    # ---------------------------------------------------------- classement

    def standings(self) -> list[Player]:
        """Joueurs triés par classement décroissant (puis par points marqués)."""
        return sorted(self.players.values(),
                      key=lambda p: (-p.rating, -p.score, p.name))

    def table(self) -> str:
        """Classement du tournoi sous forme de tableau texte."""
        header = f"{'#':>2}  {'Joueur':<20} {'ELO':>6} {'Parties':>7} " \
                 f"{'V':>3} {'N':>3} {'D':>3} {'Points':>6}"
        lines = [header, "-" * len(header)]
        for i, p in enumerate(self.standings(), start=1):
            lines.append(f"{i:>2}  {p.name:<20} {p.rating:>6.0f} {p.games:>7} "
                         f"{p.wins:>3} {p.draws:>3} {p.losses:>3} {p.score:>6.1f}")
        return "\n".join(lines)

    # --------------------------------------------------------- persistance

    def save(self, path: str) -> None:
        """Sauvegarde joueurs + journal des parties en JSON."""
        data = {
            "default_rating": self.default_rating,
            "k_new": self.k_new, "k_standard": self.k_standard,
            "k_master": self.k_master,
            "provisional_games": self.provisional_games,
            "master_threshold": self.master_threshold,
            "players": [p.to_dict() for p in self.players.values()],
            "games_log": self.games_log,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "EloSystem":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        system = cls(default_rating=data["default_rating"], k_new=data["k_new"],
                     k_standard=data["k_standard"], k_master=data["k_master"],
                     provisional_games=data["provisional_games"],
                     master_threshold=data["master_threshold"])
        for pdata in data["players"]:
            system.players[pdata["name"]] = Player.from_dict(pdata)
        system.games_log = data["games_log"]
        return system


# --------------------------------------------------------------- auto-tests

if __name__ == "__main__":
    import os
    import tempfile

    def check(label, got, expected):
        status = "OK " if got == expected else "ECHEC"
        print(f"[{status}] {label}: obtenu={got} attendu={expected}")
        assert got == expected, label

    # Scores attendus.
    check("E(1200 vs 1200)", expected_score(1200, 1200), 0.5)
    check("E(1600 vs 1200) ~ 0.91", round(expected_score(1600, 1200), 2), 0.91)

    # Victoire à égalité de classement, K=40 -> +20 / -20.
    elo = EloSystem()
    elo.add_player("A")
    elo.add_player("B")
    dw, db = elo.record_game("A", "B", "1-0")
    check("delta blancs +20", round(dw, 1), 20.0)
    check("delta noirs -20", round(db, 1), -20.0)
    check("ELO de A", round(elo.get("A").rating), 1220)
    check("ELO de B", round(elo.get("B").rating), 1180)

    # Nulle : le mieux classé perd des points.
    dw, db = elo.record_game("A", "B", "1/2-1/2")
    check("nulle : A perd des points", dw < 0, True)
    check("nulle : B gagne des points", db > 0, True)

    # Somme des variations nulle quand K identique.
    check("somme des deltas ~ 0", round(dw + db, 6), 0.0)

    # Stats et classement.
    check("parties de A", elo.get("A").games, 2)
    check("A premier au classement", elo.standings()[0].name, "A")

    # Résultat numérique accepté.
    elo.record_game("B", "A", 0.5)
    check("3 parties dans le journal", len(elo.games_log), 3)

    # Sauvegarde / rechargement.
    tmp = os.path.join(tempfile.gettempdir(), "elo_test.json")
    elo.save(tmp)
    elo2 = EloSystem.load(tmp)
    check("rechargement : ELO de A", elo2.get("A").rating, elo.get("A").rating)
    check("rechargement : journal", len(elo2.games_log), 3)
    os.remove(tmp)

    print("\n" + elo.table())
    print("\nTous les tests passent")
