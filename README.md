# echecIA

*Un moteur d'échecs de type AlphaZero, pour jouer aux échecs et comprendre le fonctionnement de réseaux de neurones simples. Il prolonge mon projet Puissance 4 sur un jeu bien plus difficile, mais dont les IA sont bien mieux documentées.*

> **In English:** an AlphaZero-style chess engine written by hand (residual network, MCTS, self-play), at about 2090 Elo against strength-limited Stockfish. Full write-up in French below.

Un moteur d'échecs de type **AlphaZero** écrit à la main : réseau résiduel à deux têtes,
recherche arborescente de Monte-Carlo guidée par PUCT, pré-entraînement supervisé sur des
parties humaines, puis self-play avec sélection par arène.

**Force atteinte : ≈ 2090 ELO**, mesurés contre Stockfish 17.1 bridé à des niveaux
connus (échelle CCRL, pas FIDE).

## Ce que j'ai écrit moi-même

Le cœur du projet — [Stockfishdestroyer.py](Stockfishdestroyer.py), 866 lignes — est
entièrement de moi : réseau, recherche, entraînement, self-play, gating.

Deux briques d'infrastructure ont été déléguées à un assistant IA, conformément au
périmètre que j'avais fixé par écrit dans [Claude.md](Claude.md) *avant* de commencer :
la classe de règles du jeu ([chess_game.py](chess_game.py)) et le système de classement
([elo.py](elo.py)). Ce sont des composants de plomberie, sans rapport avec
l'apprentissage ; le reste du fichier de consignes interdit explicitement à l'assistant
d'écrire quoi que ce soit d'autre, « même partiellement, même à titre d'exemple
fonctionnel ».

## Architecture

**Réseau** — tronc convolutif à blocs résiduels (`blocresiduel`, nombre de couches
paramétrable) sur une représentation tensorielle de la position, avec deux têtes :
une **politique** sur les coups légaux et une **valeur** dans `[-1, 1]`.
Le masquage des coups illégaux est fait par lot, et un mécanisme de miroitage permet de
toujours présenter la position du point de vue du joueur au trait.

**Recherche** — MCTS avec sélection PUCT, expansion, remontée de la valeur le long du
chemin, et **évaluation par lots** (`taille_lot = 16`) pour amortir le coût des appels
au réseau. Virtual loss (`poser_vl` / `retirer_vl`) pour éviter que les simulations
d'un même lot explorent toutes la même branche.

**Entraînement** — deux régimes :

1. **Supervisé** — lecture en flux de PGN compressés (`.pgn.zst`) de Lichess et CCRL,
   avec filtrage par ELO minimum (1800 par défaut), génération d'exemples position →
   coup joué, et validation périodique.
2. **Self-play avec gating** — bruit de Dirichlet à la racine (`epsilon = 0.25`,
   `alpha = 0.3`), échantillonnage à température avec seuil de bascule au coup 30, puis
   **arène** : le nouveau réseau ne remplace le champion que s'il dépasse un seuil de
   victoires sur `k` parties. Les checkpoints `champion_*.pt` sont les survivants.

**Évaluation** — matchs contre Stockfish à profondeur variable et suivi ELO via
[elo.py](elo.py), qui implémente le facteur K variable et la persistance du classement.

## Résultats

### Pré-entraînement supervisé

**Étape A — Lichess 2014** (5 mois de parties, ELO ≥ 1800), 8 époques :

| époque | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| train | 3,053 | 2,556 | 2,360 | 2,263 | 2,193 |
| val | 2,667 | 2,528 | 2,409 | 2,370 | **2,348** |

**Étape B — CCRL 4040** (parties entre moteurs, plus fortes), 7 époques en repartant du
checkpoint précédent :

| époque | 1 | 3 | 5 | 7 |
|---|---|---|---|---|
| train | 2,432 | 2,257 | 2,179 | 2,123 |
| val | 2,373 | 2,309 | 2,294 | **2,285** |

La validation décroît de façon monotone sur les 15 époques, sans surapprentissage
(écart train/val de 0,16 en fin d'étape B). Le gain de l'étape B est modeste — 2,348 →
2,285 — ce qui suggère que le passage à des parties de moteurs apporte moins que prévu à
ce stade de l'entraînement.

### Self-play

**175 itérations, 1 750 parties, 36 185 s de calcul (≈ 10 h)**, avec reprise sur
checkpoint (le run a redémarré à l'itération 14). Le coût par itération croît
régulièrement, de 223 s à ~240 s, à mesure que le buffer se remplit.

### Force de jeu mesurée : ≈ 2090 ELO

ELO estimé par **maximum de vraisemblance logistique** sur des matchs contre Stockfish
17.1 bridé à plusieurs niveaux connus, qui servent d'ancres. Pour le réseau supervisé
(≈ 1612 ELO, tableau ci-dessous), l'intervalle de confiance se resserre avec le nombre de
parties : ±140 à 40 parties, ±76 à 100, ±55 à 180.

Détail de l'évaluation du réseau supervisé (180 parties) :

| adversaire | score | parties | V–N–D |
|---|---|---|---|
| Stockfish bridé 1320 | 78,8 % | 40 | 30–3–7 |
| Stockfish bridé 1500 | 65,0 % | 60 | 32–14–14 |
| Stockfish bridé 1600 | 55,0 % | 80 | 35–18–27 |

### Progression, étape par étape

| version | ELO |
|---|---|
| Self-play **sans** garde-fou | 1427 |
| Pré-entraînement Lichess 1800+ | 1612 |
| Imitation de moteurs CCRL 3000+ · 100 sims | 1837 |
| **CCRL 3000+ · 300 sims** | **2090** |

Deux marches nettes : **+225 ELO apportés par l'imitation de moteurs**, puis **+250 par
la seule augmentation de la recherche**, sans réentraîner le réseau.

### Ce que ça dit

**La recherche convertit directement en force.** À réseau identique, chaque doublement du
budget de simulations vaut environ **125 ELO** — 100 sims : 1837, 200 : 1966, 300 : 2090 —
puis ça plafonne : 400 simulations (2074) ne fait pas mieux que 300. La barre des 2000 est
franchie uniquement grâce à la recherche. Autrement dit, le réseau contenait déjà une
force que la recherche n'extrayait pas encore.

**Le self-play sans garde-fou dégrade le réseau** : 1427 contre 1612 pour le modèle
supervisé dont il partait, soit **−185 ELO**. C'est le run conservé dans
`selfplay_degrade_1419/`. C'est ce constat qui a motivé l'ajout du **gating** — un nouveau
réseau ne remplace le champion que s'il le bat en arène. Un mécanisme d'AlphaZero qu'on
peut lire comme un détail d'implémentation, et dont j'ai mesuré moi-même la nécessité.

*Leçon commune avec mon projet [Puissance 4](../Puissance4), où l'ELO de self-play
surestimait la force réelle de 150 à 190 points : un ELO mesuré contre une référence qui
progresse en même temps que soi ne veut rien dire. Seules les ancres extérieures et fixes
— ici Stockfish bridé — donnent un chiffre défendable.*

## Lancer

Jouer contre le meilleur réseau entraîné :

```bash
python jouer.py
```

Les poids entraînés (`stageB_ccrl3000.pt`) ne sont pas versionnés : ils sont trop
volumineux.

Les commandes de pré-entraînement et de self-play sont dans
[Stockfishdestroyer.py](Stockfishdestroyer.py) (`pre_entrainement`, `boucle_gatee`).

## Structure

```
Stockfishdestroyer.py   réseau, MCTS, entraînement supervisé, self-play, arène
chess_game.py           règles du jeu : coups légaux, échec/mat, FEN, PGN, perft
elo.py                  système ELO : facteur K variable, classement, persistance
jouer.py                partie contre un humain
barre.py                barre de progression
Claude.md               périmètre fixé à l'assistant IA avant le début du projet
```

Non versionnés (volumineux) : bases PGN Lichess et CCRL, checkpoints `.pt`, logs
d'entraînement, binaire Stockfish.

## Contexte

Ce moteur est conçu pour un tournoi opposant mon IA à celles de mes amis — d'où le
système ELO et le format de partie standardisé.

---

*Ce README a été rédigé avec l'aide de Claude, à partir de mes notes et de mes résultats. Le cœur du moteur et les expériences sont de moi ; les deux briques déléguées, les règles du jeu et le classement Elo, sont détaillées plus haut dans « Ce que j'ai écrit moi-même ».*
