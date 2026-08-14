"""Barre d'avancement pour terminal, pensée pour une boucle d'entraînement.

Aucune dépendance externe. On enveloppe la boucle sur les lots, on appelle
`update(...)` à chaque lot en lui passant les métriques à afficher (les pertes),
et la barre se redessine sur place. Exemple minimal en bas de fichier.
"""

import sys
import time
import shutil


class BarreAvancement:

    def __init__(self, total, description="", lissage=0.1,
                 largeur=None, flux=sys.stdout, min_intervalle=0.05):
        # `total` : nombre d'itérations attendues (ex : nombre de lots dans l'époque).
        self.total = max(1, total)
        self.description = description
        # `lissage` : facteur d'une moyenne mobile exponentielle sur les métriques.
        #   0.1 -> chaque nouvelle valeur pèse 10 %, l'affichage reste stable.
        #   0   -> pas de lissage, on affiche la valeur brute du dernier lot.
        self.lissage = lissage
        # `largeur` : longueur de la barre en caractères. None = adaptée au terminal.
        self.largeur = largeur
        self.flux = flux
        # On ne redessine pas plus souvent que `min_intervalle` secondes (anti-scintillement).
        self.min_intervalle = min_intervalle

        self.n = 0                      # nombre d'itérations déjà faites
        self.metriques = {}             # valeurs (lissées) actuellement affichées
        self.debut = time.perf_counter()
        self._dernier_render = 0.0

    def update(self, pas=1, **metriques):
        """Avance de `pas` itérations et met à jour les métriques affichées."""
        self.n += pas
        for cle, val in metriques.items():
            if cle in self.metriques and self.lissage > 0:
                # moyenne mobile exponentielle
                self.metriques[cle] = (1 - self.lissage) * self.metriques[cle] + self.lissage * val
            else:
                self.metriques[cle] = val

        maintenant = time.perf_counter()
        fini = self.n >= self.total
        # On redessine si assez de temps s'est écoulé, ou si on vient de finir.
        if maintenant - self._dernier_render >= self.min_intervalle or fini:
            self._dessiner(maintenant)
            self._dernier_render = maintenant

    @staticmethod
    def _format_temps(secondes):
        secondes = int(secondes)
        h, reste = divmod(secondes, 3600)
        m, s = divmod(reste, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _dessiner(self, maintenant):
        frac = min(1.0, self.n / self.total)
        ecoule = maintenant - self.debut
        debit = self.n / ecoule if ecoule > 0 else 0.0
        reste = (self.total - self.n) / debit if debit > 0 else 0.0

        colonnes = shutil.get_terminal_size((100, 20)).columns

        # métriques : "perte=1.234  pol=0.987  val=0.247"
        txt_metriques = "  ".join(f"{k}={v:.3f}" for k, v in self.metriques.items())

        # partie fixe (tout sauf la barre elle-même)
        gauche = f"{self.description} " if self.description else ""
        droite = (f" {self.n}/{self.total} {frac*100:5.1f}%  "
                  f"[{self._format_temps(ecoule)}<{self._format_temps(reste)}, {debit:.0f}/s]"
                  f"{'  ' + txt_metriques if txt_metriques else ''}")

        # largeur de la barre : soit imposée, soit ce qu'il reste dans le terminal
        if self.largeur is not None:
            larg = self.largeur
        else:
            larg = max(8, colonnes - len(gauche) - len(droite) - 3)  # -3 : "| |"

        rempli = int(frac * larg)
        barre = "█" * rempli + "░" * (larg - rempli)

        ligne = f"{gauche}|{barre}|{droite}"
        # \r : retour au début de ligne ; \033[K : efface jusqu'au bout (reste d'un rendu plus long)
        self.flux.write("\r" + ligne[:colonnes] + "\033[K")
        self.flux.flush()

    def close(self):
        """Dessine l'état final et passe à la ligne (à appeler en fin de boucle)."""
        self._dessiner(time.perf_counter())
        self.flux.write("\n")
        self.flux.flush()

    # Permet l'usage `with BarreAvancement(...) as barre:` qui ferme tout seul.
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


if __name__ == "__main__":
    # Démonstration : simule une époque de 60 lots.
    import random
    perte = 9.0
    with BarreAvancement(60, description="époque 1/10") as barre:
        for i in range(60):
            time.sleep(0.03)                      # simule le calcul d'un lot
            perte *= 0.97 + random.uniform(-0.02, 0.02)
            barre.update(perte=perte, pol=perte * 0.9, val=perte * 0.1)
