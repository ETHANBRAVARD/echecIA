import torch
from torch import nn
import chess_game as cg
import math
import copy
import random 
import zstandard
import io
import re
import os
from barre import BarreAvancement 
import chess, chess.engine
import numpy
import time
from collections import deque
import glob


BIN = "stockfish/stockfish-ubuntu-x86-64-bmi2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Stockfishdestoyer ():


    def __init__(self):
       pass


    @staticmethod
    def plateau_vers_tenseur ( plateau ):
        tens=[[[0 for _ in range (8)]for _ in range (8)] for _ in range (18)]
        colone = 0
        for i in plateau.board :
            ligne = 0
            for j in i :
                if j == 'P' :
                    if plateau.turn == 'w':
                        tens[0][colone][ligne] = 1
                    else :
                        tens[6][7-colone][ligne] = 1
                if j == 'N' :
                    if plateau.turn == 'w':
                        tens[1][colone][ligne] = 1
                    else :
                        tens[7][7-colone][ligne] = 1
                if j == 'B' :
                    if plateau.turn == 'w':
                        tens[2][colone][ligne] = 1
                    else :
                        tens[8][7-colone][ligne] = 1
                if j == 'R' :
                    if plateau.turn == 'w':
                        tens[3][colone][ligne] = 1
                    else :
                        tens[9][7-colone][ligne] = 1
                if j == 'Q' : 
                    if plateau.turn == 'w':
                        tens[4][colone][ligne] = 1
                    else :
                        tens[10][7-colone][ligne] = 1
                if j == 'K' : 
                    if plateau.turn == 'w':
                        tens[5][colone][ligne] = 1
                    else :
                        tens[11][7-colone][ligne] = 1
                if j == 'p' :
                    if plateau.turn == 'w':
                        tens[6][colone][ligne] = 1
                    else :
                        tens[0][7-colone][ligne] = 1
                if j == 'n' :
                    if plateau.turn == 'w':
                        tens[7][colone][ligne] = 1
                    else :
                        tens[1][7-colone][ligne] = 1
                if j == 'b' :
                    if plateau.turn == 'w':
                        tens[8][colone][ligne] = 1
                    else :
                        tens[2][7-colone][ligne] = 1
                if j == 'r' :
                    if plateau.turn == 'w':
                        tens[9][colone][ligne] = 1
                    else :
                        tens[3][7-colone][ligne] = 1
                if j == 'q' : 
                    if plateau.turn == 'w':
                        tens[10][colone][ligne] = 1
                    else :
                        tens[4][7-colone][ligne] = 1
                if j == 'k' : 
                    if plateau.turn == 'w':
                        tens[11][colone][ligne] = 1
                    else :
                        tens[5][7-colone][ligne] = 1
                ligne += 1
            colone += 1
        if plateau.turn == 'w':
            k1 = 'K'
            q1 = 'Q'
            k2 = 'k'
            q2 = 'q'
        else : 
            k1 = 'k'
            q1 = 'q'
            k2 = 'K'
            q2 = 'Q'
        if k1 in plateau.castling:
            bit1 = 1
        else :
            bit1 = 0
        if q1 in plateau.castling:
            bit2 = 1
        else :
            bit2 = 0
        if k2 in plateau.castling:
            bit3 = 1
        else :
            bit3 = 0
        if q2 in plateau.castling:  
            bit4 = 1
        else :
            bit4 = 0
        tens[16] = [[bit4]*8]*8
        tens[15] = [[bit3]*8]*8
        tens[14] = [[bit2]*8]*8
        tens[13] = [[bit1]*8]*8
        tens[17] = [[0 for _ in range (8)]for _ in range (8)]
        if plateau.ep_square != None:
            colone,ligne = plateau.ep_square
            if plateau.turn == 'w':
                tens[17][colone][ligne] = 1
            else :
                tens[17][7-colone][ligne] = 1
        if plateau.turn == 'w':
            tens[12] = [[1 for i in range (8)] for i in range (8)]
        else :
            tens[12] = [[0 for i in range (8)] for i in range (8)]

        tens = torch.tensor(tens, dtype=torch.float32,device=device)
        tens = tens.unsqueeze(0)
        return tens


class blocresiduel(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv1=nn.Conv2d(channels, channels, kernel_size=3, stride=1,padding=1)
        self.conv2=nn.Conv2d(channels, channels, kernel_size=3, stride=1,padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_features=channels)
        self.bn2 = nn.BatchNorm2d(num_features=channels)


    def forward(self , x):
        a = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = a + self.bn2(x)
        x = self.relu(x)
        return(x)

class reseau(nn.Module):

    def __init__(self , nombre_de_couche):
        super().__init__()
        self.n = nombre_de_couche
        self.conv = nn.Conv2d(18,256,3,1,1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_features=256)
        self.blocs = nn.ModuleList()
        for i in range (self.n):
            self.blocs.append(blocresiduel(256))
        self.conv11 = nn.Conv2d(256,1,kernel_size=1)
        self.bn2=nn.BatchNorm2d(num_features=1)
        self.appl = nn.Flatten()
        self.lin = nn.Linear(64,256)
        self.lin2 = nn.Linear(256,1)
        self.tanh = nn.Tanh()
        self.convp1=nn.Conv2d(256,1,kernel_size=1)
        self.bnp1=nn.BatchNorm2d(num_features=1)
        self.applp1=nn.Flatten()
        self.linp1=nn.Linear(64,4168)
        self.softmaxp1=nn.Softmax(dim=1)


    def forward(self , x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.relu(x)
        for bloc in self.blocs:
            x = bloc(x)
        return(x)
    

    def valeur(self,x):
        x=self.conv11(x)
        x=self.bn2(x)
        x=self.relu(x)
        x=self.appl(x)
        x=self.lin(x)
        x=self.relu(x)
        x=self.lin2(x)
        x=self.tanh(x)
        return(x)
    

    def politique(self,x):
        x=self.convp1(x)
        x=self.bnp1(x)
        x=self.relu(x)
        x=self.applp1(x)
        x=self.linp1(x)
        return(x)
    
    @staticmethod
    def correspondance(move,turn):
        fr, fc, tr, tc = move.fr ,move.fc ,move.tr ,move.tc
        if turn == 'b':
            fr = 7-fr
            tr = 7-tr
        if move.promotion in ('r','b','n') :
            direction = tc - fc + 1
            piece = "rbn".index(move.promotion)
            indice    = 4096 + (fc*3 + direction)*3 + piece
        else :
            indice = (fr*8 + fc)*64 + (tr*8 + tc)     
        return(indice)
    

    def masquage (self,x,partie):
        logit = self.politique(x)
        masque=[-math.inf for i in range (4168)]
        for coup in partie.legal_moves():
            indice = self.correspondance(coup,partie.turn)
            masque[indice]=0
        logits_masques = logit + torch.tensor(masque,device=logit.device)
        return (self.softmaxp1(logits_masques))


    def masquage_par_lot(self, x, parties):
        logits = self.politique(x)                    # (T, 4168)
        masque = torch.full( (len(parties), 4168), -math.inf, device=device )   # (T, 4168) sur GPU
        for t, partie in enumerate(parties):
            for coup in partie.legal_moves():
                indice = self.correspondance(coup, partie.turn)
                masque[t][indice] = 0
        return self.softmaxp1(logits + masque)         # softmax déjà sur dim=1


    def de_miroitage(self,move,turn):
        if turn=='b':
            if move.promotion!=None:
                move2=cg.Move(7-move.fr ,move.fc,7-move.tr,move.tc,move.promotion)
            else:
                move2=cg.Move(7-move.fr ,move.fc,7-move.tr,move.tc)
        else:
            move2=move
        return(move2)



class Noeud:

    c=1.4

    def __init__(self,partie,profondeur):
        self.partie=partie
        self.profondeur=profondeur
        self.enfants={}
        self.visites=0
        self.P = None
        self.w=0
        self.result       = partie.result()
        self.coups_legaux = partie.legal_moves()
        self.turn         = partie.turn     


    def trouver_enfant(self, coup):
        n=self.enfants.get(coup)    
        return(n)                   
    

    def PUCT(self , reseau ):
        Pol=self.P
        coup_final=None
        score_puct = - math.inf
        COUP = self.coups_legaux
        for coup in COUP:
            enfant=self.trouver_enfant(coup)
            if enfant != None :
                if enfant.visites !=0:
                    moy = enfant.w/enfant.visites
                else:
                    moy = 0
                a = - moy + self.c * Pol[reseau.correspondance(coup, self.turn)] * math.sqrt(self.visites+1) / (1 + enfant.visites)
                if score_puct < a :
                    score_puct = a
                    coup_final=coup
            else:
                a = self.c * Pol[reseau.correspondance(coup, self.turn)] * math.sqrt(self.visites+1) / 1
                if score_puct < a: 
                    score_puct = a 
                    coup_final=coup
        return(score_puct,coup_final)
    

    def expansion(self, coup):
        partie2=copy.deepcopy(self.partie)
        partie2.make_move(coup)
        enfant = Noeud(partie2, self.profondeur+1)
        self.enfants[coup] = enfant     
        return(self.enfants[coup])
    

    def recherche(self, reseau, nombre_de_simulations, taille_lot = 16) :

        restant = nombre_de_simulations
        while restant > 0 :

            T = min(taille_lot, restant)
            en_attente = []                    # liste de (chemin, feuille)

            # ---------- phase A : collecte, zéro appel réseau ----------
            for _ in range (T):
                chemin, feuille, valeur = descendre(self, reseau)
                poser_vl(chemin)
                if valeur != None :            # terminal : résolu tout de suite
                    retirer_vl(chemin)
                    remonter(chemin, valeur)
                else :
                    en_attente.append((chemin, feuille))

            # ---------- phase B : un seul forward ----------
            if len(en_attente) != 0 :
                entree=[]
                partie=[]
                for (chemin,feuille) in en_attente:
                    partie.append(feuille.partie)
                    entree.append(Stockfishdestoyer.plateau_vers_tenseur(feuille.partie)) 
                with torch.no_grad() :
                    entree = torch.cat(entree, dim=0)
                    tronc     = reseau(entree)
                    valeurs   = reseau.valeur(tronc)                        
                    politiques = reseau.masquage_par_lot(tronc, partie) #a ecrire non ?

                # ---------- phase C : remontée ----------
                for i, (chemin, feuille) in enumerate(en_attente) :
                    feuille.P = politiques[i].tolist()
                    retirer_vl(chemin)
                    remonter(chemin, valeurs[i].item())

            restant = restant - T

    def meilleur_coup_final (self):
        meilleur_coup = None
        visite_max=-math.inf
        for coup , enfant in self.enfants.items() :
            if visite_max < enfant.visites :
                meilleur_coup = coup
                visite_max = enfant.visites
        return(meilleur_coup)
    
def poser_vl(chemin) :
    for noeud in chemin :
        noeud.visites += 1
        noeud.w      += 1

def retirer_vl(chemin) :
    for noeud in chemin :
        noeud.visites -= 1
        noeud.w      -= 1

def remonter(chemin, valeur) :          # c'est ta boucle actuelle, extraite telle quelle
    for  noeud in chemin[::-1] :
        noeud.visites += 1
        noeud.w      += valeur
        valeur = -valeur


def descendre(racine, reseau) :
    noeud   = racine
    chemin  = [racine]
    while True : #changer la condition d'arret
        resultat = noeud.result
        if resultat != None :
            if resultat == "1/2-1/2":
                valeur = 0 
            else:
                valeur =-1
            return (chemin, None, valeur)      # feuille terminale, pas de réseau
        if noeud.P is None :
            return (chemin, noeud, None)       # feuille à évaluer par le lot
        coup   = noeud.PUCT(reseau)[1]
        m=noeud.trouver_enfant(coup)
        if m !=None:
            enfant = noeud.trouver_enfant(coup)  
        else:
            enfant = noeud.expansion(coup)
        noeud  = enfant
        chemin.append(noeud)



def partie_entre_reseaux(net_blanc, net_noir, sims) :
    board = chess.Board()
    while (not board.is_game_over(claim_draw=True)) and board.fullmove_number < 300 :
        if board.turn == chess.WHITE :
            joueur = net_blanc 
        else:
            joueur=net_noir
        p = cg.ChessGame(board.fen())
        racine = Noeud(p, 0)
        racine.recherche(joueur, sims)              # PAS de preparer_racine : zero bruit
        coup = racine.meilleur_coup_final()         # argmax : competition
        board.push_uci(coup.uci())
    return board.result(claim_draw=True)            # "1-0" / "0-1" / "1/2-1/2" / "*"


def jouer_partie(reseau,fen=cg.START_FEN,simulations=30,coups_max=400):
    coups=0
    L=[]
    partie = cg.ChessGame(fen)
    racine = Noeud(partie, 0)
    while racine.partie.result() is None and coups < coups_max :
        preparer_racine(racine, reseau)
        racine.recherche(reseau, simulations)
        coup = coup_par_temperature(racine, coups)
        total = sum(enfant.visites for enfant in racine.enfants.values())
        π = []
        for coup_possible, enfant in racine.enfants.items() :
            indice = reseau.correspondance(coup_possible, racine.partie.turn)
            π.append( (indice, enfant.visites / total) )
        s = racine.partie.fen()
        L.append([s, π, 0])       # le 0 sera remplacé par z à la fin
        racine = racine.enfants[coup]
        coups += 1
    if racine.partie.result() == '1-0' :
        n=0
        for t in L :
            t[2] = 1*(-1)**n
            n+=1
    elif racine.partie.result() == '0-1' :
        n=0
        for t in L :
            t[2] = -1*(-1)**n
            n+=1
    return(L)


def perte(lot,reseau):
    B=len(lot)
    entree = torch.cat([ Stockfishdestoyer.plateau_vers_tenseur(cg.ChessGame(ex[0])) for ex in lot ],dim=0)
    tronc = reseau(entree)
    politique=reseau.politique(tronc)
    valeur_actuelle=reseau.valeur(tronc)
    cible_t = torch.zeros(len(lot), 4168, device=device)
    for b, ex in enumerate(lot) :
        for indice, proba in ex[1] :
            cible_t[b, indice] = proba
    perte_politique = -(cible_t * torch.log_softmax(politique, dim=1)).sum(dim=1).mean()
    cible_val = torch.tensor( [[ex[2]] for ex in lot], dtype=torch.float32 ,device=device )
    perte_valeur = ((valeur_actuelle -cible_val)**2).mean()
    return(perte_politique + perte_valeur , perte_politique , perte_valeur)
  

def entrainement(net, donnees, optimiseur, B, description="") :
    net.train()
    mélange = donnees[:]
    random.shuffle(mélange)
    nb_lots = (len(mélange) + B - 1) // B
    acclt = acclp = acclv = 0                       # totale, politique, valeur
    with BarreAvancement(nb_lots, description) as barre :
        for debut in range(0, len(mélange), B) :
            lot = mélange[debut : debut+B]
            optimiseur.zero_grad()
            lt, lp, lv = perte(lot, net)         # perte renvoie déjà 3 valeurs
            lt.backward()
            optimiseur.step()
            acclt+= lt.item()
            acclp+=lp.item()
            acclv+=lv.item()       # avec .item() pour ne pas garder le graphe
            barre.update(perte=lt.item(), pol=lp.item(), val=lv.item())
    return (acclt / nb_lots , acclp/ nb_lots , acclv/ nb_lots)

        

def lectureZcompressed(chemin,a,elomin=1800):
    compteur=0
    lues=0
    with open(chemin, 'rb') as f:                 
        brut   = zstandard.ZstdDecompressor().stream_reader(f)  
        texte  = io.TextIOWrapper(brut, encoding='utf-8')
        entetes={}
        for ligne in texte :
            if compteur >= a:
                break
            if ligne.startswith('[') :
                m      = ligne.split('"')      
                tag    = m[0][1:].strip()      
                valeur = m[1]                  
                entetes[tag] = valeur
            elif ligne.strip() == '' :
                continue
            else :
                if garder(entetes,elomin=1800):
                    yield entetes, ligne
                    compteur+=1
                entetes = {}
                lues+=1


def garder(entetes,elomin=1800):
    try:
        if entetes.get('TimeControl','0')!="-":
            if int(entetes.get("WhiteElo", "0"))>=elomin and int(entetes.get("BlackElo", "0"))>=elomin and int(entetes.get('TimeControl','0').split('+')[0])>=180 and entetes.get('Termination',"")=="Normal" and entetes.get("Result","*")!="*":  
                return(True)
            return(False)
        else:
            return(False)
    except ValueError:
        return False
    

def nettoyage(texte):
    netoyé=re.sub(r'\{[^}]*\}', ' ', texte).split()
    return([ j  for j in netoyé  if not j[0].isdigit() ])


def sauvegarder(net, optimiseur, iteration, chemin) :
    poids =	net.state_dict()
    poids_optimiseur = optimiseur.state_dict()
    n_couches = net.n
    etat = { 'poids':poids, 'poids_optimiseur':poids_optimiseur, 'iteration':iteration, 'n_couches':n_couches}
    torch.save(etat, chemin + ".tmp")
    os.replace(chemin + ".tmp", chemin)


def charger(chemin, device) :
    etat = torch.load(chemin, map_location=device)
    net = reseau(etat["n_couches"]).to(device)
    net.load_state_dict(etat["poids"])
    optimiseur = torch.optim.Adam(net.parameters())      # construit AVANT le load
    optimiseur.load_state_dict(etat["poids_optimiseur"])
    return net, optimiseur, etat["iteration"]


def generateur_exemple(chemin,a,nombre_de_positions,elomin=1800,lecteur=lectureZcompressed):
    exemples = []
    for entetes, movetext in lecteur(chemin,a,elomin) :
        coups = nettoyage(movetext)
        resultat = entetes["Result"]
        partie = cg.ChessGame()
        tampon = []
        try :
            for san in coups :
                coup   = partie.parse_move(san)
                s      = partie.fen()     # AVANT make_move
                indice = reseau.correspondance(coup, partie.turn)
                tampon.append( (s, indice, partie.turn) )
                partie.make_move(coup)
        except Exception :
            continue          # PGN abîmé -> partie jetée
        for (s, indice, trait) in tampon :
            if resultat=='1-0':
                if trait == 'w':
                    valeur_relative=1
                else:
                    valeur_relative=-1
            elif resultat=='0-1':
                if trait == 'w':
                    valeur_relative=-1
                else:
                    valeur_relative=1
            elif resultat=='1/2-1/2':
                valeur_relative=0
            z  = valeur_relative  # +1/-1/0 selon le trait
            pi = [(indice, 1.0)]
            exemples.append([s, pi, z])
        if len(exemples) >= nombre_de_positions : 
            break
    return exemples


def validation(net, donnees, B) :
    net.eval()
    acclt = acclp = acclv = 0
    nb_lots = (len(donnees) + B - 1) // B
    with torch.no_grad() :
        for debut in range(0, len(donnees), B) :
            lot = donnees[debut : debut+B]
            lt, lp, lv = perte(lot, net)
            acclt += lt.item()  ;  acclp += lp.item()  ;  acclv += lv.item()
    return (acclt/nb_lots, acclp/nb_lots, acclv/nb_lots)



def pre_entrainement(chemins, a, nb_positions, B, E, n_couches,elomin) :   # chemins = liste
    net = reseau(n_couches).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    donnees = []
    for ch in chemins :                          # <-- la boucle
        donnees += generateur_exemple(ch, a, nb_positions,elomin)

    random.shuffle(donnees)
    n_val = len(donnees) // 10
    val, train = donnees[:n_val], donnees[n_val:]

    for e in range(E) :
        t = entrainement(net, train, opt, B, f"époque {e+1}/{E}")
        v = validation(net, val, B)
        print(f"époque {e+1} : train {t[0]:.3f}  val {v[0]:.3f}")
        sauvegarder(net, opt, e, f"checkpoint_{e}.pt")
    return net


def une_partie(net, engine, sims, net_en_blanc) :
    board = chess.Board()
    if net_en_blanc:
        couleur_net = chess.WHITE 
    else :
        couleur_net=chess.BLACK

    while not board.is_game_over(claim_draw=True) and board.fullmove_number < 200 :
        if board.turn == couleur_net :
            # --- coup du réseau ---
            p = cg.ChessGame(board.fen())
            racine = Noeud(p, 0)
            racine.recherche(net, sims)
            coup = racine.meilleur_coup_final()       # un cg.Move
            board.push_uci(coup.uci())                # on le rejoue cote python-chess
        else :
            # --- coup de Stockfish ---
            coup_sf = engine.play(board, chess.engine.Limit(time=0.1)).move
            board.push(coup_sf)

    return board.result(claim_draw=True)              # "1-0" / "0-1" / "1/2-1/2"

def plusieurs_parties(net,engine,n):
    resultats = []
    for i in range(n) :
        r = une_partie(net, engine, sims=100, net_en_blanc=(i % 2 == 0))
        resultats.append( (r, i % 2 == 0) )      # garde le resultat ET la couleur du reseau
    return (resultats)


def preparer_racine(racine, reseau, epsilon=0.25, alpha=0.3) :
    # 1. s'assurer que P existe (prior du reseau)
    if racine.P is None :
        with torch.no_grad():
            tronc = reseau(Stockfishdestoyer.plateau_vers_tenseur(racine.partie))
            racine.P = reseau.masquage(tronc, racine.partie)[0].tolist()

    # 2. bruit de Dirichlet sur les coups legaux
    coups   = racine.coups_legaux                         # deja en cache
    indices = [reseau.correspondance(c, racine.turn) for c in coups]
    bruit   = numpy.random.dirichlet([alpha] * len(indices))

    # 3. melange dans P, aux indices des coups legaux
    for k, idx in enumerate(indices) :
        racine.P[idx] = (1-epsilon)*racine.P[idx] + epsilon*bruit[k]


def coup_par_temperature(racine, coups, seuil=30) :
    if coups >= seuil :
        return racine.meilleur_coup_final()        # fin de partie : argmax

    coups_possibles = list(racine.enfants.keys())
    visites = [racine.enfants[c].visites for c in coups_possibles]
    total   = sum(visites)
    probs   = [v/total for v in visites]
    idx     = numpy.random.choice(len(coups_possibles), p=probs)   # tire un INDICE
    return coups_possibles[idx]
        

def boucle_selfplay(net, optimiseur, duree_max, nb_parties_par_iteration, iteration, B, nb_pas) :
    buffer = deque(maxlen=30000)
    debut = time.time()
    parties_totales = iteration * nb_parties_par_iteration
    prochain_jalon  = 10
    while prochain_jalon <= parties_totales :
        prochain_jalon *= 10
    while time.time() - debut < duree_max :
        net.eval()
        with BarreAvancement(nb_parties_par_iteration, f"self-play it.{iteration+1}") as barre :
            for _ in range(nb_parties_par_iteration) :
                buffer.extend(jouer_partie(net, simulations=100))
                parties_totales += 1                          # <-- REMETTRE
                if parties_totales >= prochain_jalon :        # <-- dans le for, bien indente
                    sauvegarder(net, optimiseur, iteration, f"selfplay_{prochain_jalon}parties.pt")
                    prochain_jalon *= 10
                barre.update(taille_buffer=len(buffer))
        if len(buffer) >= B :
            net.train()
            for _ in range(nb_pas) :
                lot = random.sample(buffer, B)
                optimiseur.zero_grad()
                perte(lot, net)[0].backward()
                optimiseur.step()
        iteration += 1
        sauvegarder(net, optimiseur, iteration, "selfplay_dernier.pt")
        print(f"iteration {iteration} : {parties_totales} parties, {time.time()-debut:.0f}s", flush=True)


def arene(net_A, net_B, k, sims) :
    score_A = 0
    for i in range(k) :
        if i % 2 == 0 :                             # A joue blanc
            r = partie_entre_reseaux(net_A, net_B, sims)
            if r=="1-0":
                score_A += 1
            elif r=="1/2-1/2" or r=="*":
                score_A+=0.5  
            else:
                score_A+=0
        else :                                      # A joue noir
            r = partie_entre_reseaux(net_B, net_A, sims)
            if r=="0-1":
                score_A += 1 
            elif r=="1/2-1/2" or r=="*":
                score_A+=0.5 
            else:
                score_A+=0
    return score_A / k


def boucle_gatee(best, net, optimiseur, duree_max, nb_parties_par_iteration,B, nb_pas, periode_gating, k_arene, seuil, sims) :
    buffer = deque(maxlen=30000)
    debut = time.time()
    iteration = 0
    while time.time() - debut < duree_max :

        # --- generation : TOUJOURS avec le champion 'best' ---
        best.eval()
        with BarreAvancement(nb_parties_par_iteration, f"gen it.{iteration+1}") as barre :
            for _ in range(nb_parties_par_iteration) :
                buffer.extend(jouer_partie(best, simulations=sims))   # jouer_partie = AVEC bruit+temperature
                barre.update(buffer=len(buffer))

        # --- entrainement du challenger 'net' ---
        if len(buffer) >= B :
            net.train()
            for _ in range(nb_pas) :
                lot = random.sample(buffer, B)
                optimiseur.zero_grad()
                perte(lot, net)[0].backward()
                optimiseur.step()

        iteration += 1

        # --- gating periodique ---
        if iteration % periode_gating == 0 :
            net.eval()
            score = arene(net, best, k_arene, sims)
            if score >= seuil :                                   # ex. 0.55
                best.load_state_dict(net.state_dict())            # PROMOTION
                sauvegarder(best, optimiseur, iteration, f"champion_{iteration}.pt")
                print(f"PROMOTION it {iteration} : challenger {score:.0%} -> nouveau champion", flush=True)
            else :
                print(f"it {iteration} : challenger rejete ({score:.0%} < {seuil:.0%})", flush=True)

        sauvegarder(net, optimiseur, iteration, "gating_net.pt")   # reprise du challenger


def garder_ccrl(entetes, elomin):
    try:
        return (int(entetes.get("WhiteElo","0")) >= elomin
                and int(entetes.get("BlackElo","0")) >= elomin
                and entetes.get("Result","*") != "*")
    except ValueError:
        return False


def lecture_ccrl(chemin, a, elomin):
    compteur = 0
    with open(chemin, 'r', encoding='utf-8', errors='ignore') as f:
        entetes = {}
        movetext = ""
        for ligne in f:
            if compteur >= a: break
            s = ligne.strip()
            if s.startswith('[') :
                if movetext :                       # une partie vient de se terminer
                    if garder_ccrl(entetes, elomin):
                        yield entetes, movetext
                        compteur += 1
                    entetes = {} ; movetext = ""
                m = ligne.split('"') ; entetes[m[0][1:].strip()] = m[1]
            elif s == "" :
                continue
            else :
                movetext += " " + s                 # accumuler les lignes de coups
        if movetext and garder_ccrl(entetes, elomin):   # derniere partie
            yield entetes, movetext


def entrainement_supervise(net, opt, chemins, a, nb_positions, elomin, lecteur, B, duree_max, nom_sortie) :

    # (1) generation avec le bon filtre + lecteur
    donnees = []
    for ch in chemins :
        donnees += generateur_exemple(ch, a, nb_positions, elomin, lecteur)
    random.shuffle(donnees)
    n_val = len(donnees) // 10
    val, train = donnees[:n_val], donnees[n_val:]

    # (2) boucle bornee par le TEMPS, garde le MEILLEUR val
    debut = time.time()
    meilleure_val = float('inf')
    e = 0
    while time.time() - debut < duree_max :
        t = entrainement(net, train, opt, B, f"{nom_sortie} ep{e+1}")
        v = validation(net, val, B)
        print(f"{nom_sortie} ep{e+1} : train {t[0]:.3f}  val {v[0]:.3f}", flush=True)
        if v[0] < meilleure_val :
            meilleure_val = v[0]
            sauvegarder(net, opt, e, nom_sortie)        # sauve UNIQUEMENT quand val s'ameliore
        e += 1

    # (3) recharge le meilleur dans net (pour chainer depuis le meilleur, pas l'overfit)
    net.load_state_dict(torch.load(nom_sortie)["poids"])
    return net


def jouer_contre_humain(net, sims=300, humain_blanc=True):
    net.eval()
    partie = cg.ChessGame()
    while partie.result() is None:
        print(partie.unicode_board())                  # affiche le plateau
        if (partie.turn == 'w') == humain_blanc:
            # --- ton tour ---
            while True:
                coup_str = input("Ton coup (ex: e4, Nf3, e2e4) : ")
                try:
                    coup = partie.parse_move(coup_str)  # accepte SAN et UCI
                    break
                except Exception:
                    print("coup invalide, réessaie")
            partie.make_move(coup)
        else:
            # --- tour du réseau ---
            racine = Noeud(partie, 0)
            racine.recherche(net, sims)                 # PAS de preparer_racine (compétition)
            coup = racine.meilleur_coup_final()
            print("Le réseau joue :", coup.uci())
            partie.make_move(coup)
    print("Résultat :", partie.result())

if __name__ == "__main__" :
    ccrl = glob.glob("CCRL-4040*.pgn")
    net, opt, _ = charger("pretraine_lichess2014_5mois_e8.pt", device)
    for g in opt.param_groups : g['lr'] = 5e-4
    entrainement_supervise(net, opt, ccrl, a=100000000, nb_positions=12000000,
                           elomin=3000, lecteur=lecture_ccrl, B=256,
                           duree_max=8*3600, nom_sortie="stageB_ccrl3000.pt")

 



    
