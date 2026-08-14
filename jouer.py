import Stockfishdestroyer as sd
net, _, _ = sd.charger("stageB_ccrl3000.pt", sd.device)
sd.jouer_contre_humain(net, sims=300, humain_blanc=True)
