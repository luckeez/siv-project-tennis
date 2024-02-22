from collections import namedtuple

Pos = namedtuple("Pos", ["x", "y"])

class Pallina:
    def __init__(self, x=0, y=0):
        """
        Inizializza una nuova pallina con le coordinate x e y a 0.
        """
        self.x = x
        self.y = y
    
    def set_pos(self, x, y):
        self.x = x
        self.y = y

# Esempio di utilizzo
pallina = Pallina()

pallina.set_pos(2, 1)

# Accedere alle coordinate
print(f"Coordinata x: {pallina.x}")
print(f"Coordinata y: {pallina.y}")