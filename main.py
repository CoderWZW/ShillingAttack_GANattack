from averageAttack import AverageAttack
from bandwagonAttack import BandWagonAttack
from randomAttack import RandomAttack
from unorganizedMaliciousAttacks import UMAttack


def onetimeUMattack():
    attack = UMAttack('config/config.conf')
    attack.insertSpam()
    attack.generateLabels('UMlabels.txt')
    attack.generateProfiles('UMprofiles.txt')
    
def onetimeAattack():
    attack = AverageAttack('config/config.conf')
    attack.insertSpam()
    attack.generateLabels('Alabels.txt')
    attack.generateProfiles('Aprofiles.txt')

def onetimeRattack():
    attack = RandomAttack('config/config.conf')
    attack.insertSpam()
    attack.generateLabels('Rlabels.txt')
    attack.generateProfiles('Rprofiles.txt')

def onetimeBattack():
    attack = BandWagonAttack('config/config.conf')
    attack.insertSpam()
    attack.generateLabels('Blabels.txt')
    attack.generateProfiles('Bprofiles.txt')

onetimeAattack()
onetimeRattack()
onetimeBattack()
onetimeUMattack()