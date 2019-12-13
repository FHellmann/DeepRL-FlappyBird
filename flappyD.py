from itertools import cycle
import random
import sys
import numpy as np

import agent as a
from gym import spaces

import pygame
from pygame.locals import *


FPS = 30
SCREENWIDTH  = 288
SCREENHEIGHT = 512
PIPEGAPSIZE  = 100 # gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79
# image, sound and hitmask  dicts
IMAGES, SOUNDS, HITMASKS = {}, {}, {}

FLAP = 1
GLIDE = 0
ACTIONS = [FLAP, GLIDE]

discount_factor = 0.99
action_space = spaces.Discrete(len(ACTIONS))

agent = a.DQLearningAgent(3, action_space.n)

# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    # blue bird
    (
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    # yellow bird
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)


def main():
    global SCREEN, FPSCLOCK
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption('Flappy Bird')

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    IMAGES['dead'] = pygame.image.load('assets/sprites/deadbird-midflap.png').convert_alpha()

    # game over sprite
    IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
    # message sprite for welcome screen
    IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    # sounds
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    SOUNDS['die']    = pygame.mixer.Sound('assets/audio/die' + soundExt)
    SOUNDS['hit']    = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    SOUNDS['point']  = pygame.mixer.Sound('assets/audio/point' + soundExt)
    SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    SOUNDS['wing']   = pygame.mixer.Sound('assets/audio/wing' + soundExt)

    # while True:
        # select random background sprites
    randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
    IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

    # select random player sprites
    randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
    IMAGES['player'] = (
        pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
        pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
        pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
    )

    # select random pipe sprites
    pipeindex = random.randint(0, len(PIPES_LIST) - 1)
    IMAGES['pipe'] = (
        pygame.transform.flip(
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), False, True),
        pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
    )

    # hismask for pipes
    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )

    # hitmask for player
    HITMASKS['player'] = (
        getHitmask(IMAGES['player'][0]),
        getHitmask(IMAGES['player'][1]),
        getHitmask(IMAGES['player'][2]),
    )

    movementInfo = initGame()
    mainGame(movementInfo)


def initGame():

    playerShmVals = {'val': 0, 'dir': 1}
    playerIndexGen = cycle([0, 1, 2, 1])
    playerx = int(SCREENWIDTH * 0.2)
    playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)

    basex = 0

    return {
        'playery': playery + playerShmVals['val'],
        'basex': basex,
        'playerIndexGen': playerIndexGen,
    }

def mainGame(movementInfo):
    score = playerIndex = loopIter =  highscore = 0
    playerIndexGen = movementInfo['playerIndexGen']
    playerx, playery = int(SCREENWIDTH * 0.2), movementInfo['playery']

    basex = movementInfo['basex']
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    # list of lowerpipe
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    pipeVelX = -4

    # player velocity, max velocity, downward accleration, accleration on flap
    playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
    playerMaxVelY =  10   # max vel along Y, max descend speed
    playerMinVelY =  -8   # min vel along Y, max ascend speed
    playerAccY    =   1   # players downward accleration
    playerRot     =  45   # player's rotation
    playerVelRot  =   3   # angular speed
    playerRotThr  =  20   # rotation threshold
    playerFlapAcc =  -9   # players speed on flapping
    playerFlapped = False # True when player flaps

    batchSize = 32

    # TODO
    # Replace with appropriate state
    state = [True, lowerPipes[0]['x'] - playerx, lowerPipes[0]['y'] - playery]
    
    fps = 30
    deaths = 0

    while True:

        for event in pygame.event.get():
            if event.type == KEYDOWN and event.key == K_SPACE:
                fps = 100000 if fps == 30 else 30
            if event.type == KEYDOWN and event.key == K_UP:
                fps = fps + 30
            if event.type == KEYDOWN and event.key == K_DOWN:
                if fps > 30:
                    fps = fps - 30
            if event.type == KEYDOWN and event.key == K_LEFT:
                if agent.epsilon > 0:
                    agent.epsilon = (agent.epsilon - 0.01)
            if event.type == KEYDOWN and event.key == K_RIGHT:
                if agent.epsilon < 1:
                    agent.epsilon = (agent.epsilon + 0.01)
            if event.type == pygame.QUIT:
                # save weights
                agent.save_model()
                return

        if lowerPipes[0]['x'] - playerx > -IMAGES['player'][0].get_width():
            currentPipe = lowerPipes[0]
        else:
            currentPipe = lowerPipes[1]

        action = agent.policy(state)

        if action == FLAP:
            if playery > -2 * IMAGES['player'][0].get_height():
                playerVelY = playerFlapAcc
                playerFlapped = True
                SOUNDS['wing'].play()

        # check for crash here
        crashTest = checkCrash({'x': playerx, 'y': playery, 'index': playerIndex},
                               upperPipes, lowerPipes)
        if crashTest[0]:
            movementInfo = initGame()
            score = playerIndex = loopIter = 0
            playerIndexGen = movementInfo['playerIndexGen']
            playerx, playery = int(SCREENWIDTH * 0.2), movementInfo['playery']

            basex = movementInfo['basex']
            upperPipes.clear()
            lowerPipes.clear()
            newPipe1 = getRandomPipe()
            newPipe2 = getRandomPipe()

            # list of upper pipes
            upperPipes = [
                {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
                {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
            ]

            # list of lowerpipe
            lowerPipes = [
                {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
                {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
            ]
            deaths += 1
            agent.replay(batchSize)
            if deaths % 500 == 0:
                agent.save_model()



        # check for score
        playerMidPos = playerx + IMAGES['player'][0].get_width() / 2
        for pipe in upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                score += 1
                SOUNDS['point'].play()

        # playerIndex basex change
        if (loopIter + 1) % 3 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        # rotate the player
        if playerRot > -90:
            playerRot -= playerVelRot

        # player's movement
        if playerVelY < playerMaxVelY and not playerFlapped:
            playerVelY += playerAccY
        if playerFlapped:
            playerFlapped = False

            # more rotation to cover the threshold (calculated in visible rotation)
            playerRot = 45

        playerHeight = IMAGES['player'][playerIndex].get_height()
        playery += min(playerVelY, BASEY - playery - playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        # print score so player overlaps the score
        showScore(score)
        highscore = showHighScore(score, highscore)
        showFPS(fps)
        showDeaths(deaths)
        showepsilon(agent.epsilon)

        # Player rotation has a threshold
        visibleRot = playerRotThr
        if playerRot <= playerRotThr:
            visibleRot = playerRot
        
        playerSurface = pygame.transform.rotate(IMAGES['player'][playerIndex], visibleRot)
        SCREEN.blit(playerSurface, (playerx, playery))

        next_state = [crashTest[0], currentPipe['x'] - playerx, currentPipe['y'] - playery]
        reward = getReward(crashTest[0])

        agent.remember(state, action, reward, next_state, crashTest[0])
        state = next_state

        pygame.display.update()
        FPSCLOCK.tick(fps)

def getReward(crash):
    """ reward function"""
    return -1000 if crash else 1


def playerShm(playerShm):
    """oscillates the value of playerShm['val'] between 8 and -8"""
    if abs(playerShm['val']) == 8:
        playerShm['dir'] *= -1

    if playerShm['dir'] == 1:
         playerShm['val'] += 1
    else:
        playerShm['val'] -= 1


def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE}, # lower pipe
    ]

def showDeaths(deaths):
    """displays fps in center of screen"""
    scoreDigits = [int(x) for x in list(str(deaths))]
    totalWidth = 0 # total width of all numbers to be printed


    for digit in scoreDigits:
        IMAGE_SMALL = pygame.transform.scale(IMAGES['numbers'][digit], (17, 20))
        totalWidth += IMAGE_SMALL.get_width()

    Xoffset = SCREENWIDTH - totalWidth

    SCREEN.blit(pygame.transform.scale(IMAGES['dead'], (25, 18)), (SCREENWIDTH - totalWidth - IMAGES['dead'].get_width(), 0))

    for digit in scoreDigits:
        IMAGE_SMALL = pygame.transform.scale(IMAGES['numbers'][digit], (17, 20))
        SCREEN.blit(IMAGE_SMALL, (Xoffset, 0))
        Xoffset += IMAGE_SMALL.get_width()

def showFPS(fps):
    """displays fps in center of screen"""
    scoreDigits = [int(x) for x in list(str(fps))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        IMAGE_SMALL = pygame.transform.scale(IMAGES['numbers'][digit], (17, 20))
        totalWidth += IMAGE_SMALL.get_width()

    Xoffset = 0

    for digit in scoreDigits:
        IMAGE_SMALL = pygame.transform.scale(IMAGES['numbers'][digit], (17, 20))
        SCREEN.blit(IMAGE_SMALL, (Xoffset, 0))
        Xoffset += IMAGE_SMALL.get_width()

def showepsilon(epsilon):
    """displays fps in center of screen"""
    eps = int(epsilon * 100)
    scoreDigits = [int(x) for x in list(str(eps))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        IMAGE_SMALL = pygame.transform.scale(IMAGES['numbers'][digit], (17, 20))
        totalWidth += IMAGE_SMALL.get_width()

    Xoffset = 0

    for digit in scoreDigits:
        IMAGE_SMALL = pygame.transform.scale(IMAGES['numbers'][digit], (17, 20))
        SCREEN.blit(IMAGE_SMALL, (Xoffset, SCREENHEIGHT * 0.88 ))
        Xoffset += IMAGE_SMALL.get_width()

def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()

def showHighScore(score, highscore):
    """displays score in center of screen"""
    if score > highscore:
        highscore = score


    scoreDigits = [int(x) for x in list(str(highscore))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.88))
        Xoffset += IMAGES['numbers'][digit].get_width()

    return highscore

def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground
    if (player['y'] + player['h'] >= BASEY - 1) or (player['y'] + player['h'] <= 0):
        return [True, True]
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])
        pipeW = IMAGES['pipe'][0].get_width()
        pipeH = IMAGES['pipe'][0].get_height()

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return [True, False]

    return [False, False]

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

if __name__ == '__main__':
    main()
