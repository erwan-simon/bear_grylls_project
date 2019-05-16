# Bear Grylls Project

## TL;DR

Using deep reinforcement learning to create an agent able to not starve to death.

## Abstract

We present a world suited to test the capacity of an agent to harvest resources
at a minimum speed. It has to navigate around obstacles which interfer with its
progression, all of that with limited vision of its environment. An agent enable
to maintain a minimal harvesting speed dies.
We also provide a description of our tries to implement an algorithm of deep
reinforcement learning which can answer to this task in the most efficient way
possible.

## Genesis

The global idea behind this project comes from a school project at Epitech called _Zappy_. The idea of this project was to build a sort of game (sort of because the only players in the game were artificial intelligence (AI)) in which agents should survive by finding food and try to level up using different resources available in a 2D discrete world. This required implementing a social behaviour to the AI because leveling up could be done only with several of them.

By the time of this project I made a basic artificial intelligence in C made with old school if ; else if ; else ; code. Not really a piece of art but it got the job done and were much efficient actually.

Now that I am more aware of the deep learning technics, I want to try to make an intelligent agent learning by itself how to survive without starving to death (whence the name of the project).

## Installation and usage

### Prerequisites

* [Python 3](www.python.org)
* [PyGame](www.pygame.org/)
* [Anaconda](www.anaconda.com/)

### Launch game

```python3 main.py```

### Change model

Every model is located in the [network directory](network), you can change the used model by implementing it in the [main file](main.py).

More information in the [scientific paper](paper/bear_grylls_project.pdf) (rules of the game, details on the models, etc.).
