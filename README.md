# Bear Grylls Project

## Abstract (TL;DR)

Using deep reinforcement learning to create an agent able to not starve to death.

## Genesis

The global idea behind this project comes from a school project at Epitech called _Zappy_. The idea of this project was to build a sort of game (sort of because the only players in the game were artificial intelligence (AI)) in which agents should survive by finding food and try to level up using different resources available in a 2D discrete world. This required implementing a social behaviour to the AI because leveling up could be done only with several of them.

By the time of this project I made a basic artificial intelligence in C made with old school if ; else if ; else ; code. Not really a piece of art but it got the job done and were much efficient actually.

Now that I am more aware of the deep learning technics, I want to try to make an intelligent agent learning by itself how to survive without starving to death (whence the name of the project).

## The game

### The environment ([file](game/Game.py))

#### The board

The board compose a two-dimensionnal discrete toric world composed of squares (technical jargon to say we have a grid where if an agent exists by the right side of the board it will come back through the left side).

#### Food

Food appears at random location of the board.

#### Agents 

Agents can spawn and progress in it. There can be one or many.

### The agents ([file](game/Player.py))

#### Movement

An agent moves one tile at a time up, down, left or right.

#### Perception of the environment

An agent only has a limited vision of 2 tiles around it.

#### The inventory

An agent knows how many food it has left. Food amount in the agent inventory decreases each turn.

## The Deep Reinforcement Learning part

### Reinforcement Learning ([file](network/NetworkWrapper.py))

#### The state ([function](https://github.com/erwan-simon/bear_grylls_project/blob/3aa957d1d095d81f8fb10284d347027499e242e5/network/NetworkWrapper.py#L19))

The state is an array containing the vision area around the agent and the food it has left.

#### The reward ([function](https://github.com/erwan-simon/bear_grylls_project/blob/3aa957d1d095d81f8fb10284d347027499e242e5/network/NetworkWrapper.py#L27))

The reward give a lot of points if the agent ate food at the present turn and add some points if the agent has food in its sight (the closer the agent is to the food, the more it adds points to the reward).

### Deep Learning networks

#### Tensorflow and Keras ([file](network/base/KerasAgent.py))

The Keras Agent is not up-to-date anymore. Maybe I will fix it in a near future.

#### Pytorch ([file](network/base/BasePytorch.py))

The Pytorch base agent works well.
