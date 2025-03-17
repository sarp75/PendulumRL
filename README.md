# Pendulum Cart Balancing with Reinforcement Learning

![Cart-Pole Balance](https://storage.googleapis.com/jschneier.com/static/img/cartpole-small.gif)

## Overview

This project implements a reinforcement learning solution to the classic cart-pole control problem using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm. The goal is to teach a neural network to balance an inverted pendulum by moving a cart horizontally along a track.

## Features

- **Physics-Based Simulation**: Accurate modeling of cart-pole dynamics
- **NEAT Implementation**: Neural networks that evolve over time
- **Real-time Visualization**: See both the pendulum system and neural network in action
- **Interactive Training**: Train, run, and reset the simulation with a single click

## Technical Details

### Core Components

1. **Pendulum Cart Simulation**
   - Implements rigid body physics for the cart and pendulum
   - Models friction, gravity, and momentum
   - Constrained to a fixed-length track

2. **NEAT Neural Networks**
   - Evolutionary algorithm with species-based competition
   - Dynamic topology that grows in complexity over time
   - Crossover and mutation of neural networks

3. **Reinforcement Learning**
   - Rewards based on how well the pendulum is balanced upright
   - Population-based training
   - Automatic speciation to maintain diverse solutions

4. **Visualization**
   - Real-time rendering of the cart-pole system
   - Neural network visualization showing weights and activations
   - Training metrics display

## How to Use

dotnet build

Go figure other things yourself
