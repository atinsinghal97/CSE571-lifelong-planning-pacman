# CSE571-lifelong-planning-pacman
Incremental Path Planning (LPA D*-Lite) in Pacman Domain

Project topic: Life-long planning
a) Implement lifelong A* search described in the following paper: “D* Lite”, Sven Koenig and Maxim Likhachev, AAAI 2002, and integrate into the Pacman domain for path-finding problems (from start to a fixed goal location) in your individual project 1 (http://www.aaai.org/Papers/AAAI/2002/AAAI02-072.pdf). 1) Assume that Pacman only knows about the size of the (grid-world) environment initially, and can only observe local environment surrounding itself (you may define what local environment is). 2) Also, assume that Pacman always knows where it is in the environment (i.e., it can localize). 3) Once the Pacman observes something, it is able to keep it in its mind (i.e., it maintains the knowledge that there is an obstacle in a given location once that is observed).
b) Compare the algorithm’s performance with an A* baseline that simply replans every time when new obstacles are observed in environments of different sizes and complexities. Provide statistical analyses for your results.
c) Submit a written report with your findings, which should include: 0. Abstract; 1. Introduction (motivation and your achievements in this project); 2) Technical approach (a brief technical discussion of the life-long planning method you implemented); 3) Results (Results that CLEARLY illustrate the strengths of your approach compared to others in a statistically meaningful way, for example, you could compare the number of nodes expanded, computational time, etc.). 4) Conclusions (any observations and discussions); 5) Team effectiveness (notes about team member contributions).
d) Submit your code with comments, and with instructions (as a README file) to run it.
