# Files

* main.py
	* This file has code specific to the BZ solver.
	* This file is also were the ode solver defined in ode is called.
* ode.py
	* This file contains the implementation of matlab's ode23 under the name stiff_solver


# Explanation

The stiff ode solver that I implemented was ode23. Ode23 is matlab's solver for stiff differential equation. It is described
in the following paper http://bicycle.tudelft.nl/schwab/TAM674/SR97.pdf and implemented albeit in Julia here https://github.com/JuliaDiffEq/ODE.jl/blob/master/src/ODE.jl.
The algorithim, from what I understand, is based on the Rosenbrock multistep method for solving ODEs. I chose this algorithim because I believed that if Matlab and thus
many scientists were using this algorithim in production that it had to be one of the better algorithims out there. My implementation is incomplete in that I did not 
implement the estimator used in the code I referenced for choosing an first step, described here "Solving Ordinary Differential Equations I" by Hairer et al., p.169.
Additionally, I could not get the algorithim working for BZ reaction model. Nonetheless, I tried to implement the basics of the algorithim. I wrote comments to explain
most of what I believe is happening.