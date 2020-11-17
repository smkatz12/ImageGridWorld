"""
Modifying the usual discrete value iteration solver to work with the relaxed mdp
"""

# The solver type
"""
    RelaxedValueIterationSolver <: Solver
The solver type. Contains the following parameters that can be passed as keyword arguments to the constructor
    - max_iterations::Int64, the maximum number of iterations value iteration runs for (default 100)
    - belres::Float64, the Bellman residual (default 1e-3)
    - verbose::Bool, if set to true, the bellman residual and the time per iteration will be printed to STDOUT (default false)
    - include_Q::Bool, if set to true, the solver outputs the Q values in addition to the utility and the policy (default true)
    - init_util::Vector{Float64}, provides a custom initialization of the utility vector. (initializes utility to 0 by default)
"""
mutable struct RelaxedValueIterationSolver <: Solver
    max_iterations::Int64 # max number of iterations
    belres::Float64 # the Bellman Residual
    verbose::Bool 
    include_Q::Bool
    init_util::Vector{Float64}
end

# Default constructor
function RelaxedValueIterationSolver(;max_iterations::Int64 = 100, 
                               belres::Float64 = 1e-3,
                               verbose::Bool = false,
                               include_Q::Bool = true,
                               init_util::Vector{Float64}=Vector{Float64}(undef, 0))    
    return RelaxedValueIterationSolver(max_iterations, belres, verbose, include_Q, init_util)
end

#####################################################################
# Solve runs the relaxed value iteration algorithm.
# The policy input argument is either provided by the user or
# initialized during the function call.
# Verbose is a flag that triggers text output to the command line
# Example code for running the function:
# mdp = GridWorld(10, 10) # initialize a 10x10 grid world MDP (user written code)
# solver = ValueIterationSolver(max_iterations=40, belres=1e-3)
# policy = ValueIterationPolicy(mdp)
# solve(solver, mdp, policy, verbose=true)
#####################################################################
function solve(solver::RelaxedValueIterationSolver, mdp::MDP; kwargs...)

    # solver parameters
    max_iterations = solver.max_iterations
    belres = solver.belres
    discount_factor = discount(mdp)
    println("discount: ", discount_factor)
    ns = length(states(mdp))
    na = length(actions(mdp))

    # intialize the utility and Q-matrix
    if !isempty(solver.init_util)
        @assert length(solver.init_util) == ns "Input utility dimension mismatch"
        util = solver.init_util
    else
        util = ones(ns)
    end
    include_Q = solver.include_Q
    if include_Q
        qmat = zeros(ns, na)
    end
    pol = zeros(Int64, ns)

    total_time = 0.0
    iter_time = 0.0

    # create an ordered list of states for fast iteration
    state_space = ordered_states(mdp)

    # main loop
    for i = 1:max_iterations
        residual = 0.0
        resdual_state = -1
        iter_time = @elapsed begin
        # state loop
        for (istate,s) in enumerate(state_space)
            sub_aspace = reverse(actions(mdp, s))
            if isterminal(mdp, s)
                util[istate] = 0.0
                pol[istate] = 1
            else
                old_util = util[istate] # for residual
                min_util = Inf
                actual_util = Inf
                # action loop
                # util(s) = max_a( R(s,a) + discount_factor * sum(T(s'|s,a)util(s') )
                for a in sub_aspace
                    
                    iaction = actionindex(mdp, a)
                    future_util = transition(mdp, s, a, util) # creates distribution over neighbors
                    new_util = reward(mdp, s, a) + discount_factor * future_util
                    δ = mdp.subset_size_reward * (length(a) - 1)
                    
                    if new_util - δ < min_util
                        min_util = new_util - δ
                        actual_util = new_util
                        pol[istate] = iaction
                    end
                    include_Q ? (qmat[istate, iaction] = new_util) : nothing
                end # action
                # update the value array
                util[istate] = actual_util
                diff = abs(util[istate] - old_util)
                if diff > residual
                    residual = diff
                    resdual_state = s
                end
            end
        end # state
        end # time
        total_time += iter_time
        solver.verbose ? @printf("[Iteration %-4d] residual: %10.3G | iteration runtime: %10.3f ms, (%10.3G s total)\n", i, residual, iter_time*1000.0, total_time) : nothing
        residual < belres ? break : nothing
    end # main
    if include_Q
        return ValueIterationPolicy(mdp, qmat, util, pol)
    else
        return ValueIterationPolicy(mdp, utility=util, policy=pol, include_Q=false)
    end
end

