function check_relaxed(mdp::RelaxedGridWorld, policy::ValueIterationPolicy; max_iterations = 100, belres = 1e-4, verbose = false)
    
    # solver parameters
    ns = length(states(mdp))
    na = length(actions(mdp))

    # intialize the probilities
    prob = zeros(ns)

    total_time = 0.0
    iter_time = 0.0

    # create an ordered list of states for fast iteration
    state_space = ordered_states(mdp)

    # determine goal state
    goal_state = GWPos(mdp.gw.size[1], mdp.gw.size[2])

    # main loop
    for i = 1:max_iterations
        residual = 0.0
        iter_time = @elapsed begin
        # state loop
        for (istate,s) in enumerate(state_space)
            old_prob = prob[istate] # for residual
            if s == goal_state
                new_prob = 1.0
            elseif s in mdp.gw.terminate_from || isterminal(mdp, s)
                new_prob = 0.0
            else
                a = action(policy, s)
                new_prob = transition(mdp, s, a, prob) # creates distribution over neighbors
            end
            prob[istate] = new_prob
            diff = abs(new_prob - old_prob)
            diff > residual ? (residual = diff) : nothing
        end # state
        end # time
        total_time += iter_time
        verbose ? @printf("[Iteration %-4d] residual: %10.3G | iteration runtime: %10.3f ms, (%10.3G s total)\n", i, residual, iter_time*1000.0, total_time) : nothing
        #residual < belres ? break : nothing
    end # main
    return prob 
end

function check_relaxed_failure(mdp::RelaxedGridWorld, policy::ValueIterationPolicy; max_iterations = 100, belres = 1e-4, verbose = false)
    
    # solver parameters
    ns = length(states(mdp))
    na = length(actions(mdp))

    # intialize the probilities
    prob = zeros(ns)

    total_time = 0.0
    iter_time = 0.0

    # create an ordered list of states for fast iteration
    state_space = ordered_states(mdp)

    # determine goal state
    goal_state = GWPos(mdp.gw.size[1], mdp.gw.size[2])

    # main loop
    for i = 1:max_iterations
        residual = 0.0
        iter_time = @elapsed begin
        # state loop
        for (istate,s) in enumerate(state_space)
            old_prob = prob[istate] # for residual
            if s == goal_state
                new_prob = 0.0
            elseif s in mdp.gw.terminate_from || isterminal(mdp, s)
                new_prob = 1.0
            else
                a = action(policy, s)
                new_prob = failure_transition(mdp, s, a, prob) # creates distribution over neighbors
            end
            prob[istate] = new_prob
            diff = abs(new_prob - old_prob)
            diff > residual ? (residual = diff) : nothing
        end # state
        end # time
        total_time += iter_time
        verbose ? @printf("[Iteration %-4d] residual: %10.3G | iteration runtime: %10.3f ms, (%10.3G s total)\n", i, residual, iter_time*1000.0, total_time) : nothing
        residual < belres ? break : nothing
    end # main
    return prob 
end