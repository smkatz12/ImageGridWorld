function worst_case_action(mdp::RelaxedGridWorld, s, a, util)
    if s in mdp.gw.terminate_from || isterminal(mdp, s)
        return actions(mdp)[1]
    end

    future_util = Inf
    worst_action = actions(mdp)[1]
    # Test each grid world action in the subset and find worst case future utility
    for gwa in a
        dist = transition(mdp.gw, s, gwa)
        action_future_util = 0.0
        for (sp, p) in weighted_iterator(dist)
            p == 0.0 ? continue : nothing
            isp = stateindex(mdp.gw, sp)
            action_future_util += p * util[isp]
        end
        if action_future_util < future_util 
            future_util = action_future_util
            worst_action = gwa
        end
    end

    return worst_action
end

function failure_transition(mdp::RelaxedGridWorld, s, a, util)
    if s in mdp.gw.terminate_from || isterminal(mdp, s)
        return actions(mdp)[1]
    end

    future_util = -Inf
    # Test each grid world action in the subset and find worst case future utility
    for gwa in a
        dist = transition(mdp.gw, s, gwa)
        action_future_util = 0.0
        for (sp, p) in weighted_iterator(dist)
            p == 0.0 ? continue : nothing
            isp = stateindex(mdp.gw, sp)
            action_future_util += p * util[isp]
        end
        if action_future_util > future_util 
            future_util = action_future_util
        end
    end

    return future_util
end