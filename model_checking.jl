struct ModelCheckingPolicy
    mdp::MDP # simple grid world (not relaxed)
    pol::Dict # state -> action
end

function model_check(mdp::RelaxedGridWorld, relaxed_policy::ValueIterationPolicy; max_iterations = 100, verbose = false)
    # Define initial policy (just select first action in each subset)
    pol = Dict()
    for s in states(mdp)
        pol[s] = action(relaxed_policy, s)[1]
    end
    p = ModelCheckingPolicy(mdp.gw, pol)

    prob = zeros(length(states(mdp)))

    for i = 1:max_iterations
        verbose && mod(i, 10) == 0 ? println("Iteration: $i") : nothing
        
        # Policy evaluation
        prob = policy_evaluation(p)

        # Policy update
        p′ = get_model_checking_policy(mdp, relaxed_policy, prob)

        # Check for convergence
        p.pol == p′.pol ? break : nothing

        # Update p
        p = p′
    end

    return prob
end

function policy_evaluation(p::ModelCheckingPolicy)
    mdp, pol = p.mdp, p.pol

    # determine goal state
    goal_state = GWPos(mdp.size[1], mdp.size[2])

    # separate terminal and nonterminal states and create indexing for them
    terminal_states = Dict()
    nonterminal_states = Dict()
    tindex = 0
    nindex = 0
    for s in states(mdp)
        if s[1] > -1
            if s in mdp.terminate_from
                tindex += 1
                terminal_states[s] = tindex
            else
                nindex += 1
                nonterminal_states[s] = nindex
            end
        end
    end

    # Build matrices
    T_N = zeros(nindex, nindex)
    T_T = zeros(nindex, tindex)

    for (s, s_ind) in nonterminal_states
        dist = transition(mdp, s, pol[s])
        for (sp, p) in weighted_iterator(dist)
            if sp[1] > -1
                if sp in mdp.terminate_from
                    sp_ind = terminal_states[sp]
                    T_T[s_ind, sp_ind] = p
                else
                    sp_ind = nonterminal_states[sp]
                    T_N[s_ind, sp_ind] = p
                end
            end
        end
    end
    
    U_T = zeros(tindex)
    U_T[terminal_states[goal_state]] = 1.0

    # Solve: U_N = (I - T_N)⁻¹ T_T U_T
    U_N = (I - T_N) \ (T_T * U_T)

    # Recreate full utility
    U = zeros(length(states(mdp)))
    for s in states(mdp)
        if s[1] < 0
            U[stateindex(mdp, s)] = 0.0
        elseif s in mdp.terminate_from
            U[stateindex(mdp, s)] = U_T[terminal_states[s]]
        else
            U[stateindex(mdp, s)] = U_N[nonterminal_states[s]]
        end
    end
    return U
end

function get_model_checking_policy(mdp::RelaxedGridWorld, relaxed_policy, U)
    pol = Dict()
    for s in states(mdp.gw)
        if isterminal(mdp.gw, s) || s in mdp.gw.terminate_from
            pol[s] = actions(mdp)[1]
        else
            as = action(relaxed_policy, s)
            if length(as) == 1
                pol[s] = as[1]
            else
                worst_a = as[1]
                worst_val = Inf
                for a in as
                    val = lookahead(mdp.gw, U, s, a)
                    if val < worst_val
                        worst_a = a
                        worst_val = val
                    end
                end
                pol[s] = worst_a
            end
        end
    end
    return ModelCheckingPolicy(mdp.gw, pol)
end

function lookahead(mdp::SimpleGridWorld, U, s, a)
    dist = transition(mdp, s, a)
    u = 0
    for (sp, p) in weighted_iterator(dist)
        u += p * U[stateindex(mdp, sp)]
    end
    return round(u, digits = 8)
end