struct RelaxedGridWorld <: MDP{GWPos, Vector{Symbol}}
    gw::MDP # corresponding normal grid world
    action_subsets::Vector # action subsets to consider
    subset_inds::Dict # maps subsets to indices for efficient indexing
    subset_size_reward::Float64 # reward for additional actions
    discount::Float64 # discount factor to prevent cycles
end

function RelaxedGridWorld(;gw = image_grid_world(), action_subset_size = 4, subset_size_reward = 0.001, discount = 1.0)
    gw_actions = actions(gw)
    action_subsets = [combo for i = 1:action_subset_size for combo in combinations(gw_actions, i)]
    subset_inds = Dict()
    for i = 1:length(action_subsets)
        subset_inds[action_subsets[i]] = i
    end
    return RelaxedGridWorld(gw, action_subsets, subset_inds, subset_size_reward, discount)
end

# States

POMDPs.states(mdp::RelaxedGridWorld) = states(mdp.gw)

POMDPs.stateindex(mdp::RelaxedGridWorld, s::AbstractVector{Int}) = stateindex(mdp.gw, s)

POMDPs.initialstate(mdp::RelaxedGridWorld) = initialstate(mdp.gw)

# Actions

POMDPs.actions(mdp::RelaxedGridWorld) = mdp.action_subsets

POMDPs.actionindex(mdp::RelaxedGridWorld, a::Vector{Symbol}) = mdp.subset_inds[a]

# Transitions

POMDPs.isterminal(mdp::RelaxedGridWorld, s::AbstractVector{Int}) = any(s .< 0)

function POMDPs.transition(mdp::RelaxedGridWorld, s::AbstractVector{Int}, a::Vector{Symbol}, util::Vector{Float64})
    """
    This function returns future utility in order to avoid repeat calculations during the
    solving process.
    """
    if s in mdp.gw.terminate_from || isterminal(mdp, s)
        return 0.0
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
        action_future_util > future_util ? future_util = action_future_util : nothing
    end

    return future_util
end

# Rewards

function POMDPs.reward(mdp::RelaxedGridWorld, s::AbstractVector{Int}, a::Vector{Symbol})
    r = 0.0

    # Reward for goal state
    # if tuple(s...) == mdp.gw.size
        # r += 1.0
    # end
    if get(mdp.gw.rewards, s, 0) < 0
        r += 1.0
    end

    # Reward for having multiple possible actions
    # Nonzero reward for any action subset larger than 2
    # r += mdp.subset_size_reward * (length(a) - 1)

    return r
end

POMDPs.discount(mdp::RelaxedGridWorld) = mdp.discount