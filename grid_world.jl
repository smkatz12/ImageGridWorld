function image_grid_world(;igw_size = 10, 
                           pit_min = 5, 
                           pit_max = 6, 
                           goal_reward = 1.0, 
                           pit_penalty = -1.0, 
                           tprob = 0.8, 
                           discount = 0.9999)
    rewards = Dict(GWPos(igw_size, igw_size) => goal_reward)
    for i = pit_min:pit_max
        for j = pit_min:pit_max
            rewards[GWPos(i, j)] = pit_penalty
        end
    end
    return SimpleGridWorld(size = (igw_size, igw_size), rewards = rewards, tprob = tprob, discount = discount)
end