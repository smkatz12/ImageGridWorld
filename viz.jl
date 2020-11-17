using PGFPlots

function plot_num_actions(mdp::RelaxedGridWorld, policy)
    ax = Axis()
    ax.xmin = 0.0
    ax.xmax = mdp.gw.size[1]
    ax.ymin = 0.0
    ax.ymax = mdp.gw.size[2]
    ax.axisEqualImage = true
    ax.title = "Number of Actions"

    for s in states(mdp)
        as = action(policy, s)
        if length(as) == 1
            color = "blue!80"
        elseif length(as) == 2
            color = "red"
        elseif length(as) == 3
            color = "yellow"
        else
            color = "black"
        end

        push!(ax, Plots.Command(get_filled_rectangle([s[1] - 1, s[2] - 1],
                                                     [s[1], s[2]], color)))

    end
    return ax
end

function plot_actions(mdp::RelaxedGridWorld, policy, prob; prob_worst = true)
    ax = Axis()
    ax.xmin = 0.0
    ax.xmax = mdp.gw.size[1]
    ax.ymin = 0.0
    ax.ymax = mdp.gw.size[2]
    ax.axisEqualImage = true
    ax.title = "Actions"

    for s in states(mdp)
        push!(ax, Plots.Command(get_filled_rectangle([s[1] - 1, s[2] - 1],
                                                     [s[1], s[2]], "black")))
        
        as = action(policy, s)
        worst_case_a = prob_worst ? worst_case_action(mdp, s, as, prob) : worst_case_action(mdp, s, as, policy.util)
        for a in as
            if a == worst_case_a
                push!(ax, Plots.Command(get_filled_triangle(a, [s[1] - 1, s[2] - 1],
                                                            [s[1], s[2]], "green")))
            else
               push!(ax, Plots.Command(get_filled_triangle(a, [s[1] - 1, s[2] - 1],
                                                            [s[1], s[2]], "white")))
            end
        end
    end
    return ax
end

function plot_prob(mdp::RelaxedGridWorld, prob)
    ax = Axis()
    ax.xmin = 0.0
    ax.xmax = mdp.gw.size[1]
    ax.ymin = 0.0
    ax.ymax = mdp.gw.size[2]
    ax.axisEqualImage = true
    ax.title = "Probability of Success"

    prob_matrix = zeros(mdp.gw.size[1], mdp.gw.size[2])

    for s in states(mdp)
        if s[1] > 0
            p = prob[stateindex(mdp, s)]
            prob_matrix[s[1], s[2]] = p
        end
    end

    push!(ax, Plots.MatrixPlot(prob_matrix, colormap = pasteljet))

    return ax
end

function plot_summary(mdp::RelaxedGridWorld, policy, prob)
    g = GroupPlot(3, 1, groupStyle = "horizontal sep = 1.5cm")
    push!(g, plot_num_actions(mdp, policy))
    push!(g, plot_actions(mdp, policy, prob))
    push!(g, plot_prob(mdp, prob))
    return g
end

function get_filled_rectangle(lb, ub, color)
    return "\\filldraw[fill=$(color), draw=black] (axis cs:$(string(lb[1])),$(string(lb[2]))) rectangle (axis cs:$(string(ub[1])),$(string(ub[2])));"
end

function get_filled_triangle(a, lb, ub, color)
    if a == :up
        return get_up_triangle(lb, ub, color)
    elseif a == :down
        return get_down_triangle(lb, ub, color)
    elseif a == :left
        return get_left_triangle(lb, ub, color)
    else
        return get_right_triangle(lb, ub, color)
    end
end

function get_up_triangle(lb, ub, color)
    cell_height = ub[2] - lb[2]
    cell_width = ub[1] - lb[1]

    bottom_y = ub[2] - cell_height / 4
    left_x = lb[1] + cell_width / 4
    right_x = ub[1] - cell_width / 4

    top_y = ub[2]
    top_x = lb[1] + cell_width / 2

    return "\\filldraw[fill=$(color), draw=black] 
                (axis cs:$(string(left_x)),$(string(bottom_y))) -- 
                (axis cs:$(string(right_x)),$(string(bottom_y))) --
                (axis cs:$(string(top_x)),$(string(top_y))) --
                (axis cs:$(string(left_x)),$(string(bottom_y)));"
end

function get_down_triangle(lb, ub, color)
    cell_height = ub[2] - lb[2]
    cell_width = ub[1] - lb[1]

    top_y = lb[2] + cell_height / 4
    left_x = lb[1] + cell_width / 4
    right_x = ub[1] - cell_width / 4

    bottom_y = lb[2]
    bottom_x = lb[1] + cell_width / 2

    return "\\filldraw[fill=$(color), draw=black] 
                (axis cs:$(string(left_x)),$(string(top_y))) -- 
                (axis cs:$(string(right_x)),$(string(top_y))) --
                (axis cs:$(string(bottom_x)),$(string(bottom_y))) --
                (axis cs:$(string(left_x)),$(string(top_y)));"
end

function get_left_triangle(lb, ub, color)
    cell_height = ub[2] - lb[2]
    cell_width = ub[1] - lb[1]

    right_x = lb[1] + cell_width / 4
    top_y = ub[2] - cell_height / 4
    bottom_y = lb[2] + cell_height / 4

    left_x = lb[1]
    left_y = lb[2] + cell_height / 2

    return "\\filldraw[fill=$(color), draw=black] 
                (axis cs:$(string(left_x)),$(string(left_y))) -- 
                (axis cs:$(string(right_x)),$(string(top_y))) --
                (axis cs:$(string(right_x)),$(string(bottom_y))) --
                (axis cs:$(string(left_x)),$(string(left_y)));"
end

function get_right_triangle(lb, ub, color)
    cell_height = ub[2] - lb[2]
    cell_width = ub[1] - lb[1]

    left_x = ub[1] - cell_width / 4
    top_y = ub[2] - cell_height / 4
    bottom_y = lb[2] + cell_height / 4

    right_x = ub[1]
    right_y = lb[2] + cell_height / 2

    return "\\filldraw[fill=$(color), draw=black] 
                (axis cs:$(string(right_x)),$(string(right_y))) -- 
                (axis cs:$(string(left_x)),$(string(top_y))) --
                (axis cs:$(string(left_x)),$(string(bottom_y))) --
                (axis cs:$(string(right_x)),$(string(right_y)));"
end