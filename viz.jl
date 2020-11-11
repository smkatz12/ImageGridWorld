using PGFPlots

function get_filled_rectangle(lb, ub, color)
    return "\\filldraw[fill=$(color), draw=black] (axis cs:$(string(lb[1])),$(string(lb[2]))) rectangle (axis cs:$(string(ub[1])),$(string(ub[2])));"
end

function plot_num_actions(mdp::SimpleGridWorld, policy)
    ax = Axis()
    ax.xmin = 0.0
    ax.xmax = mdp.size[1]
    ax.ymin = 0.0
    ax.ymax = mdp.size[2]
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

function plot_prob(mdp::SimpleGridWorld, prob)
    ax = Axis()
    ax.xmin = 0.0
    ax.xmax = mdp.size[1]
    ax.ymin = 0.0
    ax.ymax = mdp.size[2]
    ax.axisEqualImage = true
    ax.title = "Probability of Success"

    prob_matrix = zeros(mdp.size[1], mdp.size[2])

    for s in states(mdp)
        if s[1] > 0
            p = prob[stateindex(mdp, s)]
            prob_matrix[s[1], s[2]] = p
        end
    end

    push!(ax, Plots.MatrixPlot(prob_matrix, colormap = pasteljet))

    return ax
end

function plot_summary(mdp::SimpleGridWorld, policy, prob)
    g = GroupPlot(2, 2, groupStyle = "horizontal sep = 2cm")
    push!(g, plot_num_actions(mdp, policy))
    push!(g, plot_prob(mdp, prob))
    return g
end
    