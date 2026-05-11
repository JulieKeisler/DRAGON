#!/usr/bin/env julia
using SymbolicRegression
using Printf

TARGET = "bai"
NITERATIONS = 1000
POPULATIONS = 5
POPULATION_SIZE = 30


using CSV
using DataFrames

df = CSV.read("data/6000_points.csv", DataFrame)
for col in [:var"system:index", :QA60, :var".geo", :date]
    if col in propertynames(df)
        select!(df, Not(col))
    end
end

y_raw = 1 ./ ((0.1 .- df.B4).^2 .+ (0.06 .- df.B8).^2)
valid = isfinite.(y_raw)
y = Float64.(y_raw[valid])
X_df = df[valid, :]
numeric_cols = [n for n in names(X_df) if eltype(X_df[!, n]) <: Number]
X_df = X_df[!, numeric_cols]
X_df = "bai" in names(X_df) ? select(X_df, Not(Symbol("bai"))) : X_df
# SymbolicRegression.jl expects X with shape [features, rows] -> transpose.
X_matrix = permutedims(Matrix{Float64}(X_df))
feature_names = names(X_df)


println("Data: $(size(X_matrix, 2)) samples, $(size(X_matrix, 1)) features")

options = Options(
    binary_operators=[+, -, *, /],
    unary_operators=[exp, sqrt, abs, sin, cos, log],
    populations=POPULATIONS,
    population_size=POPULATION_SIZE,
    maxsize=20,
    parsimony=0.0032f0,
)

hall_of_fame = equation_search(
    X_matrix, y;
    options=options,
    niterations=NITERATIONS,
    variable_names=feature_names,
    parallelism=:multithreading,
)

println("\n" * "="^60)
println("  Results for $TARGET")
println("="^60)

dominating = calculate_pareto_frontier(hall_of_fame)

open("leaderboard_runs/bai/pysr/run_0/pysr_bai_results.txt", "w") do f
    println(f, "Target: $TARGET")
    println(f, "\nPareto frontier:")
    for (i, member) in enumerate(dominating)
        complexity = compute_complexity(member, options)
        loss = member.loss
        @printf(f, "[%2d] complexity=%2d  loss=%.8e  %s\n", i, complexity, loss, string_tree(member.tree, options))
    end
end

println("Results saved to leaderboard_runs/bai/pysr/run_0/pysr_bai_results.txt")
