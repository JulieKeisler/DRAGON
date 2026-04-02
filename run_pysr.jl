#!/usr/bin/env julia
# run_pysr.jl — Standalone Julia script for symbolic regression with SymbolicRegression.jl
# Run with: julia --project=/Users/elyaschikhaoui/Desktop/dragon/.dragonenv/julia_env --threads=1 run_pysr.jl

using SymbolicRegression
using CSV
using DataFrames
using Printf

# ── Configuration ─────────────────────────────────────────────────────
TARGET = "BSI"  # Change this to search for different formulas
DATA_PATH = "data/6000_points.csv"
MAX_ITERS = 1000
SELECT_K = 5          # auto-select top k features
NITERATIONS = 2000     # PySR default
POPULATIONS = 15
POPULATION_SIZE = 33

# ── Load data ─────────────────────────────────────────────────────────
println("Loading data from $DATA_PATH...")
df = CSV.read(DATA_PATH, DataFrame)

# Drop non-numeric columns
select!(df, Not([:var"system:index", :QA60, :var".geo", :date]))

# Compute BSI target: (B11 + B4 - B8 - B2) / (B11 + B4 + B8 + B2)
if TARGET == "BSI"
    df[!, :BSI] = (df.B11 .+ df.B4 .- df.B8 .- df.B2) ./ (df.B11 .+ df.B4 .+ df.B8 .+ df.B2)
elseif TARGET == "NDVI"
    df[!, :NDVI] = (df.B8 .- df.B4) ./ (df.B8 .+ df.B4)
elseif TARGET == "MNDWI"
    df[!, :MNDWI] = (df.B3 .- df.B11) ./ (df.B3 .+ df.B11)
elseif TARGET == "NDMI"
    df[!, :NDMI] = (df.B8 .- df.B11) ./ (df.B8 .+ df.B11)
elseif TARGET == "MSI"
    df[!, :MSI] = df.B11 ./ df.B8
end

# Remove rows with NaN/Inf in target
filter!(row -> isfinite(row[Symbol(TARGET)]), df)

# Separate features and target
y = Float64.(df[!, Symbol(TARGET)])
X = select(df, Not(Symbol(TARGET)))

# Keep only numeric columns
numeric_cols = [n for n in names(X) if eltype(X[!, n]) <: Number]
X = X[!, numeric_cols]
X_matrix = Matrix{Float64}(X)

println("Data: $(size(X_matrix, 1)) samples, $(size(X_matrix, 2)) features")
println("Features: $(names(X))")
println("Target: $TARGET")
println()

# ── Run SymbolicRegression ────────────────────────────────────────────
println("Starting symbolic regression...")

options = Options(
    binary_operators=[+, -, *, /],
    unary_operators=[exp],
    populations=POPULATIONS,
    population_size=POPULATION_SIZE,
    maxsize=20,
    parsimony=0.0032f0,
)

hall_of_fame = equation_search(
    X_matrix', y;  # Note: SymbolicRegression expects features as rows
    options=options,
    niterations=NITERATIONS,
    variable_names=names(X),
    parallelism=:multithreading,
)

# ── Results ───────────────────────────────────────────────────────────
println("\n" * "="^60)
println("  Results for $TARGET")
println("="^60)

dominating = calculate_pareto_frontier(hall_of_fame)

println("\nPareto frontier (complexity vs loss):")
println("-"^80)
for (i, member) in enumerate(dominating)
    complexity = compute_complexity(member, options)
    loss = member.loss
    @printf("  [%2d] complexity=%2d  loss=%.8e  %s\n", i, complexity, loss, member.tree)
end
println("-"^80)

# Save results to file
open("pysr_$(TARGET)_results.txt", "w") do f
    println(f, "Target: $TARGET")
    println(f, "Data: $(size(X_matrix, 1)) samples, $(size(X_matrix, 2)) features")
    println(f, "\nPareto frontier:")
    for (i, member) in enumerate(dominating)
        complexity = compute_complexity(member, options)
        loss = member.loss
        @printf(f, "[%2d] complexity=%2d  loss=%.8e  %s\n", i, complexity, loss, member.tree)
    end
end

println("\nResults saved to pysr_$(TARGET)_results.txt")
