# LTEE V3-A Forward-time Simulation Benchmark

CPU-only Python 3.9+ / NumPy. No torch, no Evo2, no external data.

## Task
Forward Wright-Fisher reproduction of *haplotypes* with mutation, optional selection,
context-dependent mutation susceptibility, population-rate heterogeneity, and
an unobserved mid-run mutator shift. No retrospective phylogenetic reconstruction.

## Scenarios
- neutral: constant shared mutation process; whole haplotypes inherited (linkage)
- heterogeneous: population-specific rate multiplier
- context: founder-derived 3-mer context changes mutation susceptibility
- regime_shift: 5x mutation rate after half the generations
- linkage_only: neutral whole-haplotype inheritance, no recombination (same generator as neutral)
- selection_only: selection on one allele
- epistasis: additional non-additive log fitness interaction between two alleles

Note: `neutral` and `linkage_only` are deliberately the same mechanism; they are
not an independent linkage-vs-unlinked control. A future extension must implement
recombination/unlinked loci for that comparison.

## Information boundaries
`observed_genomes.csv` contains all timepoints for **evaluation infrastructure**;
it is NOT a model training file. Use `historical_inputs.csv` to select only the
clone IDs available at each cutoff. `evaluation_targets_RESTRICTED.csv` and
`simulation_truth_RESTRICTED.json` are evaluator-only. The true fitness sites and
regime shift are not model inputs.

Fold rows share populations and trajectories; they are NOT independent replicates.
Seeds are separate simulated runs. Do not use future targets for model selection.

## Run
`python -m unittest -v test_v3_simulate.py`
`python -u v3_simulate.py --seeds 11 23 42 --output ../LTEE_prospective/v3a_simulation`

## Scope / limitations
Small toy haplotypes, high per-site mutation probability for tractable CPU runs,
no real genome lengths, no indels, no recombination, no phylogenetic tree, no
explicit mutation spectrum heterogeneity, and no calibrated LTEE generation units.
This phase tests simulation plumbing and chronological integrity, NOT whether M4/M5
can learn the mechanisms. Model benchmarks and identifiability tests come next.
