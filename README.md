**Working Documents – Third Year Project**

Folder structure:
- data/: source market historical data used for preprocessing and simulation (Investing.com).
- main_code/: final scripts used in the dissertation.
- exploratory_code/: previous or exploratory scripts not used as final implementations.
- outputs/: generated utility matrices and visualizations.

=================================================================================

*Main final scripts (main_code/):*
- MonteCarloModified_N8.py: Final Monte Carlo simulator with 8 portfolios and 5 market states construction
                            that produces utility matrices and computes state feature functions.

- Portfolios_Complete.sql:  SQL preprocessing pipeline.

- solver_N8_UV_profiles_version2.py:    Final E-admissibility solver.

- geometric_intersection.py:    Final geometric visualizer based on hyperplanes intersection.

===

*Exploratory scripts (exploratory_code/):*
- MonteCarloModified.py:    Earlier Monte Carlo simulation script based on the first 10-portfolio setting,
                            used during the exploratory phase before the final reduction to 8 portfolios.

- geometric_almost_admissible.py:   Exploratory script developed to investigate a possible geometric representation
                                    of near-miss solutions. This approach was later discarded and is not part of
                                    the final dissertation results.

- geometric_submodel_UV.py: Earlier geometric visualization script based on simplex grading/sampling methods,
                            used before the final exact intersection-based implementation.

===

*Outputs (outputs/):*
- utility_matrices/:    Text outputs containing intermediate and final utility matrices produced during the
                        simulation and reduction stages.

- visualizations/:  Generated geometric figures of admissible regions and portfolio optimality regions.

    - visualizations/ProfileA/: Final visualizations for Profile A.

    - visualizations/ProfileB/: final visualizations for Profile B.

    - visualizations/ProfileC/: final visualizations for Profile C.

    - visualizations/baseline/: Visualizations for a preliminary credal set construction that is not included 
                                in the dissertation itself.

    - visualizations/near_miss_discarded/:  Visualizations saved as attempts to represent near-miss solutions
                                            but discarded due to their lack of coherence.
