# Literature

Foundational papers and standard references, grouped by where the course uses
them. Preference is for original publications over secondary explanations.

## Economic dispatch and unit commitment

- Wood, Wollenberg & Sheblé, *Power Generation, Operation and Control*, 3rd ed.,
  Wiley, 2013. The classical treatment; chapters 3–4 for dispatch and the
  equal-incremental-cost rule. — *T01*
- Morales-España, Latorre & Ramos, "Tight and compact MILP formulation for the
  thermal unit commitment problem", *IEEE Trans. Power Syst.* 28(4), 2013. Why
  formulation tightness, not solver choice, decides tractability. — *T02*

## Optimization theory

- Boyd & Vandenberghe, *Convex Optimization*, Cambridge University Press, 2004.
  Chapter 5 for duality and KKT. — *T02, T05*
- Bertsimas & Tsitsiklis, *Introduction to Linear Optimization*, Athena
  Scientific, 1997. — *T02*
- Nocedal & Wright, *Numerical Optimization*, 2nd ed., Springer, 2006. — *T04*

## Optimal power flow

- Carpentier, "Contribution à l'étude du dispatching économique", *Bulletin de
  la Société Française des Électriciens*, 1962. The original OPF. — *T04*
- Cain, O'Neill & Castillo, "History of optimal power flow and formulations",
  FERC staff paper, 2012. — *T04*
- Stott, Jardim & Alsaç, "DC power flow revisited", *IEEE Trans. Power Syst.*
  24(3), 2009. An honest account of what DC does and does not do. — *T03*
- Bukhsh, Grothey, McKinnon & Trodden, "Local solutions of the optimal power
  flow problem", *IEEE Trans. Power Syst.* 28(4), 2013. Small instances with
  multiple local optima — the reason Tutorial 04 does not claim global
  optimality from a multi-start. — *T04*
- Capitanescu, "Critical review of recent advances and further developments
  needed in AC optimal power flow", *Electric Power Systems Research* 136,
  2016. — *T10*

## Pricing and networks

- Schweppe, Caramanis, Tabors & Bohn, *Spot Pricing of Electricity*, Kluwer,
  1988. The origin of locational marginal pricing. — *T03*
- Litvinov, Zheng, Rosenwald & Shamsollahi, "Marginal loss modeling in LMP
  calculation", *IEEE Trans. Power Syst.* 19(2), 2004. — *T03*

## Convex relaxations

- Jabr, "Radial distribution load flow using conic programming", *IEEE Trans.
  Power Syst.* 21(3), 2006. The conic branch-flow formulation. — *T05*
- Farivar & Low, "Branch flow model: relaxations and convexification (parts I
  and II)", *IEEE Trans. Power Syst.* 28(3), 2013. — *T05*
- Gan, Li, Topcu & Low, "Exact convex relaxation of optimal power flow in radial
  networks", *IEEE Trans. Autom. Control* 60(1), 2015. The exactness conditions
  the course tests against empirically. — *T05*
- Low, "Convex relaxation of optimal power flow (parts I and II)", *IEEE Trans.
  Control of Network Systems* 1(1) and 1(2), 2014. — *T05*
- Lavaei & Low, "Zero duality gap in optimal power flow problem", *IEEE Trans.
  Power Syst.* 27(1), 2012. The SDP relaxation. — *T05*
- Molzahn & Hiskens, *A Survey of Relaxations and Approximations of the Power
  Flow Equations*, NOW Publishers, 2019. The reference survey, and the best
  single source on the relaxation-versus-approximation distinction. — *T05, T10*

## Storage and multi-period

- Morales, Conejo, Madsen, Pinson & Zugno, *Integrating Renewables in
  Electricity Markets*, Springer, 2014. — *T06*
- Sioshansi, Denholm, Jenkin & Weiss, "Estimating the value of electricity
  storage in PJM", *Energy Economics* 31(2), 2009. — *T06*
- Pozo, "Convex hull formulations for linear modeling of energy storage
  systems", *IEEE Trans. Power Syst.* 38(6), 2023. When the complementarity
  binary is and is not needed. — *T06*

## Uncertainty

- Bienstock, Chertkov & Harnett, "Chance-constrained optimal power flow:
  risk-aware network control under uncertainty", *SIAM Review* 56(3), 2014. — *T07*
- Roald & Andersson, "Chance-constrained AC optimal power flow: reformulations
  and efficient algorithms", *IEEE Trans. Power Syst.* 33(3), 2018. — *T07*
- Vrakopoulou, Margellos, Lygeros & Andersson, "A probabilistic framework for
  reserve scheduling and N-1 security assessment", *IEEE Trans. Power Syst.*
  28(4), 2013. — *T07*
- Calafiore & Campi, "The scenario approach to robust control design", *IEEE
  Trans. Autom. Control* 51(5), 2006. Sample-size guarantees for the scenario
  method. — *T07*
- Roald, Pozo, Papavasiliou, Molzahn, Kazempour & Conejo, "Power systems
  optimization under uncertainty: a review of methods and applications",
  *Electric Power Systems Research* 214, 2023. — *T10*

## Decomposition

- Benders, "Partitioning procedures for solving mixed-variables programming
  problems", *Numerische Mathematik* 4, 1962. — *T08*
- Geoffrion, "Generalized Benders decomposition", *J. Optimization Theory and
  Applications* 10(4), 1972. — *T08*
- Rahmaniani, Crainic, Gendreau & Rei, "The Benders decomposition algorithm: a
  literature review", *European J. Operational Research* 259(3), 2017. — *T08*
- Dantzig & Wolfe, "Decomposition principle for linear programs", *Operations
  Research* 8(1), 1960. — *T09*
- Barnhart, Johnson, Nemhauser, Savelsbergh & Vance, "Branch-and-price: column
  generation for solving huge integer programs", *Operations Research* 46(3),
  1998. — *T09*
- Lübbecke & Desrosiers, "Selected topics in column generation", *Operations
  Research* 53(6), 2005. — *T09*
- Geoffrion, "Lagrangean relaxation for integer programming", *Mathematical
  Programming Study* 2, 1974. The integrality-property result that explains why
  the Dantzig-Wolfe bound equalled the LP relaxation in Tutorial 09. — *T09*

### Benders decomposition with convexified AC power flow

The specific combination Tutorial 08 builds towards. When reading these, keep
five questions in view, because papers differ on all of them: radial or meshed
network; exact or inexact relaxation; operational or planning problem;
continuous or integer master variables; how feasibility cuts are constructed.

- **Geoffrion (1972)**, above, is the theoretical basis. Classical Benders rests
  on LP duality; Generalized Benders extends the same argument to suitable
  convex nonlinear subproblems, which is precisely what admits a conic
  subproblem.
- **Farivar & Low (2013)** and **Gan et al. (2015)**, above, supply the
  relaxation and the conditions under which it is exact — hence whether a
  certificate obtained on the relaxation transfers to the AC problem at all.
- Hijazi, Coffrin & Van Hentenryck, "Convex quadratic relaxations for
  mixed-integer nonlinear programs in power systems", *Mathematical Programming
  Computation* 9, 2017. Convex relaxations under integer decisions — the MISOCP
  structure Tutorial 08 decomposes.
- Coffrin, Hijazi & Van Hentenryck, "The QC relaxation: a theoretical and
  computational study on optimal power flow", *IEEE Trans. Power Syst.* 31(4),
  2016. A relaxation usually tighter than SOC, and a useful comparison point.

The caveat the course keeps returning to: solving the MISOCP rigorously solves
the **relaxation**. It certifies the original nonconvex AC problem only when the
relaxation is exact, or when an independently obtained AC-feasible point closes
the interval.

## Software

- Bynum, Hackebeil, Hart, Laird, Nicholson, Siirola, Watson & Woodruff, *Pyomo —
  Optimization Modeling in Python*, 3rd ed., Springer, 2021.
- Thurner, Scheidler, Schäfer, Menke, Dollichon, Meier, Meinecke & Braun,
  "pandapower — an open-source Python tool for convenient modeling, analysis and
  optimization of electric power systems", *IEEE Trans. Power Syst.* 33(6), 2018.
- Meinecke, Sarajlić, Drauz, Klettke, Lauven, Rehtanz, Moser & Braun, "SimBench —
  a benchmark dataset of electric power systems", *Energies* 13(12), 2020.
- Babaeinejadsarookolaee et al., "The power grid library for benchmarking AC
  optimal power flow algorithms", arXiv:1908.02788, 2019. PGLib-OPF.
- `opf-potpourri`, RWTH Aachen IAEW.
  https://github.com/RWTH-IAEW/opf-potpourri — pinned at tag `v0.7.0`. Note that
  `potpourri.research` is in no published release; see `course_overview.md` §2.3.

## Pedagogical inspiration

- Heidari & Bo, *Power System Operations Modeling and Optimization Using Pyomo*,
  Missouri S&T, 2025. https://scholarsmine.mst.edu/gradstudent_works/6/

  The philosophy is borrowed — mathematics, power-system applications, Pyomo,
  executable examples — not the content. This course pushes considerably further
  into convex relaxation, conic programming, chance constraints and
  decomposition, and it treats the relaxation-versus-approximation distinction
  as a central theme rather than an aside.
