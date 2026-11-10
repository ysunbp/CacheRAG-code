# CacheRAG supplementary material

Supplement to *CacheRAG: History-Aware Query Planning over Heterogeneous Knowledge Graph Interfaces* by Yushi Sun.

**[Read the supplementary PDF](supplement.pdf)**

The main manuscript cites this PDF directly:
https://github.com/ysunbp/CacheRAG-code/blob/main/supplementary/supplement.pdf

## Contents

| Section | Material |
| --- | --- |
| II | CRAG workload composition |
| III | Evaluation critic and reported agreement figures |
| IV-A, Table III | Full head/torso/tail comparison |
| IV-B, Figure 1 | MMR parameter sensitivity |
| IV-C, Figure 2 | Multi-domain routing comparison |
| V-A | Logical ISR extraction prompt skeleton |
| V-B | Offline question-generation prompt skeleton |
| V-C, Table IV | Domain-description update prompt and examples |
| VI | Cost interpretation |
| VII | Reproduction context |

This document is separate from the main submission PDF. It collects reported measurements and explanatory prompt skeletons; no additional experiments were run for this revision.

## Source and compilation

The editable LaTeX source and required figures are included. Set `supplement.tex` as the main file in Overleaf with pdfLaTeX, or run:

```sh
pdflatex supplement
bibtex supplement
pdflatex supplement
pdflatex supplement
```

Alternatively, use `tectonic supplement.tex`. The supplied PDF was compiled with Tectonic.
