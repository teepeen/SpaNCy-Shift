# Reference map

✓ = found and checked in a web search (2026-10-07). ○ = standard citation from memory, so check it
in Google Scholar before use. Key = BibTeX key in `references.bib`.

## Where each reference is used

| Key | Reference | Use it for | Chapter | |
|---|---|---|---|---|
| lin2018cycif | Lin et al. 2018, eLife, t-CyCIF | Definition of CyCIF and of a *cycle* | 1.1, 2.1 | ✓ |
| lin2015cycif | Lin et al. 2015, Nat Commun, CyCIF in cells | Origin of the method | 2.1 | ✓ |
| hickey2022primer | Hickey et al. 2022, Nat Methods, primer | General MTI intro; reproducibility; study design | 1.1, 2.1 | ✓ |
| schapiro2022miti | Schapiro et al. 2022, Nat Methods, MITI | Units: specimen, slide, image, channel metadata | 2.1 | ✓ |
| schapiro2022mcmicro | Schapiro et al. 2022, Nat Methods, MCMICRO | Image → single-cell table | 2.1 | ✓ |
| leek2010batch | Leek et al. 2010, Nat Rev Genet | Definition and danger of batch effects | 1.2, 2.2 | ✓ |
| intensity2023clinical | "Accounting for intensity variation…" 2023 | Sources of variation in mIF cohorts | 2.2 | ✓ (authors: fill in) |
| chang2020restore | Chang et al. 2020, Commun Biol, RESTORE | Negative-cell normalization; variation sources | 2.2, 2.3 | ✓ |
| harris2022slide | Harris et al. 2022, Bioinformatics | Slide effects; ComBat and registration for MTI | 1.3, 2.3 | ✓ |
| harris2022mxnorm | Harris, Wrobel, Vandekar 2022, JOSS, mxnorm | MXnorm baseline | 2.3, 3.5 | ✓ |
| wang2025uniform | Wang/Zeng et al. 2025, Cell Rep Methods, UniFORM | Main baseline; PRAD-CyCIF dataset; metrics | 1.3, 2.3, 4.1, 6.3 | ✓ |
| johnson2007combat | Johnson et al. 2007, Biostatistics, ComBat | ComBat baseline | 2.3 | ○ |
| vangassen2020cytonorm | Van Gassen et al. 2020, Cytometry A | Cytometry normalization | 2.3 | ✓ |
| pedersen2022cycombine | Pedersen et al. 2022, Nat Commun | Cytometry batch integration | 2.3 | ✓ |
| arevalo2024imaging | Arevalo et al. 2024 (bioRxiv 2023) | Batch correction benchmark, image-based profiling | 2.3 | ✓ (authors: verify) |
| korsunsky2019harmony | Korsunsky et al. 2019, Nat Methods | Embedding-based integration | 2.3 | ✓ |
| lopez2018scvi | Lopez et al. 2018, Nat Methods, scVI | Deep generative integration | 2.3 | ○ |
| gretton2012mmd | Gretton et al. 2012, JMLR | MMD definition | 2.3, 3.3 | ○ |
| shaham2017mmdresnet | Shaham et al. 2017, Bioinformatics | Direct prior art of Stage 2 | 1.3, 2.3, 5.2 | ✓ |
| ganin2016dann | Ganin et al. 2016, JMLR | Gradient reversal / adversarial branch | 2.4, 3.3 | ○ |
| chen2020simclr | Chen et al. 2020, ICML | NT-Xent | 2.4, 3.3 | ○ |
| brody2022gatv2 | Brody, Alon, Yahav 2022, ICLR | GATv2 encoder | 2.4, 3.3 | ✓ |
| huber1964 | Huber 1964, Ann Math Stat | Huber loss | 2.4 | ○ |
| zhou2023staligner | Zhou et al. 2023, Nat Comput Sci | Spatial GNN integration (related work) | 6.3 | ✓ |
| long2023graphst | Long et al. 2023, Nat Commun | GNN + contrastive spatial integration | 6.3 | ✓ |
| buttner2019kbet | Büttner et al. 2019, Nat Methods | kBET | 2.5, 4.3 | ✓ |
| luecken2022scib | Luecken et al. 2022, Nat Methods | Removal vs conservation; multi-metric evaluation | 2.5, 4.4, 6.3 | ✓ |
| tran2020benchmark | Tran et al. 2020, Genome Biol | Batch-correction benchmark | 2.5 | ○ |
| rousseeuw1987silhouette | Rousseeuw 1987, J Comput Appl Math | Silhouette | 2.5, 4.3 | ○ |
| tong2024cfm | Tong et al. 2024, TMLR, OT-CFM | Only if CFM stays | 2.3, 5.6 | ✓ |
| bunne2023cellot | Bunne et al. 2023, Nat Methods, CellOT | OT on single cells incl. multiplexed imaging (qualifies novelty) | 2.3 | ✓ |
| ho2020ddpm | Ho et al. 2020, NeurIPS | Only if DDPM stays | 2.3 | ○ |
| meng2022sdedit | Meng et al. 2022, ICLR, SDEdit | Only if DDPM stays | 2.3 | ✓ |

## Still missing
- The original source publication of PRAD-CyCIF (see UniFORM's data-availability section).
- What a *batch* physically is in PRAD-CyCIF (staining run / imaging run / slide).
- A peer-reviewed reference for GMM-based positive/negative gating in MTI.
- Scanner documentation for "scene" (Zeiss Axioscan manual; cite as a footnote / technical doc).
- A recent (2025–2026) check for learned MTI normalizers, so the novelty claim is current.

## Links (from the search sessions)
- UniFORM: https://www.sciencedirect.com/science/article/pii/S2667237525002085
- Harris 2022: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8896603/
- mxnorm: https://joss.theoj.org/papers/10.21105/joss.04180.pdf
- RESTORE: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7062831/
- t-CyCIF: https://elifesciences.org/articles/31657
- MITI: https://ccsp.hms.harvard.edu/wp-content/uploads/2022/03/Schapiro-2022-MITI.pdf
- Hickey primer: https://pmc.ncbi.nlm.nih.gov/articles/PMC9264278/
- MCMICRO: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8916956/
- Leek 2010: https://doi.org/10.1038/nrg2825
- Intensity variation 2023: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10556275/
- kBET: https://portal.fis.tum.de/en/publications/a-test-metric-for-assessing-single-cell-rna-seq-batch-correction/
- scIB: https://www.biorxiv.org/content/10.1101/2020.05.22.111161v2.full.pdf+html
- MMD-ResNet: https://arxiv.org/abs/1610.04181v3
- CytoNorm: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7078957/
- cyCombine: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8971492/
- Harmony: https://www.biorxiv.org/content/10.1101/461954.full.pdf
- Image-profiling benchmark: https://www.biorxiv.org/content/10.1101/2023.09.15.558001v1.full.pdf
- GATv2: https://arxiv.org/abs/2105.14491v1
- STAligner: https://link.springer.com/article/10.1038/s43588-023-00543-x
- GraphST: https://link.springer.com/10.1038/s41467-023-36796-3
- OT-CFM: https://arxiv.org/abs/2302.00482v4
- CellOT: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10630137/
- SDEdit: https://arxiv.org/abs/2108.01073v2
- Zeiss Axioscan manual (scene): https://www.research.uky.edu/uploads/axio-scan-manual-ukupdated
