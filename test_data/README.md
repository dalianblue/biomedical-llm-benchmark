# BRCA1 Analysis — Input Data Preparation

Four bioinformatics inputs, all under `input/`. Every file was downloaded via the
local proxy `http://127.0.0.1:7890` (NCBI / GEO / UniProt direct connections are
unstable from this machine).

## Inputs

### 1. Sequence — `input/seq/`
- **File:** `BRCA1_NM_007294.4.fasta`
- **Source:** NCBI Nucleotide, accession `NM_007294.4`
  (Homo sapiens BRCA1 DNA repair associated, transcript variant 1, mRNA)
- **Fetched from:**
  `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NM_007294.4&rettype=fasta&retmode=text`
- **Length:** 7,088 nt (1 FASTA record)

### 2. Simulated mutation — `input/mutation/`
- **Files:**
  - `BRCA1_NM_007294.4_20mut.fasta` (mutated sequence)
  - `mutation_events.log` (event list)
- **Generator:** `scripts/mutate_brca1.py`
- **Rule (mirrors MutationSimulator's substitution-only `subNucleotide`):**
  each event replaces one base with a *different* base; sequence length is preserved.
- **Result:** exactly **20 point mutations**, length unchanged at **7,088 nt**.
- **Mutated positions:** 205, 245, 261, 713, 768, 840, 913, 1144, 1792, 1829,
  2007, 2254, 3457, 4468, 4838, 5239, 5544, 6034, 6068, 6075.

### 3. Expression matrix — `input/expression/`
- **Dataset:** GEO **GSE100866** — *CITE-seq: Large scale simultaneous measurement
  of epitopes and transcriptomes in single cells* (Stoeckius et al. 2017, Homo sapiens)
- **Source file:** `GSE100866_CD8_merged-RNA_umi.csv.gz`
  (gene × cell UMI count matrix, genes in rows / cell barcodes in columns)
- **Downloaded from:**
  `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE100nnn/GSE100866/suppl/GSE100866_CD8_merged-RNA_umi.csv.gz`
- **Matrix size:** 11,757 genes × 1,774 cells
- **Derived files (top 500 most highly expressed genes by total UMI):**
  - `top500_genes.tsv` — ranked gene list (`gene`, `total_umi`, `rank`)
  - `top500_expression_matrix.tsv` — 500 genes × 1,774 cells matrix
  - `top500_summary.tsv` — run summary/provenance
- **Generator:** `scripts/extract_top500_genes.py`
- **Note:** the top-500 genes account for **56.60%** of total UMI.

### 4. Protein structure — `input/protein/`
- **Source:** UniProtKB accession **P38398** (`BRCA1_HUMAN`, 1,863 aa)
- **Fetched from:** `https://rest.uniprot.org/uniprotkb/P38398.txt`
- **Files:**
  - `BRCA1_P38398_p53_binding_domain.fasta` — p53-binding domain sequence
  - `BRCA1_P38398_secondary_structure.tsv` — HELIX/STRAND/TURN annotation
  - `BRCA1_P38398_summary.tsv` — summary/provenance
- **P53-binding domain:** amino acids **224–500** (277 aa).
  Mapping per Zhang et al. 1998, Oncogene 16:1713–1721, **PMID 9582019**
  ("the interacting regions map, in vitro, to aa 224–500 of BRCA1 and the
  C-terminal domain of p53").
- **Secondary structure:** 42 UniProt features (19 helix, 18 strand, 5 turn),
  derived from PDB structures. **None overlap aa 224–500** — the p53-binding
  domain is an intrinsically disordered region with no experimentally resolved
  secondary structure in UniProt.

## Scripts
- `scripts/mutate_brca1.py` — generate the 20-mutation BRCA1 FASTA + log
- `scripts/extract_top500_genes.py` — top-500 genes from the GSE100866 UMI matrix
- `scripts/extract_brca1_protein.py` — p53-binding domain + secondary structure from UniProt

## Environment
- Python venv (reused from the TCGA-LAML task):
  `/Users/yuzhang/Documents/wisp-science/tcga-laml-analysis/venv/bin/python3`
  (pandas 2.3.3, numpy 2.5.2)
