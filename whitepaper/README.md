# Generalist technical whitepaper

[Read the PDF](generalist.pdf). The editable source is [main.tex](main.tex),
with one BibTeX entry per external reference in [references.bib](references.bib).

The paper describes the implemented runtime, its permission and persistence
boundaries, the three finite TLA+ models, and what the observed traces and
negative tests actually establish. It makes no claim of exhaustive Rust
refinement, task-solving superiority, or hostile multi-tenant isolation.

Build with a TeX distribution containing `latexmk`, pdfLaTeX, BibTeX, and the
standard packages named in `main.tex`:

```sh
make -C whitepaper
```

From the repository root, `make whitepaper` is equivalent. `make check` runs the
implementation and model gates; it does not require a TeX installation. The
paper's recorded validation is in [evidence/validation.md](evidence/validation.md).
The evidence identifies a local source snapshot; it is not a hosted release or
an assertion that later changes have been checked. Refresh the checks and
source hashes when updating implementation claims or numerical results.
