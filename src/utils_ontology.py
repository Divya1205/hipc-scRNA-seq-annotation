"""
utils_ontology.py — load the HIPC Cell Ontology and provide helpers for:
  - exact-string membership check
  - parent lookup (for hierarchy fallback)
  - alias map from external tools (CellTypist, Azimuth) to challenge terms

The official ontology file is named like CT_Ontology_Spreadsheet_*.tsv|xlsx.
We expect columns roughly:
    celltype  | parent  | level | definition  | ...

Adapt _COLS if the official file uses different names.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path


_COLS = {
    "celltype": "celltype",   # the canonical label string
    "parent":   "parent",     # the immediate parent label string
}

# The official HIPC ontology uses these exact headers. We auto-map common
# variants (case / spacing) so the code works without hand-editing.
_CELLTYPE_ALIASES = ["celltype", "cell_type", "cell type", "Celltype", "CellType"]
_PARENT_ALIASES   = ["parent", "parent_class", "Parent Class", "parent class",
                     "ParentClass", "parent_term", "Parent"]


def _resolve_col(df: pd.DataFrame, aliases: list[str], role: str) -> str:
    # exact match first, then case-insensitive
    for a in aliases:
        if a in df.columns:
            return a
    lower = {c.lower().strip(): c for c in df.columns}
    for a in aliases:
        if a.lower().strip() in lower:
            return lower[a.lower().strip()]
    raise KeyError(f"Could not find the '{role}' column. "
                   f"Looked for {aliases}; file has {list(df.columns)}.")


class Ontology:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        ct_col = _resolve_col(self.df, _CELLTYPE_ALIASES, "celltype")
        pa_col = _resolve_col(self.df, _PARENT_ALIASES, "parent")
        # Normalize to canonical internal names
        self.df = self.df.rename(columns={ct_col: "celltype", pa_col: "parent"})
        for c in ("celltype", "parent"):
            self.df[c] = self.df[c].astype(str).str.strip()
        self.terms = set(self.df["celltype"].dropna().tolist())
        self.parent = dict(zip(self.df["celltype"], self.df["parent"]))

    # -- canonical membership ------------------------------------------------
    def __contains__(self, term: str) -> bool:
        return term in self.terms

    # -- parent-walk for hierarchy fallback ---------------------------------
    def parent_of(self, term: str) -> str | None:
        p = self.parent.get(term)
        if p is None or str(p).lower() in {"nan", "none", ""}:
            return None
        return p

    def ancestors(self, term: str) -> list[str]:
        """Walk to root. Returns [parent, grandparent, ...]."""
        out = []
        cur = self.parent_of(term)
        while cur is not None and cur not in out:  # cycle guard
            out.append(cur)
            cur = self.parent_of(cur)
        return out

    def fallback(self, term: str, levels_up: int = 1) -> str:
        """
        Climb `levels_up` steps; if we hit None, return whatever was last valid.
        Use this for low-confidence cells: instead of guessing a leaf, walk up.
        """
        cur = term
        for _ in range(levels_up):
            p = self.parent_of(cur)
            if p is None:
                return cur
            cur = p
        return cur


def load_ontology(path: str | Path) -> Ontology:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep="\t")
    return Ontology(df)


# -----------------------------------------------------------------------------
# Alias map: CellTypist Immune_All_Low labels -> HIPC ontology terms (39 terms)
# -----------------------------------------------------------------------------
# The HIPC ontology is COARSER than CellTypist's native output, so many fine
# CellTypist subtypes map up to a single HIPC term. Anything not mapped here
# falls through map_label()'s case-insensitive match, then to parent fallback.
#
# Valid HIPC terms (target side) MUST be exactly:
#   Blood Cell, Platelet, RBC, HSC, Doublet, Leukocyte, Lymphoid Cell, NK Cell,
#   T Cell, CD4 T Cell (ab), CD4 Naive / T Central Memory, CD4 T Effector Memory,
#   Treg, CD8 T Cell (ab), CD8 Naive / T Central Memory,
#   CD8 Cytotoxic / T Effector Memory, gdT Cell, MAIT Cell, NKT Cell, B Cell,
#   Effector B, Plasma Cell, Plasmablast, Naive B Cell, Memory B Cell,
#   Myeloid Cell, Monocyte, Classical Monocyte, Non-Classical Monocyte,
#   Intermediate Monocyte, Granulocyte, Neutrophil, Eosinophil, Basophil,
#   Mast Cell, DC, Plasmacytoid DC, Conventional DC 1, Conventional DC 2
CELLTYPIST_ALIAS = {
    # ---- CD4 T ----
    "Tcm/Naive helper T cells":            "CD4 Naive / T Central Memory",
    "Naive CD4 T cells":                   "CD4 Naive / T Central Memory",
    "CD4 Tcm":                             "CD4 Naive / T Central Memory",
    "Helper T cells":                      "CD4 T Cell (ab)",
    "Tem/Effector helper T cells":         "CD4 T Effector Memory",
    "Tem/Temra helper T cells":            "CD4 T Effector Memory",
    "Tem/Trm helper T cells":              "CD4 T Effector Memory",
    "T helper cells":                      "CD4 T Cell (ab)",
    "Follicular helper T cells":           "CD4 T Cell (ab)",
    "Regulatory T cells":                  "Treg",
    "Tregs":                               "Treg",
    # ---- CD8 T ----
    "Tcm/Naive cytotoxic T cells":         "CD8 Naive / T Central Memory",
    "Naive CD8 T cells":                   "CD8 Naive / T Central Memory",
    "Tem/Temra cytotoxic T cells":         "CD8 Cytotoxic / T Effector Memory",
    "Tem/Trm cytotoxic T cells":           "CD8 Cytotoxic / T Effector Memory",
    "Tem/Effector cytotoxic T cells":      "CD8 Cytotoxic / T Effector Memory",
    "Cytotoxic T cells":                   "CD8 Cytotoxic / T Effector Memory",
    # ---- Unconventional T ----
    "MAIT cells":                          "MAIT Cell",
    "gamma-delta T cells":                 "gdT Cell",
    "gdT cells":                           "gdT Cell",
    "NKT cells":                           "NKT Cell",
    "Double-negative thymocytes":          "T Cell",
    "Double-positive thymocytes":          "T Cell",
    "T cells":                             "T Cell",
    # ---- NK ----
    "NK cells":                            "NK Cell",
    "CD16+ NK cells":                      "NK Cell",
    "CD16- NK cells":                      "NK Cell",
    "CD56bright NK cells":                 "NK Cell",
    "ILC":                                 "Lymphoid Cell",
    "ILC3":                                "Lymphoid Cell",
    # ---- B ----
    "Naive B cells":                       "Naive B Cell",
    "Memory B cells":                      "Memory B Cell",
    "Age-associated B cells":              "Memory B Cell",
    "Germinal center B cells":             "B Cell",
    "Proliferative germinal center B cells":"B Cell",
    "Follicular B cells":                  "B Cell",
    "Large pre-B cells":                   "B Cell",
    "Small pre-B cells":                   "B Cell",
    "Pro-B cells":                         "B Cell",
    "Plasma cells":                        "Plasma Cell",
    "Plasmablasts":                        "Plasmablast",
    "B cells":                             "B Cell",
    # ---- Monocytes ----
    "Classical monocytes":                 "Classical Monocyte",
    "Non-classical monocytes":             "Non-Classical Monocyte",
    "Intermediate monocytes":              "Intermediate Monocyte",
    "Monocytes":                           "Monocyte",
    "Mono-mac":                            "Monocyte",
    "Macrophages":                         "Myeloid Cell",
    "Alveolar macrophages":                "Myeloid Cell",
    "Erythrophagocytic macrophages":       "Myeloid Cell",
    "Kupffer cells":                       "Myeloid Cell",
    # ---- DC ----
    "DC":                                  "DC",
    "DC1":                                 "Conventional DC 1",
    "DC2":                                 "Conventional DC 2",
    "DC3":                                 "DC",
    "Migratory DCs":                       "DC",
    "pDC":                                 "Plasmacytoid DC",
    "pDC precursor":                       "Plasmacytoid DC",
    "Plasmacytoid DCs":                    "Plasmacytoid DC",
    "Conventional dendritic cells":        "DC",
    # ---- Granulocytes ----
    "Neutrophils":                         "Neutrophil",
    "Neutrophil-myeloid progenitors":      "Granulocyte",
    "Eosinophils":                         "Eosinophil",
    "Basophils":                           "Basophil",
    "Mast cells":                          "Mast Cell",
    "Granulocytes":                        "Granulocyte",
    # ---- Progenitors / other ----
    "HSC/MPP":                             "HSC",
    "Hematopoietic stem cells":            "HSC",
    "Megakaryocyte precursor":             "Platelet",
    "Megakaryocytes/platelets":            "Platelet",
    "Platelets":                           "Platelet",
    "Erythrocytes":                        "RBC",
    "Erythroid":                           "RBC",
    "Early erythroid":                     "RBC",
    "Mid erythroid":                       "RBC",
    "Late erythroid":                      "RBC",
    "Doublet":                             "Doublet",
    "Doublets":                            "Doublet",
}


def map_label(label: str, ont: Ontology, alias: dict | None = None) -> str | None:
    """
    Map an external label to the challenge ontology.
      1) If already an exact term -> return it.
      2) Try alias dictionary -> return mapped term (if in ontology).
      3) Try case-insensitive match -> return canonical casing.
      4) Return None (caller should fallback to parent or 'Unknown').
    """
    if label is None:
        return None
    label = str(label).strip()
    if label in ont:
        return label
    alias = alias or CELLTYPIST_ALIAS
    if label in alias and alias[label] in ont:
        return alias[label]
    # case-insensitive last resort
    lc = {t.lower(): t for t in ont.terms}
    if label.lower() in lc:
        return lc[label.lower()]
    return None