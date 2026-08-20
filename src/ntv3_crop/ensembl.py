from __future__ import annotations

import time
from io import StringIO
from urllib.parse import quote

import pandas as pd
import requests

from ntv3_crop.config import ENSEMBL_DATASETS, GO_TERMS

PLANTS_BIOMART = "https://plants.ensembl.org/biomart/martservice"
ENSEMBL_REST = "https://rest.ensembl.org"
HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}


def _get(url: str, **kwargs) -> requests.Response:
    for attempt in range(5):
        response = requests.get(url, timeout=60, **kwargs)
        if response.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        return response
    raise RuntimeError(f"Rate-limited by Ensembl: {url}")


def biomart_genes_for_go(species: str, go_ids: tuple[str, ...]) -> pd.DataFrame:
    dataset = ENSEMBL_DATASETS[species]
    go_filter = ",".join(go_ids)
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="plants_mart" formatter="TSV" header="1" uniqueRows="1">
  <Dataset name="{dataset}" interface="default">
    <Filter name="go_parent_term" value="{go_filter}"/>
    <Attribute name="ensembl_gene_id"/>
    <Attribute name="external_gene_name"/>
    <Attribute name="chromosome_name"/>
    <Attribute name="start_position"/>
    <Attribute name="end_position"/>
    <Attribute name="strand"/>
  </Dataset>
</Query>
"""
    response = _get(PLANTS_BIOMART, params={"query": xml})
    text = response.text.strip()
    if not text or text.startswith("Query ERROR") or "<html" in text.lower():
        raise RuntimeError(f"BioMart query failed for {species}: {text[:400]}")
    return pd.read_csv(StringIO(text), sep="\t")


def biomart_all_genes(species: str) -> pd.DataFrame:
    dataset = ENSEMBL_DATASETS[species]
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="plants_mart" formatter="TSV" header="1" uniqueRows="1">
  <Dataset name="{dataset}" interface="default">
    <Attribute name="ensembl_gene_id"/>
    <Attribute name="external_gene_name"/>
    <Attribute name="chromosome_name"/>
    <Attribute name="start_position"/>
    <Attribute name="end_position"/>
    <Attribute name="strand"/>
  </Dataset>
</Query>
"""
    response = _get(PLANTS_BIOMART, params={"query": xml})
    return pd.read_csv(StringIO(response.text), sep="\t")


def fetch_gene_window(
    gene_id: str,
    expand_5prime: int = 2000,
    expand_3prime: int = 500,
) -> str:
    url = (
        f"{ENSEMBL_REST}/sequence/id/{quote(gene_id)}"
        f"?type=genomic&expand_5prime={expand_5prime}&expand_3prime={expand_3prime}"
    )
    response = _get(url, headers=HEADERS)
    return response.json()["seq"]


def labeled_gene_table(species: str) -> pd.DataFrame:
    frames = []
    for label, go_ids in GO_TERMS.items():
        try:
            df = biomart_genes_for_go(species, go_ids)
        except Exception as exc:  # BioMart schema can change across Ensembl releases
            print(f"[warn] {species} / {label}: {exc}")
            continue
        if df.empty:
            continue
        df = df.rename(columns={"ensembl_gene_id": "gene_id"})
        df["label"] = label
        frames.append(df[["gene_id", "label"]])
    if not frames:
        raise RuntimeError(f"No GO-labeled genes retrieved for {species}")
    long = pd.concat(frames, ignore_index=True).drop_duplicates()
    wide = (
        long.assign(present=1)
        .pivot_table(index="gene_id", columns="label", values="present", fill_value=0)
        .reset_index()
    )
    for label in GO_TERMS:
        if label not in wide.columns:
            wide[label] = 0
    wide["species"] = species
    return wide
