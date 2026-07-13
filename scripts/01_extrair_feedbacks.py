#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_extrair_feedbacks.py (v7)

- Le um parquet cru do AES-ENEM (ex.: parquet/train-sourceAOnly.parquet)
- Extrai feedback humano da Competência 5 (C5) no campo `specific_comment`
- Cria um identificador por redação: `essay_hash` (SHA-256 do essay_text)
- Gera 3 saídas:
  1) feedbacks_extraidos_c5_<stem>.parquet  (todas as redações)
  2) feedbacks_validos_c5_<stem>.parquet   (redações com feedback_c5 válido)
  3) relatorio_extracao_c5_<stem>.json     (diagnóstico)

Uso (no root do projeto):
  python scripts/01_extrair_feedbacks.py \
    --input data/parquet/train-sourceAOnly.parquet \
    --output data/feedbacks \
    --min-len 10
"""

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


def normalize_text(s: str) -> str:
    """Normaliza texto para gerar hash (remove espaços repetidos, normaliza quebras de linha)."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def essay_hash_sha256(essay_text: str) -> str:
    """Hash do texto normalizado da redação."""
    norm = normalize_text(essay_text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def extrair_feedback_c5(specific_comment):
    """
    Extrai o feedback da Competência 5 (C5) do campo `specific_comment`.

    - Lista de strings
    - String com padrão "1) ... 2) ... 3) ... 4) ... 5) ..."
    """
    if pd.isna(specific_comment):
        return None

    # Caso string
    if isinstance(specific_comment, str):
        sc = specific_comment.strip()

        # Tentativa 1: lista serializada
        try:
            lista = ast.literal_eval(sc)
            if isinstance(lista, list) and len(lista) >= 5:
                return lista[4]
        except Exception:
            pass

        # Tentativa 2: regex "5) ... (até fim)"
        # Aceita "5)" ou "5 )" etc.
        match = re.search(r"5\)\s*(.*)\$", sc, flags=re.DOTALL)
        if match:
            c5 = match.group(1).strip()
            return c5 if c5 else None

        return None

    # Caso lista
    if isinstance(specific_comment, list):
        if len(specific_comment) >= 5:
            return specific_comment[4]
        return None

    return None


def is_valid_c5(text, min_len: int) -> bool:
    if text is None:
        return False
    if pd.isna(text):
        return False
    return len(str(text).strip()) > min_len


def main():
    parser = argparse.ArgumentParser(description="Extrai feedback humano de C5 e gera hash estável por redação (v7).")
    parser.add_argument("--input", required=True, help="Caminho do parquet de entrada (ex.: parquet/train-sourceAOnly.parquet).")
    parser.add_argument("--output", required=True, help="Diretório de saída (ex.: data/feedbacks).")
    parser.add_argument("--min-len", type=int, default=10, help="Comprimento mínimo (em caracteres) para considerar C5 válido (default: 10).")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {in_path}")

    df = pd.read_parquet(in_path)

    required_cols = ["essay_text", "specific_comment"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no parquet de entrada: {missing}")

    # Construir dataset de saída
    rows = []
    for i, row in df.iterrows():
        essay_text = row.get("essay_text", "")
        c5 = extrair_feedback_c5(row.get("specific_comment"))

        rows.append({
            # rastreabilidade
            "source_file": in_path.name,
            "source_row": int(i),

            # ids/campos do dataset (se existirem)
            "id": row.get("id"),
            "essay_id": row.get("essay_id"),  # pode não existir
            "id_prompt": row.get("id_prompt"),  # pode não existir

            # chave estável
            "essay_hash": essay_hash_sha256(essay_text),

            # texto e contexto
            "essay_text": essay_text,
            "prompt": row.get("prompt"),
            "supporting_text": row.get("supporting_text"),
            "grades": row.get("grades"),

            # alvo
            "feedback_c5": c5,
        })

    df_out = pd.DataFrame(rows)

    # Diagnósticos
    total = len(df_out)
    n_hash_unique = int(df_out["essay_hash"].nunique())
    n_c5_valid = int(df_out["feedback_c5"].apply(lambda x: is_valid_c5(x, args.min_len)).sum())

    # Arquivo 1: extraídos (todos)
    out_extraidos = out_dir / f"feedbacks_extraidos_c5_{in_path.stem}.parquet"
    df_out.to_parquet(out_extraidos, index=False)

    # Arquivo 2: válidos (somente C5 válido)
    mask_valid = df_out["feedback_c5"].apply(lambda x: is_valid_c5(x, args.min_len))
    df_valid = df_out[mask_valid].copy()
    out_validos = out_dir / f"feedbacks_validos_c5_{in_path.stem}.parquet"
    df_valid.to_parquet(out_validos, index=False)

    # Arquivo 3: relatório
    report = {
        "input_file": str(in_path),
        "output_dir": str(out_dir),
        "total_rows_input": int(len(df)),
        "total_rows_output": int(total),
        "essay_hash_unique": n_hash_unique,
        "essay_hash_is_unique_key": bool(n_hash_unique == total),
        "c5_valid_min_len": int(args.min_len),
        "c5_valid_count": n_c5_valid,
        "c5_valid_percent": round((n_c5_valid / total * 100.0) if total else 0.0, 2),
        "outputs": {
            "extraidos": str(out_extraidos),
            "validos": str(out_validos),
        },
        "notes": [
            "essay_hash = SHA-256 do essay_text normalizado (estável entre execuções).",
            "feedback_c5 extraído de specific_comment (lista ou padrão '5) ...').",
        ],
    }

    out_report = out_dir / f"relatorio_extracao_c5_{in_path.stem}.json"
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(" Extração C5 concluída")
    print(f"  Entrada: {in_path} ({len(df)} linhas)")
    print(f"  Saída (extraídos): {out_extraidos} ({len(df_out)} linhas)")
    print(f"  Saída (válidos C5): {out_validos} ({len(df_valid)} linhas)")
    print(f"  Relatório: {out_report}")
    print(f"  essay_hash únicos: {n_hash_unique}/{total}")
    print(f"  C5 válidos (min_len>{args.min_len}): {n_c5_valid}/{total} ({report['c5_valid_percent']}%)")


if __name__ == "__main__":
    main()
