#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_generate_baseline.py (v7)

Generates baseline feedback for Competency 5 (C5) using a zero-shot prompt
adapted from Anchieta et al. (2025), translated to Portuguese and focused
on C5 only.

C5 extraction strategy:
  1. Text after marker ## Competency 5## (case-insensitive)
  2. Fallback: full output (prompt requests only C5)

Usage:
  python scripts/03_generate_baseline.py \
    --input  data/feedbacks/feedbacks_validos_c5_train-sourceAOnly.parquet \
    --output data/baseline/baseline_c5_mistral_llama32.parquet
"""

import argparse
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import ollama


PROMPT_C5_PT = """Atue como um revisor especializado em redações dissertativo-argumentativas do ENEM.

A redação a seguir foi escrita por um estudante do ensino médio, e você deverá revisá-la em apenas 1 sentença (somente para a Competência 5): {essay_text}

Use o texto de apoio a seguir para verificar se a redação está alinhada ao tema: {supporting_text}

Você deve fornecer feedback apenas para a Competência 5 do ENEM.

Na única sentença, analise a proposta de intervenção apresentada na redação.

Mantenha um tom construtivo, claro e detalhado.

IMPORTANTE: Siga o formato de saída EXATAMENTE como especificado abaixo.
Não adicione qualquer texto adicional, explicações ou formatação fora do formato especificado.
Use APENAS o marcador "## Competency 5##" para apresentar o feedback.

A saída deve ser a seguinte:

## Competency 5##
Sua sentença
"""


def extrair_c5(texto: str) -> str | None:
    """
    Extracts C5 feedback from raw LLM output.

    Attempt 1: text after marker ## Competency 5##
    Attempt 2: fallback — first non-empty line of full output
    """
    if texto is None:
        return None
    s = str(texto).strip()
    if not s:
        return None

    m = re.search(
        r"##\s*Competency\s*5\s*##\s*(.*)",
        s,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        for linha in m.group(1).split("\n"):
            linha = linha.strip()
            if len(linha) > 10:
                return linha

    # Fallback: first non-empty line
    for linha in s.split("\n"):
        linha = linha.strip()
        if len(linha) > 10:
            return linha

    return None


def chamar_ollama_com_retry(
    model_name: str,
    prompt: str,
    options: dict,
    max_attempts: int = 2,
    backoff: float = 2.0,
) -> tuple[str | None, int, str | None]:
    """Calls Ollama with retry on failure. Returns (content, tokens, error)."""
    last_error = None
    for attempt in range(max_attempts):
        try:
            resp = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options=options,
            )
            return resp["message"]["content"], resp.get("eval_count", 0), None
        except Exception as e:
            last_error = str(e)
            if attempt < max_attempts - 1:
                time.sleep(backoff * (attempt + 1))
    return None, 0, last_error


def main():
    ap = argparse.ArgumentParser(
        description="Generate zero-shot baseline feedback for C5 (v7)."
    )
    ap.add_argument("--input",  required=True, help="Input parquet with essay_hash, essay_text, supporting_text")
    ap.add_argument("--output", required=True, help="Output parquet")
    ap.add_argument("--limit",  type=int, default=0, help="Limit number of essays (0 = all)")
    args = ap.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    df = pd.read_parquet(in_path).copy()

    required = {"essay_hash", "essay_text", "supporting_text"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if args.limit > 0:
        df = df.head(args.limit).copy()

    modelos = {
        "mistral_pt": {
            "ollama_name": "cnmoro/mistral_7b_portuguese:q2_K",
            "options": {"temperature": 0.7, "top_p": 0.95, "num_predict": 300},
        },
        "llama32": {
            "ollama_name": "llama3.2:latest",
            "options": {
                "temperature": 0.8,
                "top_p": 0.95,
                "top_k": 50,
                "num_predict": 300,
            },
        },
    }

    results  = []
    started  = datetime.now().isoformat()
    total    = len(df) * len(modelos)
    contador = 0

    for _, row in df.iterrows():
        essay_hash      = row["essay_hash"]
        essay_text      = str(row["essay_text"])
        supporting_text = str(row.get("supporting_text", ""))

        prompt = PROMPT_C5_PT.format(
            essay_text=essay_text,
            supporting_text=supporting_text,
        )

        for model_key, cfg in modelos.items():
            contador += 1
            t0 = time.time()

            content, tokens, error = chamar_ollama_com_retry(
                model_name=cfg["ollama_name"],
                prompt=prompt,
                options=cfg["options"],
            )

            elapsed = time.time() - t0
            c5      = extrair_c5(content)
            success = content is not None and c5 is not None

            results.append({
                "essay_hash":        essay_hash,
                "model":             model_key,
                "condition":         "baseline",
                "success":           success,
                "time":              elapsed,
                "tokens":            tokens,
                "prompt_version":    "pt_c5_v1",
                "feedback_completo": content,
                "c5":                c5,
                "error":             error,
            })

            if contador % 20 == 0 or contador == total:
                print(f"  [{contador}/{total}] {model_key} | "
                      f"{essay_hash[:16]}... | "
                      f"ok={success} | "
                      f"c5={str(c5)[:60] if c5 else 'None'}")

    df_out = pd.DataFrame(results)
    df_out.to_parquet(out_path, index=False)

    print(f"\n Done: {out_path}")
    print(f"   Started:  {started}")
    print(f"   Rows:     {len(df_out)}")
    for model, g in df_out.groupby("model"):
        print(f"   [{model}] success: {g['success'].sum()}/{len(g)} | "
              f"c5 valid: {g['c5'].notna().sum()}/{len(g)}")


if __name__ == "__main__":
    main()
