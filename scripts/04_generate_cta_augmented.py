#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_generate_cta_augmented.py (v7)

Gera feedback automático para C5 com prompt aumentado pelo CTA.

Decisões metodológicas:
- Prompt semelhante ao baseline, com acréscimo do bloco CTA
- Extração de C5: primeira linha não-vazia após ## Competency 5##
- Fallback: primeira linha não-vazia do output inteiro
- Retry: 2 tentativas com backoff de 2s em caso de falha

Uso:
  python scripts/04_generate_cta_augmented.py \
    --input   data/feedbacks/feedbacks_validos_c5_train-sourceAOnly.parquet \
    --cta     data/processed/knowledge/cta_completo.json \
    --output  data/hibrido/hibrido_c5_cta_completo.parquet \
    --max-terms 10
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import ollama


#  Prompt 
# Estrutura: CONTEXTO + CTA + INSTRUÇÕES
# CONTEXTO e INSTRUÇÕES são semelhantes ao baseline (script 02).
# O bloco CTA é o único acréscimo, inserido na Posição 2.

PROMPT_CONTEXTO = """Atue como um revisor especializado em redações dissertativo-argumentativas do ENEM.

A redação a seguir foi escrita por um estudante do ensino médio, e você deverá revisá-la em apenas 1 sentença (somente para a Competência 5): {essay_text}

Use o texto de apoio a seguir para verificar se a redação está alinhada ao tema: {supporting_text}

"""

PROMPT_CTA = """BASE DE CONHECIMENTO — use os termos abaixo como referência avaliativa para enriquecer sua análise. NÃO os repita mecanicamente.
{cta_block}

"""

PROMPT_INSTRUCOES = """Você deve fornecer feedback apenas para a Competência 5 do ENEM.

Na única sentença, analise a proposta de intervenção apresentada na redação.

Mantenha um tom construtivo, claro e detalhado.

IMPORTANTE: Siga o formato de saída EXATAMENTE como especificado abaixo.
Não adicione qualquer texto adicional, explicações ou formatação fora do formato especificado.
Use APENAS o marcador "## Competency 5##" para apresentar o feedback.

A saída deve ser a seguinte:

## Competency 5##
Sua sentença
"""


def build_prompt(essay_text: str, supporting_text: str,
                 cta_terms: list[str], max_terms: int) -> str:
    """
    Monta o prompt final: CONTEXTO + CTA + INSTRUÇÕES.
    CONTEXTO e INSTRUÇÕES são idênticos ao baseline.
    """
    contexto = PROMPT_CONTEXTO.format(
        essay_text=essay_text,
        supporting_text=supporting_text,
    )
    terms = cta_terms[:max_terms]
    lines = "\n".join([f"• {t}" for t in terms])
    cta_section = PROMPT_CTA.format(cta_block=lines)
    return contexto + cta_section + PROMPT_INSTRUCOES


def extrair_c5_do_output(texto: str) -> str | None:
    """
    Extrai C5 do output bruto do LLM.

    Tentativa 1: primeira linha não-vazia após ## Competency 5##
    Tentativa 2: fallback — primeira linha não-vazia do output inteiro
    """
    if texto is None:
        return None
    s = str(texto).strip()
    if not s or len(s) < 10:
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

    # Fallback: primeira linha não-vazia do output inteiro
    for linha in s.split("\n"):
        linha = linha.strip()
        if len(linha) > 10:
            return linha

    return None


def load_cta(cta_path: Path) -> list[str]:
    """
    Carrega CTA como lista de termos.
    Suporta lista direta ou dict com chave 'termos'.
    """
    with open(cta_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return [str(t).strip() for t in obj if str(t).strip()]

    if isinstance(obj, dict):
        for k in ["termos", "terms", "cta", "data"]:
            if k in obj and isinstance(obj[k], list):
                return [str(t).strip() for t in obj[k] if str(t).strip()]

    raise ValueError(
        f"Formato de CTA não reconhecido em {cta_path}. "
        f"Esperado: lista de strings ou dict com chave 'termos'."
    )


def chamar_ollama_com_retry(
    model_name: str,
    prompt: str,
    options: dict,
    max_attempts: int = 2,
    backoff: float = 2.0,
) -> tuple[str | None, int, str | None]:
    """
    Chama o Ollama com retry em caso de falha.
    Retorna (content, tokens, error_msg).
    """
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",     required=True)
    ap.add_argument("--cta",       required=True)
    ap.add_argument("--output",    required=True)
    ap.add_argument("--max-terms", type=int, default=10)
    ap.add_argument("--limit",     type=int, default=0)
    args = ap.parse_args()

    in_path  = Path(args.input)
    cta_path = Path(args.cta)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {in_path}")
    if not cta_path.exists():
        raise FileNotFoundError(f"CTA não encontrado: {cta_path}")

    df = pd.read_parquet(in_path).copy()
    required = {"essay_hash", "essay_text", "supporting_text"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")

    if args.limit > 0:
        df = df.head(args.limit).copy()

    cta_terms = load_cta(cta_path)
    print(f" CTA carregado: {cta_path.name}")
    print(f"   Termos disponíveis: {len(cta_terms)}")
    print(f"   Termos injetados (max {args.max_terms}): "
          f"{cta_terms[:args.max_terms]}")

    modelos = {
        "mistral_pt": {
            "ollama_name": "cnmoro/mistral_7b_portuguese:q2_K",
            "options": {
                "temperature": 0.7,
                "top_p": 0.95,
                "num_predict": 300,
            },
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

        prompt = build_prompt(
            essay_text, supporting_text,
            cta_terms, args.max_terms,
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
            c5      = extrair_c5_do_output(content)
            success = content is not None and c5 is not None

            results.append({
                "essay_hash":        essay_hash,
                "model":             model_key,
                "condition":         "cta",
                "cta_file":          cta_path.name,
                "cta_terms_count":   min(len(cta_terms), args.max_terms),
                "success":           success,
                "time":              elapsed,
                "tokens":            tokens,
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

    print(f"\n Concluído: {out_path}")
    print(f"   Iniciado:       {started}")
    print(f"   Linhas geradas: {len(df_out)}")
    for model, g in df_out.groupby("model"):
        print(f"   [{model}] "
              f"success: {g['success'].sum()}/{len(g)} | "
              f"c5 válido: {g['c5'].notna().sum()}/{len(g)}")


if __name__ == "__main__":
    main()
