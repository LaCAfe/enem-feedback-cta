#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_build_cta.py (v7)

Gera o CTA (Conjunto de Termos dos Avaliadores) usando feedbacks humanos de C5 das redações do AES-ENEM.

Decisões metodológicas:
- Fonte: campo feedback_c5 de feedbacks_validos_c5_train-sourceAOnly.parquet
- N-gramas: 2 a 4 tokens
- Stopwords: removidas nas bordas do n-grama (primeiro e último token)
  - cta_completo: stopwords padrão do português (NLTK)
- Frequência mínima: n-grama presente em >= 3 feedbacks distintos
- Termos no CTA final: 20 mais frequentes por documento

Saídas:
  data/processed/knowledge/cta_completo.json
  data/processed/knowledge/cta_limpo.json
  data/processed/knowledge/relatorio_cta.json

Uso:
  python scripts/02_build_cta.py \
    --input data/feedbacks/feedbacks_validos_c5_train-sourceAOnly.parquet \
    --output data/processed/knowledge \
    --min-doc-freq 3 \
    --top-n 20
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import nltk
import pandas as pd

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# ── Stopwords estendidas ────────────────────────────────────────
# Identificadas por análise de frequência de unigramas nos 167
# feedbacks humanos de C5. Critérios documentados no artigo.
STOPWORDS_EXTRAS = {
    # Critério 1 — onipresença não discriminativa (>= 15%)
    "autor",        # 41.9% — referência estrutural ao escritor
    "parágrafo",    # 18.6% — referência estrutural + artefato de formato
    # Critério 1 com justificativa qualitativa complementar (< 15%)
    "texto",        # 9.6%  — referência estrutural à redação
    "redação",      # 5.4%  — referência estrutural à redação
    "apresenta",    # 4.8%  — verbo estrutural genérico de feedback
    # Critério 2 — especificidade temática
    "governo",      # 6.6%  — temas políticos
    "hino",         # 6.0%  — tema hino nacional
    "ciência",      # 6.0%  — tema educação/ciência
    "cantar",       # 5.4%  — tema hino nacional
    "armas",        # 3.6%  — tema posse de armas
    "posse",        # 3.6%  — tema posse de armas
    # Critério 3 — artefato de formato
    "competência",  # 4.2%  — marcador de competência no formato
    "vermelho",     # 3.6%  — marcação visual do avaliador no texto
}


def limpar_prefixo(texto: str) -> str:
    """Remove apenas o prefixo '5) ' do início do feedback."""
    if pd.isna(texto):
        return ""
    return re.sub(r"^5\)\s*", "", str(texto).strip())


def extrair_ngrams_com_filtro_bordas(
    texto: str,
    stop: set,
    ns: list[int]
) -> list[tuple]:
    """
    Extrai n-gramas de tamanhos ns de um texto.

    Regra de borda: descarta n-gramas cujo primeiro ou último
    token pertença ao conjunto de stopwords (stop).
    Tokens internos não são verificados.
    """
    # Tokenizar sem remover stopwords — necessário para bordas
    tokens = word_tokenize(texto.lower(), language="portuguese")
    tokens = [t for t in tokens if t.isalpha() and len(t) >= 3]

    ngrams = []
    for n in ns:
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i:i + n])
            # Verificar bordas com o conjunto completo de stopwords
            if gram[0] not in stop and gram[-1] not in stop:
                ngrams.append(gram)
    return ngrams


def gerar_cta(
    textos: list[str],
    stop: set,
    min_doc_freq: int,
    top_n: int,
    ns: list[int]
) -> dict:
    """
    Gera o CTA a partir de uma lista de textos.

    Retorna dict com:
      - 'termos': lista dos top_n n-gramas mais frequentes por documento
      - 'stats': estatísticas de geração
    """
    doc_freq: Counter = Counter()
    total_freq: Counter = Counter()

    for texto in textos:
        ngrams_doc = extrair_ngrams_com_filtro_bordas(texto, stop, ns)
        # Frequência por documento: cada n-grama conta 1x por feedback
        doc_freq.update(set(ngrams_doc))
        # Frequência total
        total_freq.update(ngrams_doc)

    # Filtrar por frequência mínima de documento
    candidatos = {
        gram: freq
        for gram, freq in doc_freq.items()
        if freq >= min_doc_freq
    }

    # Ordenar por frequência de documento (decrescente)
    ordenados = sorted(
        candidatos.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Selecionar top_n
    selecionados = ordenados[:top_n]
    termos = [" ".join(gram) for gram, _ in selecionados]

    stats = {
        "total_ngrams_candidatos": len(candidatos),
        "total_ngrams_brutos": sum(total_freq.values()),
        "vocabulario_unico_bruto": len(doc_freq),
        "termos_selecionados": len(termos),
        "top_termos_com_freq": [
            {
                "termo": " ".join(gram),
                "doc_freq": freq,
                "total_freq": total_freq[gram]
            }
            for gram, freq in selecionados
        ],
    }

    return {"termos": termos, "stats": stats}


def main():
    ap = argparse.ArgumentParser(
        description="Gera CTA a partir de feedbacks humanos de C5 (v7)."
    )
    ap.add_argument(
        "--input", required=True,
        help="Parquet com feedback_c5 humano"
    )
    ap.add_argument(
        "--output", required=True,
        help="Diretório de saída"
    )
    ap.add_argument(
        "--min-doc-freq", type=int, default=3,
        help="Frequência mínima de documento (default: 3)"
    )
    ap.add_argument(
        "--top-n", type=int, default=20,
        help="Número de termos no CTA final (default: 20)"
    )
    args = ap.parse_args()

    in_path  = Path(args.input)
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {in_path}")

    # Carregar feedbacks
    df = pd.read_parquet(in_path)
    print(f" Carregado: {in_path} ({len(df)} linhas)")

    if "feedback_c5" not in df.columns:
        raise ValueError("Coluna 'feedback_c5' não encontrada.")

    validos = df["feedback_c5"].notna().sum()
    print(f"   feedback_c5 válidos: {validos} / {len(df)}")

    # Limpar prefixo e filtrar vazios
    textos = df["feedback_c5"].apply(limpar_prefixo).tolist()
    textos = [t for t in textos if t]
    print(f"   Textos após limpeza de prefixo: {len(textos)}")

    # Stopwords
    stop_base  = set(stopwords.words("portuguese"))
    stop_limpo = stop_base | STOPWORDS_EXTRAS

    ns = [2, 3, 4]

    #  CTA_completo 
    print("\n⏳ Gerando CTA_completo (stopwords padrão)...")
    res_completo = gerar_cta(
        textos, stop_base,
        min_doc_freq=args.min_doc_freq,
        top_n=args.top_n,
        ns=ns
    )
    print(f"   Candidatos (freq>={args.min_doc_freq}): "
          f"{res_completo['stats']['total_ngrams_candidatos']}")
    print(f"   Termos selecionados: {res_completo['stats']['termos_selecionados']}")
    print(f"   Termos: {res_completo['termos']}")

    #  CTA_limpo 
    print("\n⏳ Gerando CTA_limpo (stopwords padrão + 13 extras)...")
    res_limpo = gerar_cta(
        textos, stop_limpo,
        min_doc_freq=args.min_doc_freq,
        top_n=args.top_n,
        ns=ns
    )
    print(f"   Candidatos (freq>={args.min_doc_freq}): "
          f"{res_limpo['stats']['total_ngrams_candidatos']}")
    print(f"   Termos selecionados: {res_limpo['stats']['termos_selecionados']}")
    print(f"   Termos: {res_limpo['termos']}")

    #  Salvar
    for nome, resultado in [
        ("cta_completo", res_completo),
        ("cta_limpo",    res_limpo),
    ]:
        path = out_dir / f"{nome}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(resultado["termos"], f, indent=2, ensure_ascii=False)
        print(f"\n Salvo: {path}")

    # Relatório
    relatorio = {
        "input": str(in_path),
        "parametros": {
            "min_doc_freq":      args.min_doc_freq,
            "top_n":             args.top_n,
            "ngram_sizes":       ns,
            "stopwords_extras":  sorted(STOPWORDS_EXTRAS),
            "total_feedbacks":   len(textos),
        },
        "cta_completo": res_completo["stats"],
        "cta_limpo":    res_limpo["stats"],
    }
    relatorio_path = out_dir / "relatorio_cta.json"
    with open(relatorio_path, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False)
    print(f" Relatório: {relatorio_path}")


if __name__ == "__main__":
    main()
