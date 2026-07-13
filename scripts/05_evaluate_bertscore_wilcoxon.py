#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_evaluate_bertscore_wilcoxon.py (v7)

Calcula BERTScore F1 e aplica Wilcoxon signed-rank test
para cada condição CTA vs. baseline, pareado por essay_hash.

Uso:
  python scripts/05_evaluate_bertscore_wilcoxon.py \
    --humano    data/feedbacks/feedbacks_validos_c5_train-sourceAOnly.parquet \
    --baseline  data/baseline/baseline_c5_mistral_llama32.parquet \
    --hibrido   data/hibrido/hibrido_c5_cta_completo.parquet \
    --output    data/resultados/bertscore_cta_completo_10 \
    --label     cta_completo_10
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from bert_score import score as bertscore
from scipy import stats


def carregar_e_validar(path: Path, col_c5: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    assert "essay_hash" in df.columns, f"Coluna essay_hash ausente em {path}"
    assert col_c5 in df.columns, f"Coluna {col_c5} ausente em {path}"
    return df


def montar_pares(df_humano, df_condicao, model_name):
    """
    Para cada essay_hash presente nos dois datasets,
    retorna DataFrame com colunas: essay_hash, humano, gerado.
    Descarta pares com c5 nulo em qualquer lado.
    """
    df_mod = df_condicao[df_condicao["model"] == model_name].copy()

    merged = df_humano[["essay_hash", "feedback_c5"]].merge(
        df_mod[["essay_hash", "c5"]],
        on="essay_hash",
        how="inner",
    )

    merged = merged.rename(columns={"feedback_c5": "humano", "c5": "gerado"})
    merged = merged.dropna(subset=["humano", "gerado"])
    merged = merged[
        (merged["humano"].str.len() > 10) &
        (merged["gerado"].str.len() > 10)
    ]
    return merged.reset_index(drop=True)


def calcular_bertscore(referencias, candidatos):
    _, _, F1 = bertscore(
        candidatos,
        referencias,
        lang="pt",
        model_type="bert-base-multilingual-cased",
        verbose=False,
    )
    return F1.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--humano",   required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--hibrido",  required=True)
    ap.add_argument("--output",   required=True)
    ap.add_argument("--label",    required=True,
                    help="Nome da condição (ex: cta_completo_10)")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_humano   = carregar_e_validar(Path(args.humano),   "feedback_c5")
    df_baseline = carregar_e_validar(Path(args.baseline), "c5")
    df_hibrido  = carregar_e_validar(Path(args.hibrido),  "c5")

    modelos = df_baseline["model"].unique()
    resultados = []

    for model in modelos:
        print(f"\n{'='*60}")
        print(f"MODELO: {model}")
        print(f"{'='*60}")

        # Montar pares baseline
        pares_base = montar_pares(df_humano, df_baseline, model)
        # Montar pares híbrido
        pares_hib  = montar_pares(df_humano, df_hibrido, model)

        # Interseção por essay_hash — apenas pares válidos em ambas condições
        hashes_comuns = set(pares_base["essay_hash"]) & set(pares_hib["essay_hash"])
        pares_base = pares_base[pares_base["essay_hash"].isin(hashes_comuns)]
        pares_hib  = pares_hib[pares_hib["essay_hash"].isin(hashes_comuns)]

        # Ordenar pelo mesmo essay_hash para garantir pareamento
        pares_base = pares_base.sort_values("essay_hash").reset_index(drop=True)
        pares_hib  = pares_hib.sort_values("essay_hash").reset_index(drop=True)

        n = len(pares_base)
        print(f"  Pares válidos: {n}")

        if n < 10:
            print(f"   Pares insuficientes — pulando modelo")
            continue

        # BERTScore baseline
        print(f"  Calculando BERTScore baseline...")
        scores_base = calcular_bertscore(
            pares_base["humano"].tolist(),
            pares_base["gerado"].tolist(),
        )

        # BERTScore híbrido
        print(f"  Calculando BERTScore {args.label}...")
        scores_hib = calcular_bertscore(
            pares_hib["humano"].tolist(),
            pares_hib["gerado"].tolist(),
        )

        # Wilcoxon (unilateral: híbrido > baseline)
        stat, p_value = stats.wilcoxon(
            scores_base,
            scores_hib,
            alternative="less",  # H1: scores_hib > scores_base
        )

        media_base = float(np.mean(scores_base))
        std_base   = float(np.std(scores_base))
        media_hib  = float(np.mean(scores_hib))
        std_hib    = float(np.std(scores_hib))
        diferenca  = media_hib - media_base
        significativo = p_value < 0.05

        print(f"\n  Baseline:   {media_base:.4f} ± {std_base:.4f}")
        print(f"  {args.label}: {media_hib:.4f} ± {std_hib:.4f}")
        print(f"  Diferença:  {diferenca:+.4f}")
        print(f"  W={stat:.2f}  p={p_value:.4f}  "
              f"{'OK - SIGNIFICATIVO' if significativo else 'FALHA - não significativo'}")

        resultado = {
            "modelo":        model,
            "condicao":      args.label,
            "n_pares":       n,
            "baseline_media": round(media_base, 4),
            "baseline_std":   round(std_base, 4),
            "hibrido_media":  round(media_hib, 4),
            "hibrido_std":    round(std_hib, 4),
            "diferenca":      round(diferenca, 4),
            "wilcoxon_W":     round(float(stat), 4),
            "p_value":        round(float(p_value), 4),
            "significativo":  significativo,
        }
        resultados.append(resultado)

        # Salvar scores individuais por par
        df_scores = pares_base[["essay_hash"]].copy()
        df_scores["score_baseline"] = scores_base
        df_scores["score_hibrido"]  = scores_hib
        df_scores["diferenca"]      = scores_hib - scores_base
        csv_path = out_dir / f"scores_{model}.csv"
        df_scores.to_csv(csv_path, index=False)
        print(f"  Scores salvos: {csv_path}")

    # Salvar resultados agregados
    json_path = out_dir / "resultados.json"
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.bool_,)):    return bool(obj)
            return super().default(obj)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    print(f"\n Resultados salvos: {json_path}")

    # Resumo final
    print(f"\n{'='*60}")
    print("RESUMO")
    print(f"{'='*60}")
    print(f"{'Modelo':<15} {'N':>5}  {'Baseline':>10}  "
          f"{'CTA':>10}  {'Δ':>7}  {'p':>7}  Sig")
    print("-" * 65)
    for r in resultados:
        sig = "OK" if r["significativo"] else "FALHA"
        print(f"{r['modelo']:<15} {r['n_pares']:>5}  "
              f"{r['baseline_media']:.4f}±{r['baseline_std']:.4f}  "
              f"{r['hibrido_media']:.4f}±{r['hibrido_std']:.4f}  "
              f"{r['diferenca']:>+.4f}  "
              f"{r['p_value']:>7.4f}  {sig}")


if __name__ == "__main__":
    main()
