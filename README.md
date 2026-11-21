# Estabilidade_Dados_Sinteticos

Código e dados do artigo **"Generating Synthetic Power System Load Margin Datasets Using a
Generative AI Approach"**: geração, completude e validação de dados tabulares sintéticos
de margem de carga (estabilidade de tensão) com CTGAN, comparado a TVAE e cópula gaussiana.

## Estrutura

```
data/
  ieee14/Bus14.csv              base do artigo: 10.000 pontos, 20 medições + LoadMargin (barra 14)
  ieee14/Bus14Contingency.csv   IEEE 14 barras com contingências (ainda não usada)
  ieee118/LPM_Bus17_118.csv     IEEE 118 barras, barra-alvo 17 (53 medições)
  ieee118/LPM_Bus45_118.csv     IEEE 118 barras, barra-alvo 45 (43 medições)
src/sdlm/                       biblioteca: geradores, métricas, imputação, figuras
scripts/                        um script por etapa do artigo (01 a 07)
paper/reference_ieee14.json     valores reportados no manuscrito submetido
results/<base>/                 dados sintéticos, tabelas (CSV), figuras (PDF/PNG) e relatório
run_all.sh                      executa o pipeline completo para uma base
```

## Como reproduzir

```bash
pip install -r requirements.txt
./run_all.sh ieee14            # ~1 h em 2 núcleos de CPU (treino do CTGAN: 1600 épocas)
```

Ou etapa por etapa (todos aceitam `--dataset` e `--models`):

| Script | Seção do artigo | Saída |
|---|---|---|
| `01_train_generators.py` | 4, 4.3, Apêndice A | `synthetic/{ctgan,tvae,copula}.csv`, `tables/ctgan_loss.csv` |
| `02_fidelity.py` | 4.2, 4.3 (Tabelas 1 a 3) | `tables/fidelity_*.csv`, `mdnn.csv`, `model_comparison.csv` |
| `03_tstr.py` | 4.4 (Tabela 4) | `tables/tstr.csv` |
| `04_privacy.py` | 4.5 (Tabela 5) | `tables/dcr.csv` |
| `05_missing_values.py` | 4.1 (Figs. 3 e 4) | `figures/KDE_*_missing.pdf`, `tables/missing_values.csv` |
| `06_figures.py` | Figs. 5 a 8 | `figures/*.pdf` |
| `07_report.py` | — | `REPRODUCTION.md` (artigo × reprodução) |
| `08_cross_system.py` | escalabilidade | `results/cross_system.{csv,tex}` (14 × 118 barras) |

## Protocolo experimental

- **Partição**: `train_test_split(test_size=0.2, random_state=42)` → 8.000 treino / 2.000 teste.
- **Geradores** (SDV): treinados só na partição de treino; 10.000 amostras sintéticas cada.
  - CTGAN: configuração do Apêndice A do artigo (`src/sdlm/config.py`): 1600 épocas, lote 64,
    Adam (2e-4; β = 0,5/0,9; decaimento 1e-6), camadas (256, 256), `pac = 1`, semente 42.
  - TVAE e cópula gaussiana: padrões do SDV (o manuscrito não informa outra configuração).
- **Fidelidade**: KDE gaussiano com regra de Scott em malha uniforme comum de 1000 pontos,
  renormalizado na malha (Apêndice B); HD, KL(empírico‖sintético) e KS por variável, contra a
  base empírica completa. Variáveis constantes e iguais nas duas bases recebem zero.
- **MDNN / DCR**: espaço padronizado com média e desvio da base empírica (21 variáveis);
  baseline real→real por vizinho mais próximo *leave-one-out*.
- **TSTR**: Random Forest (200 árvores, semente 42); erros normalizados pela faixa da margem
  de carga no conjunto de teste.
- **Dados faltantes**: máscara MCAR em 5/10/20/30 % das amostras da variável; média local
  com os vizinhos disponíveis mais próximos de cada lado; no CTGAN, cada entrada faltante
  recebe o valor da mesma variável em um registro sintético, e as observadas são mantidas.

## Variáveis informativas e sistema de 118 barras

No IEEE 118 barras, as magnitudes de tensão nas barras com PMU ficam presas aos valores de
referência (desvio-padrão entre 1e-6 e 1,5e-4 p.u.). Padronizadas, essas colunas dominam o
MDNN e o DCR. Por isso `02_fidelity.py` e `04_privacy.py` gravam cada tabela duas vezes: com
todas as variáveis e só com as informativas (`*_informative.csv`), isto é, com coeficiente de
variação ≥ `config.CV_MIN` = 1e-3. No IEEE 14 barras isso remove apenas V2 e V6 (constantes).

```bash
./run_all.sh ieee118_bus45     # ~2,5 h (CTGAN)
./run_all.sh ieee118_bus17     # ~3,5 h (CTGAN)
python scripts/08_cross_system.py
```

Resumo (variáveis informativas; NRMSE do TSTR em % da faixa da margem de carga):

| Base | Gerador | HD médio | MDNN (base real) | NRMSE | DCR mín. (base real) |
|---|---|---|---|---|---|
| IEEE 14, barra 14 | CTGAN | 0,100 | 2,44 (2,24) | 3,7 | 0,99 (1,08) |
| | cópula | 0,023 | 2,24 | 1,9 | 0,92 |
| IEEE 118, barra 45 | CTGAN | 0,112 | 2,44 (1,93) | 6,2 | 1,36 (0,85) |
| | cópula | 0,018 | 1,93 | 0,4 | 0,77 |
| IEEE 118, barra 17 | CTGAN | 0,158 | 5,11 (3,57) | 11,2 | 3,04 (2,21) |
| | cópula | 0,049 | 3,74 | 1,2 | 2,00 |

## Resultados

Tabelas e figuras em `results/ieee14/`; a comparação com os números do manuscrito está em
[`results/ieee14/REPRODUCTION.md`](results/ieee14/REPRODUCTION.md).
