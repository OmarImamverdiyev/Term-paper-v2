# Task2 - Word2Vec Training and Semantic Analysis

- Dataset: `D:\GitHub Repos\NLP-Projects\NLP-Project-3\Corpora\news\content_only.csv`
- Tokenized corpus: `D:\GitHub Repos\NLP-Projects\NLP-Project-3\Task2\output\training_corpus.txt`
- Model vectors file: `D:\GitHub Repos\NLP-Projects\NLP-Project-3\Task2\output\vectors.txt`
- Documents used: **97997**
- Total tokens used: **13480722**
- Vocabulary size (trained vectors): **48780**
- Training runtime: **228.51 seconds**

## Chosen Word2Vec Parameters

| Parameter | Value | Why this value |
|---|---:|---|
| `cbow` | `0` | `0` means Skip-gram, chosen for better semantic quality on synonym-like queries. |
| `size` | `150` | 150 dimensions: enough capacity for semantic relations without overlarge files. |
| `window` | `8` | Wider context (`8`) to capture topical co-occurrence in news text. |
| `negative` | `10` | 10 negative samples for stronger contrastive learning. |
| `hs` | `0` | Hierarchical softmax disabled; negative sampling used instead. |
| `sample` | `0.0001` | Downsampling frequent terms (`1e-4`) to reduce stopword dominance. |
| `iter` | `5` | 5 passes over corpus for stable vectors on this dataset size. |
| `min-count` | `10` | Keep words with frequency >= 10 to remove heavy noise. |
| `threads` | `8` | Uses available CPU parallelism for faster training. |
| `binary` | `0` | Text output (`0`) to make analysis and reproducibility straightforward. |

## Synonym / Similar Word Results (10 Query Words)

Selected query words:

`azərbaycan`, `bakı`, `rusiya`, `türkiyə`, `ukrayna`, `prezident`, `naziri`, `hərbi`, `təhsil`, `nəqliyyat`

- `azərbaycan` -> `respublikası` (0.771), `respublikasının` (0.751), `tokayevi` (0.616), `özbəkistan` (0.609), `surinam` (0.603)
- `bakı` -> `sumqayıt` (0.707), `şəhəri` (0.613), `avtovağzalından` (0.612), `gəncə` (0.611), `xırdalan` (0.607)
- `rusiya` -> `rusiyanın` (0.807), `ukrayna` (0.799), `moskva` (0.793), `vladimir` (0.735), `putin` (0.730)
- `türkiyə` -> `türkiyənin` (0.749), `müxbirinin` (0.678), `rəcəb` (0.673), `tayyib` (0.664), `ərdoğan` (0.663)
- `ukrayna` -> `zelenski` (0.810), `ukraynanın` (0.807), `rusiya` (0.799), `volodimir` (0.796), `zelenskinin` (0.757)
- `prezident` -> `i̇lham` (0.891), `prezidenti` (0.763), `cənab` (0.730), `əliyevin` (0.698), `əliyev` (0.689)
- `naziri` -> `nazirinin` (0.724), `nazir` (0.706), `safadi` (0.695), `ayman` (0.694), `papikyanla` (0.691)
- `hərbi` -> `qüvvələrində` (0.664), `qüvvələrinin` (0.648), `vəzifəlilərə` (0.638), `qulluqçularından` (0.617), `qoşunlarına` (0.611)
- `təhsil` -> `elm` (0.826), `təhsili` (0.757), `məktəbəqədər` (0.725), `müəssisələrinin` (0.717), `müəssisələrində` (0.703)
- `nəqliyyat` -> `yol` (0.700), `tranzit` (0.645), `logistika` (0.629), `dəhlizlərinin` (0.627), `vasitələrinin` (0.624)

### Accuracy Discussion

- Mean top-5 cosine across the 10 query words: **0.7034** (high coherence).
- Interpretation: high-frequency political/geographical words usually returned coherent neighbors;
  lower-frequency or broad topical words produced more mixed results.

## Vector Arithmetic Equations

| Equation | Top prediction | Expected word | Expected rank (top-10) | Status |
|---|---|---|---:|---|
| `prezident - azərbaycan + rusiya` | `putin` | `putin` | 1 | hit |
| `prezident - azərbaycan + ukrayna` | `zelenski` | `zelenski` | 1 | hit |
| `prezident - azərbaycan + abş` | `donald` | `tramp` | 4 | near_hit |
| `bakı - azərbaycan + türkiyə` | `i̇stanbul` | `ankara` | -1 | miss |
| `moskva - rusiya + ukrayna` | `kiyev` | `kiyev` | 1 | hit |
| `futbol - komanda + oyun` | `meydançaları` | `matç` | 9 | miss |

- Equation quality summary: **3 exact hits**, **1 near hits (rank 2-5)**.

## Visible Vector Patterns

| Relation group | Mean cosine of relation vectors |
|---|---:|
| `country_to_capital` | 0.3701 |
| `country_to_leader` | 0.4570 |

- If these mean cosines are clearly above 0, similar relations align in vector space.
- In practice for this corpus, country-capital and country-leader relations typically form partially consistent directions.

## Output Files

- `output/training_corpus.txt`: tokenized corpus used for training.
- `output/vectors.txt`: trained Word2Vec vectors (text format).
- `output/vocab.txt`: vocabulary saved by C word2vec.
- `output/synonyms.tsv`: top-5 similar words for 10 query words.
- `output/vector_equations.tsv`: equation predictions and expected-word ranking.
- `output/relation_patterns.tsv`: pairwise cosine between relation vectors.