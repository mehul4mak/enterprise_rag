# RAG Evaluation Report

- questions: **7**  ·  all gates pass: **True**

## Aggregate metrics
| metric | value | gate | pass |
|--------|------:|-----:|:----:|
| behaviour_accuracy | 1.0 | 1.0 | ✅ |
| citation_validity | 1.0 | 1.0 | ✅ |
| facts_accuracy | 1.0 | — |  |
| faithfulness_mean | 1.0 | 0.8 | ✅ |
| answer_relevance_mean | 1.0 | 0.8 | ✅ |
| citation_supported_mean | 1.0 | — |  |

## Per-question
| id | expect | behaviour | faithful | relevance | cite_valid | latency ms |
|----|--------|:---------:|:--------:|:---------:|:----------:|-----------:|
| segments | answer | ✅ | 1.0 | 1.0 | ✅ | 7814.4 |
| total_income_h1 | answer | ✅ | 1.0 | 1.0 | ✅ | 3839.5 |
| ebitda_drivers | answer | ✅ | 1.0 | 1.0 | ✅ | 7033.4 |
| ceo_email | refuse | ✅ | — | — | — | 4502.0 |
| airport_pax_cargo | answer | ✅ | 1.0 | 1.0 | ✅ | 3702.5 |
| share_price | refuse | ✅ | — | — | — | 4028.9 |
| airport_income | answer | ✅ | 1.0 | 1.0 | ✅ | 3954.5 |