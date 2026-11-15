# Evaluation results

#### Answer quality and cost by question type

| template   | n   | numeric_recall   | all_correct   | unsupported_numbers   | citation_coverage   | specialists_ok   | approved   | revisions   | tool_calls   | cost_usd   | latency_s   |
|:-----------|:----|:-----------------|:--------------|:----------------------|:--------------------|:-----------------|:-----------|:------------|:-------------|:-----------|:------------|
| comparison | 6   | 1.00             | 1.00          | 0.00                  | 1.00                | 1.00             | 1.00       | 1.00        | 12.50        | $0.274     | 81.58       |
| drivers    | 6   | 0.83             | 0.83          | 1.50                  | 0.72                | 1.00             | 0.33       | 1.67        | 11.17        | $0.402     | 134.87      |
| growth     | 6   | 1.00             | 1.00          | 0.33                  | 0.61                | 1.00             | 0.83       | 0.67        | 7.33         | $0.114     | 45.97       |
| margin     | 6   | 1.00             | 1.00          | 0.00                  | 1.00                | 1.00             | 1.00       | 0.83        | 3.33         | $0.059     | 26.67       |
| **all**    | 24  | 0.96             | 0.96          | 0.46                  | 0.83                | 1.00             | 0.79       | 1.04        | 8.58         | $0.212     | 72.27       |

#### Critic fault injection: share of corrupted memos rejected

| mutation         | n   | caught by code checks   | caught by code + LLM review   |
|:-----------------|:----|:------------------------|:------------------------------|
| drop_citation    | 19  | 1.00                    | 1.00                          |
| fabricate_number | 19  | 1.00                    | 1.00                          |
| perturb_number   | 18  | 0.94                    | 1.00                          |
| swap_company     | 19  | 0.00                    | 1.00                          |
| unknown_citation | 19  | 1.00                    | 1.00                          |

#### Memory: same questions, cold vs. warm memory

| memory                | tool_calls   | llm_calls   | cost_usd   | latency_s   | numeric_recall   |
|:----------------------|:-------------|:------------|:-----------|:------------|:-----------------|
| cold (empty memory)   | 4.83         | 9.33        | $0.094     | 39.97       | 1.00             |
| warm (after one pass) | 4.00         | 8.67        | $0.065     | 30.80       | 1.00             |

