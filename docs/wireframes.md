# Wireframes (P3.1)

Layout of blueprint Section 9.1, as built in `app/app.py`. Sketches are text so they live in the repository.

## Scan tab: input

```
+----------------------+----------------------------------------------------------------+
| SIDEBAR              |  [ Scan ]  How good is the model?  Scan many files  About      |
|                      |                                                                |
| Try a sample  [v]    |  Solidity source code                                          |
|  (none)              |  +----------------------------------------------------------+  |
|  Clean token         |  | pragma solidity ^0.8.0;                                  |  |
|  Reentrancy bank     |  | contract Bank { ...                                      |  |
|  ...                 |  |                                                          |  |
|                      |  +----------------------------------------------------------+  |
| Model                |  ...or upload a .sol file   [ Browse ]                         |
|  (o) MLP (primary)   |                                                                |
|  ( ) Logistic reg.   |  [               Scan               ]                          |
|                      |                                                                |
| Sensitivity          |                                                                |
|  0.5 ---o---- 1.5    |                                                                |
|                      |                                                                |
| > How to read        |                                                                |
| > Model card         |                                                                |
+----------------------+----------------------------------------------------------------+
| Footer: triage aid, not a replacement for a professional audit                         |
+----------------------------------------------------------------------------------------+
```

## Scan tab: result (below the input, kept in session state)

```
+----------------------------------------------------------------------------------------+
|  [ HIGH RISK ]  "Reentrancy is above its high-confidence level"     latency 0.3 s      |
|                                                                                        |
|  How confident is the model, per weakness type     (| = alert threshold)              |
|  Reentrancy        ##################|#######  0.91  flagged                           |
|  Unchecked calls   ######|                    0.22                                     |
|  ...                                                                                   |
|                                                                                        |
|  Explain this weakness type [ Reentrancy  v ]                                          |
|  +----------------------------------------------------------------------------------+  |
|  | 12 |   function withdraw() public {                                               |  |
|  | 13 |>>   msg.sender.call{value: amount}("");        <- highlighted, rank 1        |  |
|  | 14 |>>   balances[msg.sender] = 0;                  <- highlighted, rank 2        |  |
|  +----------------------------------------------------------------------------------+  |
|  > Function-level view (risk per function)                                             |
|                                                                                        |
|  Similar training contracts                                                            |
|  [ Name, similarity 87 %, labels, preview ] [ ... ] [ ... ]                            |
|                                                                                        |
|  [ Download report (Markdown) ]  [ Download report (JSON) ]                            |
+----------------------------------------------------------------------------------------+
```

## Other tabs

* **How good is the model?** Headline numbers, the model comparison table (E1), the random vs
  cluster-aware split chart (E2), SmartBugs recall (E3) and the line hit rate (E3b).
* **Scan many files** (F13). Multi-file uploader, then [Scan all] gives a table ranked by risk with a
  CSV download.
* **About & limits.** What the tool does, what it cannot do, the data sources and the disclaimer.
