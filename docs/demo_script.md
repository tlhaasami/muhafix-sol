# Five-minute live demo script (blueprint 9.2)

**Before the demo.** Use the pre-tested laptop, which works offline because the model files are in
`models/`. Run `streamlit run app/app.py` and open http://localhost:8501. Keep these three tabs ready:
`?sample=01_clean_token`, `?sample=02_reentrancy_bank` and the *How good is the model?* tab. If the
laptop fails, `MUHAFIZ_MOCK=1 streamlit run app/app.py` shows the same interface with fixed example
values. Say clearly that this is the fallback.

| Time | Who | What to do | What to say |
|---|---|---|---|
| 0:00-0:30 | Talha | Title slide, then the app home page | "Smart contracts hold real money and cannot be patched after deployment. Pakistan is building a licensed virtual-asset sector under PVARA, and full audits are slow and expensive for students and small teams. (We claim no link with PVARA; it is context only.) Muhafiz.sol is a free first-pass triage tool. It reads Solidity source and says which of seven weakness types are likely and where to look." |
| 0:30-1:30 | Shaiman | Sidebar: *Clean token*, then **Scan** | "A standard token contract. The risk badge is low and every bar stays below its threshold mark. Low risk is not a certificate: it means nothing crossed the alert level." Open *How to read the result*. |
| 1:30-2:45 | Shaiman | Sidebar: *Reentrancy bank*, then **Scan**. Choose *Reentrancy* in "Explain this weakness type" | "High risk. The highlighted lines are inside `withdraw`: the external call happens before the balance is set to zero, the classic reentrancy pattern. The highlight comes from occlusion: we blank out each function and line and measure how much the probability drops." Point at the similar training contracts and the **Download report** button. |
| 2:45-3:45 | Sohaib | *How good is the model?* tab | "Every number comes from contracts the model never saw. The prior baseline shows how much is due to class imbalance. The random-versus-cluster chart shows why we removed near-duplicates: with a random split, cloned contracts leak into the test set and the scores look better than they really are. On SmartBugs, an independent human-annotated set, recall is lower. That is the honest number." |
| 3:45-4:30 | Talha | Sidebar: *Tricky renamed reentrancy* (or the failure case from `reports/tables/e5_case_studies.json`) | "A failure case. When the variables are renamed and the pattern is split across helper functions, the model is less sure, because it learned from token patterns, not from program semantics. That is why this is a triage aid and not an audit." |
| 4:30-5:00 | Talha | Disclaimer footer, then the conclusion slide | "Defensive only: no exploit generation. Labels come from Slither, so its mistakes can propagate, which is why we test on expert labels too. Thank you." |

**Questions to expect.** Why PR-AUC instead of accuracy? Why TF-IDF and not a code language model?
How are the thresholds chosen (F2 on validation, with a precision guard)? What does the sensitivity
slider change? Short answers are in report chapters 4, 5 and 7.
