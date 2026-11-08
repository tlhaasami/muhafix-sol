# Five-minute live demo script (blueprint 9.2)

**Before the demo.** Use the pre-tested laptop, which works offline because the model files are in
`models/`. Run `streamlit run app/app.py` and open http://localhost:8501. Keep these tabs ready:
`?sample=01_clean_token.sol&scan=1`, `?sample=02_reentrancy_bank.sol&scan=1` and the *How good is the model?*
tab. Have the failure-case file ready to upload: copy `bad_randomness/old_blockhash.sol` from the SmartBugs
Curated repository (commit 230e649) to the desktop. If the laptop fails, `MUHAFIZ_MOCK=1 streamlit run app/app.py`
shows the same interface with fixed example values. Say clearly that this is the fallback.

| Time | Who | What to do | What to say |
|---|---|---|---|
| 0:00-0:30 | Talha | Title slide, then the app home page | "Smart contracts hold real money and cannot be patched after deployment. Pakistan is building a licensed virtual-asset sector under PVARA, and full audits are slow and expensive for students and small teams. (We claim no link with PVARA; it is context only.) Muhafiz.sol is a free first-pass triage tool. It reads Solidity source and says which of seven weakness types are likely and where to look." |
| 0:30-1:30 | Shaiman | Sidebar: *Clean token*, then **Scan** | "A standard token contract. The risk badge is low and every bar stays below its threshold mark, and no line is highlighted because no line changes the score. Low risk is not a certificate: it means nothing crossed the alert level." Open *How to read the result*. |
| 1:30-2:45 | Shaiman | Sidebar: *Reentrancy bank*, then **Scan**. Choose *Reentrancy* in "Explain this weakness type" | "High risk. The top highlighted line is the external call inside `withdraw`: the money is sent before the balance is set to zero, the classic reentrancy pattern. The highlight comes from occlusion: we blank out each function and line and measure how much the score drops." Point at the similar training contracts and the **Download report** button. Optional: *Tricky renamed reentrancy* shows the same bug is still caught when the names are changed. |
| 2:45-3:45 | Sohaib | *How good is the model?* tab | "The test numbers come from contracts kept out of training, with near-copies kept apart. The prior baseline shows how much is due to class imbalance. The random-versus-cluster table shows why we kept near-duplicates apart: with a random split, cloned contracts leak into the test set and every model looks better than it is. On SmartBugs, an independent human-annotated set, recall is lower, and we found that some SmartBugs contracts also occur in our training data, so we report the clean subset separately. That is the honest number." |
| 3:45-4:30 | Talha | Upload `old_blockhash.sol`, then **Scan**; choose *Bad randomness* | "A failure case from our error analysis. The bug is `blockhash()` of an old block, which returns zero after 256 blocks. The model gives it almost zero, because it learned the common block-timestamp pattern, not what the code means. Notice the explanation still ranks the right line first: it shows where the model looked, not a verdict. That is why this is a triage aid and not an audit." |
| 4:30-5:00 | Talha | Disclaimer footer, then the conclusion slide | "Defensive only: no exploit generation. Labels come from Slither, so its mistakes can propagate, which is why we test on expert labels too. Thank you." |

**Questions to expect.** Why PR-AUC instead of accuracy? Why TF-IDF and not a code language model?
How are the thresholds chosen (F2 on validation, with a precision guard)? What does the sensitivity
slider change? Why do some SmartBugs contracts overlap the training data, and what changes without them?
Short answers are in report chapters 4, 5 and 7.
