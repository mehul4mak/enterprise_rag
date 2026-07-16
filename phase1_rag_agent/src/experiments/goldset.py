"""Gold set for retrieval benchmarking — question → page(s) that genuinely contain the answer.

Gold pages were *verified* by locating the answer string in the deck (not guessed). Several facts
legitimately appear on more than one page (e.g. total income is on p2 as a business breakdown and on
p22 as a Y-o-Y waterfall — both correct), so a hit on ANY gold page counts.

This lets us score retrieval with zero LLM calls: recall@k and MRR.
"""

GOLD: list[dict] = [
    {
        "id": "total_income_h1",
        "q": "What is the consolidated total income in H1-26?",
        "gold_pages": [2, 22],  # verified: "44,281" appears on p2 and p22
    },
    {
        "id": "airport_income",
        "q": "What was the total income for Adani Airport Holdings in H1-26?",
        "gold_pages": [3, 17, 22],  # verified: "5,882"
    },
    {
        "id": "pax_volume",
        "q": "What were the passenger volume changes for airports in H1-26?",
        "gold_pages": [16, 32],  # verified: "46.0"
    },
    {
        "id": "cargo_volume",
        "q": "What were the cargo volume changes in H1-26?",
        "gold_pages": [16, 32, 34],  # verified: "5.7"
    },
    {
        "id": "segments",
        "q": "What are the major business segments discussed in the document?",
        "gold_pages": [3, 11],  # verified: "Green H2" / portfolio pages
    },
    {
        "id": "ebitda_drivers",
        "q": "What drivers are mentioned for EBITDA changes in H1-26?",
        "gold_pages": [2, 22, 23],  # verified: EBITDA + IRM
    },
]
