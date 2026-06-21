# Block 5 — LangSmith Eval Observations



## Scores

| Evaluator | Score |
|---|---|
| Correctness | 6/6 (100%) |
| Friendliness | 0/6 (0%) |

---

## Answer 1 ) Observation

**Correctness was 100%** across all 6 cases including the ambiguous one. Interestingly, the agent couldn't parse that request and returned an error but the LLM judge still passed it. This shows LLM-as-judge is more lenient than exact match.; it understood the request was vague and gave benefit of the doubt.

**Friendliness was 0%** for all cases. The agent replies with a markdown table for successful orders which looks like a receipt, not a real barista. Short responses like "Sorry, Flat White is currently unavailable." does not show friendly nature. But it can be fixed using proper system prompt

- Correctness and friendliness measure different things. An agent can be technically right but still feel robotic. Both evaluators are needed to get the full picture.

