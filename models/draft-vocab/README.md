# MTP draft vocabulary (lab E29)

`qwen38-gen3-wiki-code-rank.txt`: Qwen3.8-Flash-Next token ids, most frequent first (model outputs on 108 held-out prompts,
then wikitext-2 + code frequency). Production (owner decision, QUESTIONS #6):

    LLAMA_DRAFT_VOCAB=<path>/qwen38-gen3-wiki-code-rank.txt LLAMA_DRAFT_VOCAB_N=49152 llama-server ... --spec-draft-n-max 3

Only the MTP draft head uses the subset; every token is still verified by the full head, so output text is unchanged.
The draft switches back to the full head when recent tokens leave the subset (e.g. non-English text).
