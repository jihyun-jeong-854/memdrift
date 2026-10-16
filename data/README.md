# data/

Empty on a fresh clone. The benchmark lives on Hugging Face so that the dataset
and the code cannot drift apart:

```bash
python scripts/download_data.py              # everything
python scripts/download_data.py --subset scenarios   # direct-injection setting only
```

Expected layout once populated:

```
data/
├── scenarios/<bias_dimension>.json              # 105 verified entries, 21 per dimension
│                                                #   incl. top_2_critical annotations
├── rubrics/all_params/<bias_dimension>.json     # objective-metric rubric (364 param specs)
├── conversations/<bias_dimension>.json          # + biased_conversation
├── conversations/<bias_dimension>_neutral.json  # + neutral_conversation
└── memories/
    ├── mem0/<bias_dimension>.json               # + {biased,neutral}_retrieved_memory
    ├── mempalace/<bias_dimension>.json
    └── simplemem/<bias_dimension>.json
```

Override the location with `MEMDRIFT_DATA=/abs/path`.

## How the conversations were generated

The biased and neutral conversation sets came from different generators:

| | user turn | assistant turn |
|---|---|---|
| biased | `claude-sonnet-4-5-20250929` | Claude Agent SDK default — not pinned |
| neutral | `gpt-4.1` | `gpt-4.1` |

The prompts differ too — only the biased generator is shown the tool schema,
since the neutral condition has no drift target. This is visible in the data: at identical turn counts, neutral user turns average
222-238 characters against 500-582 for biased, so the memory frameworks
ingest roughly half as much text in the neutral condition.

The biased assistant turn did not pin a model, so that side is not reproducible
exactly as-run. The neutral side is pinned and reproducible.
