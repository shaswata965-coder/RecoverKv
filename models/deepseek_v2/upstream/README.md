# Vendored DeepSeek-V2 remote code (reference oracle)

Copied byte-for-byte from the Hugging Face model repository
`deepseek-ai/DeepSeek-V2-Lite-Chat` at revision
`85864749cd611b4353ce1decdb286193298f64c7` (the base model
`deepseek-ai/DeepSeek-V2-Lite@604d5664dddd88a0433dbae533b7fe9472482de0` ships the
same architecture). `modeling_deepseek.py` and `configuration_deepseek.py` carry
their own Apache-2.0 headers. `config.json` is the Lite-Chat config.

| file | sha256 |
|---|---|
| `modeling_deepseek.py` | `7d8e5221095286eea991137760893fd7ba52727c0b4ebf48ec09e8bc56b45b9c` |
| `configuration_deepseek.py` | `6de5e7445ab490a16482027b131d5bd6a003da9b58a182a11432115fac84ce59` |
| `config.json` | `f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d` |

**Do not edit these files.** They are the oracle the owned MLA forward is pinned
against (`tests/test_deepseek_mla.py`). `tests/test_deepseek_mla.py` checks the
hashes, so an edit fails loudly instead of quietly moving the reference.
