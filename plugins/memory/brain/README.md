# Brain Graph Memory Provider

Typed knowledge graph (`subject —predicate→ object`) backed by `brain` (Postgres + pgvector).
Complements mem0: mem0 recalls by meaning, brain answers **how things connect**.

## Why a provider (not MCP)
MCP tools are opt-in and the model ignores them. MemoryProvider hooks are mandatory:
- `prefetch` — graph recall injected into every turn
- `sync_turn` — every turn distilled into triples by an LLM and written back
- `brain_graph` tool — explicit traversal: recall/neighbors/path/god_nodes/communities/entity/store

## Config ($HERMES_HOME/config.yaml)
```yaml
plugins:
  hermes-brain:
    brain_db: paul_brain
    cli_dir: /home/dietpi/clawd/brain
    auto_store: true
    extract_model: mistral/mistral-large-latest
    recall_limit: 8
```
Enable with `memory.provider: mem0, brain` (requires the multi-provider core patch).
