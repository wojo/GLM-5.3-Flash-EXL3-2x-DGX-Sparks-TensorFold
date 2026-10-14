# KV sharding on four Sparks: test branch

**Test branch, not for production.** This is upstream's `tp4` branch plus three patches from open pull requests,
ported so KV sharding can be tried on four Sparks for the first time. Nothing here has run on GPUs yet. Discussion and
results: [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold#152](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/152).

## What is added

| Patch | Pull request | What it does |
|---|---|---|
| `0110-glm-spill-blocks` | #124 | Spill tier v2: kept states written to NVMe in shared blocks when kept, read back instead of a prefill. Off unless `SPILL_GIB` is set. Also turns on `PREFIX_WAIT` (agents sharing a system prompt still prefilling wait for it instead of prefilling it twice; same replies) |
| `0111-glm-chat-continued` | #128 | A conversation keeps its id across a rewind or a resume from disk |
| `0118-glm-kv-shard` | #152 | KV sharding (`TF_GLM_KV_SHARD=1`): each conversation's latent cache stored once, on the Spark that owns it. **Ported to `tp4` with the test changes below** |

With `TF_GLM_KV_SHARD` unset the branch serves like `tp4`, apart from `0110`'s `PREFIX_WAIT` and `0111`.

## The test changes (this branch only)

`0118` has not been ported to three parts of `tp4`. With `TF_GLM_KV_SHARD=1`, `scripts/config.sh` turns them off and
logs one line saying so:

- **Own-row prompt front** (`TF_GLM_PROMPT_OWN_FRONT=0`) and **prompt microbatching** (`TF_GLM_PROMPT_MICROBATCH=0`):
  their DSA prompt block reads the whole latent cache on every rank, not the owner's rows. The engine refuses either
  one set to 1 with the shard. Expect prefill about 11-16% below `tp4`'s defaults (131k: about 2,110 against 2,510
  tok/s by `tp4`'s own measurements); decode is not affected.
- **The carry** (`TF_GLM_KV_SHARD_CARRY=0` at three Sparks or more): it sends rows straight from one part's owner to another's, and on a
  switchless ring opposite Sparks have no link. The engine refuses the carry on a ring (`TF_NCCL_RING=1`). On a switch
  it can be tried with `TF_GLM_KV_SHARD_CARRY=1`.
- **No warm-up without the carry:** with the carry off nothing is sent point to point past two ranks (the prompt
  history goes out by broadcast), so the start-up warm-up of every pair is skipped.

Four ranks are allowed here. The `kv-shard` pull request itself refuses four ranks until this test is done.

CPU checks against this branch's engine: `tools/kv_shard_check.py` 305 pass (four-part placement, eviction and
sizing, and the guard), `tools/spill_blocks_check.py`. Run them in the image as their headers say.

## How to run it

1. Check out this branch on the head Spark. The first start builds the image locally (the patches no longer match
   `tp4`'s published image).
2. Unsharded baseline first: `./start-tp4.sh` (or `./start-tp4-switchless.sh`) as usual.
3. Sharded: add `export TF_GLM_KV_SHARD=1` to `scripts/local.sh` and start again. The start log should show the
   `[config] TF_GLM_KV_SHARD=1 (test branch ...)` line and a `TF_GLM_KV_SHARD: the latent cache once over the 4 ranks: a
   pool of ...` line.

## What to compare (sharded against unsharded, otherwise identical settings)

1. Short prompts at `temperature: 0`, thinking off: full reply texts byte for byte.
2. Then a ~1M-token prompt within the reported context limit.
3. A restart with spill restores (`SPILL_GIB` set, its own `SPILL_DIR`).
4. Pool sizes from the start log, prefill and decode rates, and `/health`'s `kv_shard` section (errors and refusals).

Please report the exact commits and settings with the results on #152.
