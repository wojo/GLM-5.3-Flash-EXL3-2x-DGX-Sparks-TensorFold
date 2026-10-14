# KV sharding on four Sparks: test and experiment branches

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

## Experiment branch: `kv-shard-tp4-exp`

**Experimental, after the test branch.** `kv-shard-tp4-exp` is `kv-shard-tp4` plus the three things the test branch
turns off, made to work with the shard. Try it only once the test branch's sharded runs match unsharded ones. Nothing
here has run on GPUs.

| Turned back on | How it works with the shard |
|---|---|
| Own-row prompt front (`TF_GLM_PROMPT_OWN_FRONT`) | Its front is unchanged (the pooled keys it selects over are on every Spark). Its attention goes through the same owner fetch as the normal sharded prompt path (`forward._sharded_attend`), with the tokens from the shared pools. The owner writes the chunk's latents; the other Sparks write them into their copy. A twin stays local, as before |
| Prompt microbatching (`TF_GLM_PROMPT_MICROBATCH`) | The owner fetch is the only collective a prompt chunk runs on the main stream. It now waits for the split's side stream, and the side stream waits for it, so the global communicator's broadcast never runs beside the side stream's exchanges. Half B still reads half A's latents: the owner wrote them first, the other Sparks fetch them |
| The carry on a switchless ring | Carried rows go hop by hop the shorter way round the ring, each hop a two-rank all-gather on the cabled link between neighbours (the same links `tp4`'s own exchanges use), the Spark in between passing each piece on. No warm-up on a ring |

Also: the sharded prompt path now picks the same sparse kernel as the unsharded one, so `TF_GLM_MSA=1` keeps sharded
and unsharded replies byte-equal (the test branch's sharded path ignores `TF_GLM_MSA`: keep it at its default, 0, there).

With `TF_GLM_KV_SHARD=1` the `tp4` defaults stay on. `TF_GLM_KV_SHARD_SAFE=1` brings back the test branch's settings
(own-front, microbatch and, at three Sparks or more, the carry off), for an A/B on the same build.

**Known costs, to measure:**
- Microbatching overlaps less on DSA layers: each half's fetch waits for the other half's chain on the side stream.
  KDA layers are unaffected. Expect less of microbatching's prefill gain than `tp4` measured.
- A relayed carry is not pipelined: a two-hop carry takes about twice a one-hop one.
- The relay's staging buffer (~130 MiB, up to two 64 MiB pieces) stays allocated on every Spark a carry has passed
  through, outside the pool's budget.

**Checks:** `tools/kv_shard_check.py` 336 pass against this branch's engine (CPU), including a simulated four-Spark ring
carry (neighbours, opposite Sparks through a relay, across the ring's seam, small pieces, TP3) and structure checks of
the own-row front and the fetch ordering; `tools/spill_blocks_check.py` passes. One independent code review found no
bugs.

**What to compare,** in this order: sharded against unsharded with own-front and microbatch on, short prompts then
~1M, replies byte-equal; prefill against the test branch's sharded run (`TF_GLM_KV_SHARD_SAFE=1`); then a workload that
makes carries (agents sharing a system prompt, more than the pool holds in one part), watching `/health`'s
`kv_shard.carry` (count, `ms_max`) and the replies.
