# Data Cleaning as an RL Game: teaching a model to fix messy CSVs

I didn’t set out to build “yet another RL environment”.

I set out to solve a very boring, very expensive problem: the part of every ML project where your excitement dies slowly while you rename columns, chase nulls, argue with date formats, and discover that “₹1,299” is not a float. Data cleaning is the unglamorous tax we all pay before the real work starts — and we pay it again for every new dataset.

So I tried a different framing:

> What if data cleaning wasn’t a checklist… but a game?  
> A deterministic, replayable game where the agent learns *how* to repair tables — not for one schema, but for whatever weird CSV shows up next.

This is that game: **Data Cleaning Env**, an **OpenEnv** reinforcement-learning environment where an agent gets a dirty table + a target schema, and learns to fix the table by emitting **one JSON action per line**.

---

## The “world” the agent lives in

At every episode reset, the environment hands the agent an observation that looks like what a real data engineer would inspect first:

- current column names
- dtypes (as strings)
- null counts per column
- a small sample of rows
- task id / step count

And then it asks the agent to do something very specific:

> Transform the current table into the target schema (names, dtypes, null policy) using valid cleaning operations.

Episodes are short (capped steps), deterministic (seeded), and graded by a **deterministic grader** — so reward isn’t “vibes”, it’s **measurable schema compliance**.

---

## Action space: tiny JSON spells

The agent doesn’t get to write code. It gets to cast small spells.

One action per step, JSON only:

```json
{"op":"rename_column","params":{"from_col":"OLD","to_col":"NEW"}}
```

Some actions are simple (rename, drop, strip whitespace). Some are the kind of thing that breaks pipelines in the real world (dtype casting, null filling).

Two rules mattered for training stability:

1. **Invalid actions never crash** (they fail safely and the episode continues).
2. The agent is strongly nudged to keep outputs structured (JSONL), because in production you want a policy you can execute, not a paragraph you have to interpret.

---

## Reward: the environment gives feedback like a picky reviewer

The reward is dense and multi-signal. It doesn’t just say “win/lose”.

It answers three questions every step:

- **Name alignment**: did the columns move closer to the target schema?
- **Dtype correctness**: did the types become more correct/usable?
- **Null policy progress**: did missing values get handled according to policy?

I like to think of this as a reviewer who doesn’t wait until the end of your PR to comment. If you rename one column correctly, you get credit immediately. That makes RL much less painful.

---

## Training: where the env stops being a demo and starts being a tool

Training is done using **Hugging Face TRL** with a **GRPO-style** loop (multiple generations per prompt, reward computed by running the environment + grader).

The training notebook is included here:

- `RL_Data_Cleaning_Agent.ipynb`

It uses the environment directly (reset → generate JSON → apply actions → grade) so the reward is always “what the env thinks”, not a separate proxy metric.

---

## The plot twist that made v11 matter

The most annoying bugs are the ones that look like “it’s learning… but it won’t get better.”

In my early runs (v10), training plateaued hard. The model got good at the easy stuff, but it never truly learned the medium/hard behaviors. After a bunch of debugging, the reason was embarrassingly simple:

**I accidentally built the training dataset using only Task 1 prompts.**

So the trainer never saw Task 2/3 distributions during training. Reward maxed out for the easy task, and then the gradients basically told the model: “good job, keep doing that forever.”

In **v11**, I fixed it by ensuring *all tasks are present from the start*, and using a **soft curriculum** only via sampling weights (more easy early, but never exclusive).

That change looks boring in code, but it’s the difference between:

- an agent that memorizes one toy task
- an agent that starts picking up transferable “cleanup habits”

---

## Evidence: real training curves

These are plots from real runs (saved as images in this Space repo).

### v9 curve

![curve_v9](assets/images/curve_v9.png)

### v11 curve

![curve_v11](assets/images/curve_v11.png)

If you skim one thing in this writeup, skim the v11 curve: it’s where the training loop stopped “cheating” and started learning across tasks.

---

## Why I think this environment has real power

If you’ve ever built a “data cleaning pipeline”, you know the trap:

- It works for the dataset you tested.
- It silently fails on the next dataset.

What makes this environment interesting isn’t that it can rename columns.

It’s that the environment **forces** a policy to interact with:

- **unseen schemas**
- **messy real-world formatting**
- **step-by-step constraints**
- **a strict executable action interface**

That combination is exactly what you want if your long-term goal is a model that can:

- clean new CSVs without a human writing a custom script each time
- produce transformations that you can audit, replay, and diff
- slot into GRPO-style fine-tuning pipelines as a *real* reward function

In other words: this environment is a training ground for **tool-using data agents** that behave like disciplined engineers, not chatbots.

---

## Where this can go next (the fun part)

If I had more time, the next upgrades I’d push are:

- **Bigger task suite**: more domains (healthcare billing tables, ecommerce catalogs, event logs), more schema drift.
- **Richer ops**: currency normalization, percent normalization, datetime parsing variants, safe joins, dedupe, outlier handling.
- **Stronger generalization evaluation**: hold out entire domain families, not just seeds.
- **Better execution safety**: even stricter action validation + “safe fallback actions” when uncertain.

The dream is a model that doesn’t just clean a table — it **explains its cleaning plan as a sequence of verified operations**, and then executes it reproducibly.

---

## Links (judges / quick access)

- **Hugging Face Space (env)**: TODO_ADD_SPACE_LINK  
- **Swagger docs**: TODO_ADD_SPACE_DOCS_LINK  
- **Training notebook**: `RL_Data_Cleaning_Agent.ipynb`  
- **Repository**: TODO_ADD_REPO_LINK  

If you’re reading this as a judge: start with `/docs`, then glance at the training curves, then open the notebook to see how reward is computed by stepping the environment and calling the grader.

