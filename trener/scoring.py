"""Teacher-forced scoring: token log-probabilities in an arbitrary context.

The module computes `log π̄(a_t | prompt, a_{<t})` for the given tokens in the given
context, either without gradient (cross-scoring) or with gradient (optimization passes).
Both modes take **the same numerical path**, so the differences
`Δ^T = ℓ^{T} − ℓ^{T→S}` and `Δ^S = ℓ^{S→T} − ℓ^{S}` are computed comparably.
Both differences are "log-probability in the teacher context minus log-probability in
the student context" (see `deltas`).

Token log-probabilities in a role's **own** context are taken from the optimization pass
(the same snapshot `π̄`, the gradient is detached via `detach`), so without gradient only
passes in a **foreign** context are computed (`FOREIGN_PASSES`).

`lm_head` is applied in chunks of positions: the full logit vector over the 128 000-token
vocabulary is never materialized for the whole sequence (docs/07-usage.md §7.8).

Packing (`max_tokens > 0`): sequences of one group are concatenated into rows without
padding; `position_ids` restart from 0 for each sequence, so attention and the LFM3
convolution window do not cross document boundaries. The unpacked path (`max_tokens = 0`)
remains the reference.

Media (`trener.media`): a sequence may carry images, audio clips and speech frames. A row with
media enters the language model as `inputs_embeds` — the token embeddings with the media in
place of their placeholders; each distinct image or clip of the row is encoded once, with
gradient when its encoder trains. Rows without media keep the `input_ids` path.

Speech answers (LFM3-Omni, `trener.speech`): with `speech=SpeechLosses(...)` the same forward
also scores the speech frames of the completions — the depth decoder predicts each frame from
the hidden state at the position before it — and the loss of every frame is stored per sequence.

A text answer of a model with speech never contains its mode switches and placeholders: with
`text_mode=True` (rollouts) their logits are −inf, exactly as in the decoding that sampled them.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.checkpoint import checkpoint

from .attention import DOCUMENTS_KWARG
from .media import Media, has_media, media_embeds
from .speech import frame_losses

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from .config import RolloutConfig
    from .modeling import Policy
    from .rollout import Trajectory

logger = logging.getLogger(__name__)

#: Number of positions scored per `lm_head` pass (fp32 logits of one chunk —
#: `LOGIT_CHUNK × 128 000 × 4` bytes, ≈0.5 GB at 1024).
LOGIT_CHUNK = 1024

#: Pass names: (token role, scoring context).
PASS_TEACHER_OWN = ("T", "T")
PASS_TEACHER_FOREIGN = ("T", "S")
PASS_STUDENT_OWN = ("S", "S")
PASS_STUDENT_FOREIGN = ("S", "T")

#: Passes without gradient (foreign context) that the method needs for `Δ` (docs/01-jolt.md §1.8).
#: A role's own context is scored in the optimization pass. `grpo` does not use
#: `Δ` and performs no cross-scoring.
FOREIGN_PASSES: dict[str, tuple[tuple[str, str], ...]] = {
    "jolt": (PASS_TEACHER_FOREIGN, PASS_STUDENT_FOREIGN),
    "jolt+": (PASS_TEACHER_FOREIGN, PASS_STUDENT_FOREIGN),
    "opsd": (PASS_STUDENT_FOREIGN,),
    "grpo": (),
}

#: Former name of the pass table.
REQUIRED_PASSES = FOREIGN_PASSES


def required_passes(method: str) -> tuple[tuple[str, str], ...]:
    """Cross-scoring passes (without gradient) that the method needs."""
    try:
        return FOREIGN_PASSES[method]
    except KeyError as exc:
        raise ValueError(f"unknown method {method!r}") from exc


def _base_model(policy: Policy) -> Any:
    """Base model without `lm_head` (returns hidden states)."""
    return getattr(policy.model, "model", policy.model)


def _chunk_logprobs(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    lm_head: Any,
    use_checkpoint: bool,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Log-probabilities of the target tokens for a chunk of positions."""
    if use_checkpoint:
        return checkpoint(
            _chunk_logprobs_impl,
            hidden,
            targets,
            lm_head,
            bias,
            use_reentrant=False,
        )
    return _chunk_logprobs_impl(hidden, targets, lm_head, bias)


def _chunk_logprobs_impl(
    hidden: torch.Tensor, targets: torch.Tensor, lm_head: Any, bias: torch.Tensor | None = None
) -> torch.Tensor:
    # log_softmax(x)[y] = x[y] − logsumexp(x): the full log_softmax tensor is not needed
    logits = lm_head(hidden).float()
    if bias is not None:
        logits = logits + bias
    return logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)


def _flat_logprobs(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    lm_head: Any,
    chunk: int,
    grad: bool,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Log-probabilities of the targets over a flat list of positions (`lm_head` in chunks)."""
    pieces = [
        _chunk_logprobs(
            hidden[offset : offset + chunk], targets[offset : offset + chunk], lm_head, use_checkpoint=grad, bias=bias
        )
        for offset in range(0, targets.shape[0], chunk)
    ]
    return torch.cat(pieces)


@dataclass
class SpeechLosses:
    """Output of the speech-frame scoring of `score_sequences`: the loss of each supervised frame per sequence.

    `values[i]` — `[frames]` losses of the speech frames of the completion of sequence `i` (`None` — no frames);
    `generator` draws the masked codebooks (`trener.speech.frame_masks`).
    """

    values: list[torch.Tensor | None]
    generator: torch.Generator | None = None


def add_stat(stats: dict[str, Any] | None, key: str, value: int | float) -> None:
    """Accumulates a value in an optional statistics dict."""
    if stats is not None:
        stats[key] = stats.get(key, 0) + value


def pack_rows(lengths: Sequence[int], groups: Sequence[Any], max_tokens: int) -> list[list[int]]:
    """Lays out items into rows of capacity `max_tokens` (first-fit decreasing).

    Rows of different groups are not mixed (I6). An item longer than `max_tokens`
    occupies a row by itself. Returns the item indices per row: first all rows of the
    first group in order, then the next group.
    """
    if len(lengths) != len(groups):
        raise ValueError(f"number of lengths ({len(lengths)}) must match the number of groups ({len(groups)})")
    members: dict[Any, list[int]] = {}
    for index, group in enumerate(groups):
        members.setdefault(group, []).append(index)

    rows: list[list[int]] = []
    for indices in members.values():
        indices = sorted(indices, key=lambda index: (-lengths[index], index))
        group_rows: list[list[int]] = []
        free: list[int] = []
        for index in indices:
            size = int(lengths[index])
            for row_number, room in enumerate(free):
                if size <= room:
                    group_rows[row_number].append(index)
                    free[row_number] = room - size
                    break
            else:
                group_rows.append([index])
                free.append(max_tokens - size)
        rows.extend(group_rows)
    return rows


class StreamPacker:
    """Streaming packer: sequential first-fit per group (I6).

    Each group has one open row of capacity `max_tokens`. An item that does not
    fit into the open row closes it and opens a new one; a filled row is closed
    immediately. An item longer than `max_tokens` occupies a row by itself.
    The row composition depends only on the order of addition, not on time.
    """

    def __init__(self, max_tokens: int) -> None:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {max_tokens}")
        self.max_tokens = int(max_tokens)
        self._open: dict[Any, tuple[list[int], int]] = {}

    def add(self, group: Any, index: int, length: int) -> list[tuple[Any, list[int]]]:
        """Adds an item; returns the rows closed by this addition as `[(group, indices)]`."""
        closed: list[tuple[Any, list[int]]] = []
        row, used = self._open.get(group, ([], 0))
        if row and used + int(length) > self.max_tokens:
            closed.append((group, row))
            row, used = [], 0
        row = [*row, int(index)]
        used += int(length)
        if used >= self.max_tokens:
            closed.append((group, row))
            row, used = [], 0
        self._open[group] = (row, used)
        return closed

    def flush(self) -> list[tuple[Any, list[int]]]:
        """Closes all open rows (in the order in which the groups first appeared)."""
        rows = [(group, row) for group, (row, _) in self._open.items() if row]
        self._open.clear()
        return rows


def _model_inputs(policy: Policy, input_ids: torch.Tensor, media: Sequence[Media | None]) -> dict[str, Any]:
    """Inputs of the base model: `input_ids`, or `inputs_embeds` with the media when the rows carry any."""
    if not has_media(media):
        return {"input_ids": input_ids}
    if policy.media is None:
        raise ValueError("the sequences carry media, but the model takes none (text-only model)")
    return {"inputs_embeds": media_embeds(policy.model, policy.media, input_ids, media)}


def _speech_frames(
    policy: Policy, prompt: Sequence[int], completion: Sequence[int], media: Media | None
) -> tuple[list[int], torch.Tensor] | None:
    """Positions whose hidden state predicts each speech frame of the completion, and the frames' codes."""
    if media is None or media.speech is None or policy.media is None:
        return None
    token = policy.media.audio_out_token_id
    tokens = [*prompt, *completion]
    positions = [position for position, value in enumerate(tokens) if value == token]
    if len(positions) != media.frames:
        raise ValueError(f"the sequence has {len(positions)} speech tokens, its media have {media.frames} frames")
    kept = [index for index, position in enumerate(positions) if position >= len(prompt)]
    return [positions[index] - 1 for index in kept], media.speech[kept]


def _score_speech(
    policy: Policy,
    hidden: torch.Tensor,
    latents: list[torch.Tensor],
    codes: list[torch.Tensor],
    owners: list[int],
    speech: SpeechLosses,
    grad: bool,
) -> None:
    """Frame losses of the gathered frames, stored per sequence (`owners[k]` — the sequence of frame group `k`)."""
    if not latents:
        return
    eos = int(policy.model.config.audio_eos_code)
    values = frame_losses(
        policy.model, torch.cat(latents), torch.cat(codes).to(hidden.device), eos, generator=speech.generator
    )
    for owner, value in zip(owners, values.split([code.shape[0] for code in codes]), strict=True):
        speech.values[owner] = value if grad else value.detach()


def _kept_positions(completion: Sequence[int], mask: Sequence[bool] | None) -> list[int]:
    """Completion positions for which logp is computed: all, or only those with `True` in the mask."""
    if mask is None:
        return list(range(len(completion)))
    return [position for position, flag in enumerate(mask) if flag]


def score_sequences(
    policy: Policy,
    items: Sequence[tuple[Sequence[int], Sequence[int]]],
    micro_batch_size: int = 4,
    grad: bool = False,
    *,
    max_tokens: int = 0,
    groups: Sequence[Any] | None = None,
    stats: dict[str, Any] | None = None,
    masks: Sequence[Sequence[bool] | None] | None = None,
    media: Sequence[Media | None] | None = None,
    text_mode: bool = False,
    speech: SpeechLosses | None = None,
) -> list[torch.Tensor]:
    """Computes the log-probabilities of completion tokens for a set of (prompt, completion) pairs.

    Returns one vector per sequence, in the order of `items`.
    `masks` (optional) is one mask per sequence (or `None`): only the completion tokens
    marked `True` go into the result; the others (for example, a user turn in a dialog)
    remain context and their logp is not computed.
    Without `grad`, sequences are grouped into micro-batches by length (less padding);
    with `grad`, the caller sets the split into micro-batches.

    With `max_tokens > 0`, packing is enabled: sequences of one group (`groups`; by default
    all sequences form one group) are concatenated into rows of at most `max_tokens` tokens,
    `position_ids` restart from 0 for each sequence, and one forward pass covers one row
    without padding; `micro_batch_size` is not used when packing. With `grad`, the caller
    must pass a set that fits into one row (sum of lengths ≤ `max_tokens`), otherwise the
    graphs of several rows would be held in memory at the same time.

    `stats` (optional) accumulates `real_tokens` — the tokens of the sequences — and
    `batch_tokens` — the tokens that went through the model (with padding).

    `media` (optional) is one entry per sequence: its images, clips and speech frames (`None` — none).
    `text_mode` — the completions are text answers of the decoding (`Policy.text_bias`): the tokens the
    decoding never samples get −inf. `speech` (optional) receives the losses of the speech frames of the
    completions (`SpeechLosses`).
    """
    if not items:
        return []
    if groups is not None and len(groups) != len(items):
        raise ValueError(f"number of groups ({len(groups)}) must match the number of sequences ({len(items)})")
    if masks is not None:
        if len(masks) != len(items):
            raise ValueError(f"number of masks ({len(masks)}) must match the number of sequences ({len(items)})")
        for (_, completion), mask in zip(items, masks, strict=True):
            if mask is not None and len(mask) != len(completion):
                raise ValueError(f"mask length ({len(mask)}) does not match completion length ({len(completion)})")
    if media is not None and len(media) != len(items):
        raise ValueError(f"number of media entries ({len(media)}) must match the number of sequences ({len(items)})")
    if speech is not None and len(speech.values) != len(items):
        raise ValueError(f"speech losses have {len(speech.values)} slots for {len(items)} sequences")
    entries: Sequence[Media | None] = media if media is not None else [None] * len(items)
    lm_head = policy.model.lm_head
    bias = policy.text_bias(int(lm_head.weight.shape[0])) if text_mode else None
    if max_tokens > 0:
        return _score_packed(policy, items, grad, int(max_tokens), groups, stats, masks, entries, bias, speech)

    device = policy.device
    base = _base_model(policy)
    chunk = max(1, int(LOGIT_CHUNK))

    order = list(range(len(items)))
    if not grad:
        order.sort(key=lambda index: len(items[index][0]) + len(items[index][1]), reverse=True)

    results: list[torch.Tensor] = [torch.empty(0)] * len(items)
    micro = max(1, int(micro_batch_size))
    for start in range(0, len(order), micro):
        indices = order[start : start + micro]
        batch = [items[index] for index in indices]
        kept = [
            _kept_positions(completion, masks[index] if masks else None)
            for index, (_, completion) in zip(indices, batch, strict=True)
        ]
        completion_lengths = [len(positions) for positions in kept]
        max_width = max(len(prompt) + len(completion) for prompt, completion in batch)
        add_stat(stats, "real_tokens", sum(len(prompt) + len(completion) for prompt, completion in batch))
        add_stat(stats, "batch_tokens", len(batch) * max_width)

        input_ids = torch.zeros((len(batch), max_width), dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_width), dtype=torch.long)
        rows: list[int] = []
        columns: list[int] = []
        target_ids: list[int] = []
        frames: list[tuple[int, int, list[int], torch.Tensor]] = []
        for row, (prompt, completion) in enumerate(batch):
            tokens = list(prompt) + list(completion)
            input_ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            attention_mask[row, : len(tokens)] = 1
            # the logit at position `P−1+t` predicts completion token `t`
            rows.extend([row] * len(kept[row]))
            columns.extend(len(prompt) - 1 + position for position in kept[row])
            target_ids.extend(int(completion[position]) for position in kept[row])
            found = _speech_frames(policy, prompt, completion, entries[indices[row]]) if speech is not None else None
            if found is not None:
                frames.append((indices[row], row, found[0], found[1]))
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # The whole computation, including `lm_head`, runs inside the context: otherwise
        # cross-scoring (without gradient) would build a graph and keep the saved
        # fp32 logits over the whole vocabulary.
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            inputs = _model_inputs(policy, input_ids, [entries[index] for index in indices])
            outputs = base(**inputs, attention_mask=attention_mask, use_cache=False)
            hidden_states = outputs.last_hidden_state
            if speech is not None and frames:
                _score_speech(
                    policy,
                    hidden_states,
                    [hidden_states[row, positions] for _, row, positions, _ in frames],
                    [codes for _, _, _, codes in frames],
                    [owner for owner, _, _, _ in frames],
                    speech,
                    grad,
                )
            if target_ids:
                row_index = torch.tensor(rows, dtype=torch.long, device=device)
                column_index = torch.tensor(columns, dtype=torch.long, device=device)
                targets = torch.tensor(target_ids, dtype=torch.long, device=device)
                # completion positions of the whole micro-batch as one flat tensor
                hidden = hidden_states[row_index, column_index]
                flat = _flat_logprobs(hidden, targets, lm_head, chunk, grad, bias)
                values = list((flat if grad else flat.detach()).split(completion_lengths))
            else:
                values = [torch.zeros(0, dtype=torch.float32, device=device) for _ in batch]
        for index, value in zip(indices, values, strict=True):
            results[index] = value
    return results


def _score_packed(
    policy: Policy,
    items: Sequence[tuple[Sequence[int], Sequence[int]]],
    grad: bool,
    max_tokens: int,
    groups: Sequence[Any] | None,
    stats: dict[str, Any] | None,
    masks: Sequence[Sequence[bool] | None] | None = None,
    media: Sequence[Media | None] | None = None,
    bias: torch.Tensor | None = None,
    speech: SpeechLosses | None = None,
) -> list[torch.Tensor]:
    """Packed scoring: one forward pass over one row of sequences from a single group."""
    device = policy.device
    lm_head = policy.model.lm_head
    base = _base_model(policy)
    chunk = max(1, int(LOGIT_CHUNK))

    lengths = [len(prompt) + len(completion) for prompt, completion in items]
    rows = pack_rows(lengths, list(groups) if groups is not None else [0] * len(items), max_tokens)
    if grad and len(rows) > 1:
        raise ValueError(
            f"with grad=True the set must fit into a single row of ≤ {max_tokens} tokens, got {len(rows)} rows"
        )

    results: list[torch.Tensor] = [torch.empty(0)] * len(items)
    for members in rows:
        token_ids: list[int] = []
        position_ids: list[int] = []
        columns: list[int] = []
        target_ids: list[int] = []
        completion_lengths: list[int] = []
        frames: list[tuple[int, list[int], torch.Tensor]] = []
        for index in members:
            prompt, completion = items[index]
            offset = len(token_ids)
            found = _speech_frames(policy, prompt, completion, media[index] if media else None) if speech else None
            if found is not None:
                frames.append((index, [offset + position for position in found[0]], found[1]))
            token_ids.extend(int(token) for token in prompt)
            token_ids.extend(int(token) for token in completion)
            position_ids.extend(range(len(prompt) + len(completion)))
            # the logit at position `offset + P−1+t` predicts completion token `t`
            positions_kept = _kept_positions(completion, masks[index] if masks else None)
            columns.extend(offset + len(prompt) - 1 + position for position in positions_kept)
            target_ids.extend(int(completion[position]) for position in positions_kept)
            completion_lengths.append(len(positions_kept))
        add_stat(stats, "real_tokens", len(token_ids))
        add_stat(stats, "batch_tokens", len(token_ids))

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        positions = torch.tensor([position_ids], dtype=torch.long, device=device)
        documents = tuple(len(items[index][0]) + len(items[index][1]) for index in members)
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            # no `attention_mask`: restarting `position_ids` isolates the documents of the row;
            # document lengths are for per-document attention (`trener.attention`)
            row_media = [media[index] for index in members] if media is not None else []
            inputs = _model_inputs(policy, input_ids, row_media)
            outputs = base(**inputs, position_ids=positions, use_cache=False, **{DOCUMENTS_KWARG: documents})
            hidden_states = outputs.last_hidden_state[0]
            if speech is not None and frames:
                _score_speech(
                    policy,
                    hidden_states,
                    [hidden_states[positions] for _, positions, _ in frames],
                    [codes for _, _, codes in frames],
                    [owner for owner, _, _ in frames],
                    speech,
                    grad,
                )
            if target_ids:
                column_index = torch.tensor(columns, dtype=torch.long, device=device)
                targets = torch.tensor(target_ids, dtype=torch.long, device=device)
                flat = _flat_logprobs(hidden_states[column_index], targets, lm_head, chunk, grad, bias)
                values = list((flat if grad else flat.detach()).split(completion_lengths))
            else:
                values = [torch.zeros(0, dtype=torch.float32, device=device) for _ in members]
        for index, value in zip(members, values, strict=True):
            results[index] = value
    return results


def score_tokens(
    policy: Policy,
    prompt_ids: Sequence[int],
    completion_ids: Sequence[int],
    micro_batch_size: int = 4,
    grad: bool = False,
) -> torch.Tensor:
    """`log π̄(a_t | prompt, a_{<t})` for the completion tokens.

    Without `grad` the result is a constant (cross-scoring). With `grad=True` the
    optimization pass is computed by the same formulas, and the gradient flows only
    through `log π_θ` in the role's own context.
    """
    values = score_sequences(
        policy,
        [(list(prompt_ids), list(completion_ids))],
        micro_batch_size=micro_batch_size,
        grad=grad,
    )
    return values[0]


def score_role_pair(
    policy: Policy,
    trajectories: Sequence[Trajectory],
    other_prompt_ids: Sequence[Sequence[int]],
    cfg: RolloutConfig | None = None,
    grad: bool = False,
) -> list[torch.Tensor]:
    """Scores the tokens of one role in a foreign context (with the trajectories' media, as text answers).

    `other_prompt_ids` holds one prompt per trajectory, in the same order as
    `trajectories` (the context in which the role's tokens are scored).
    """
    if len(trajectories) != len(other_prompt_ids):
        raise ValueError(
            f"number of prompts ({len(other_prompt_ids)}) must match number of trajectories ({len(trajectories)})"
        )
    micro = int(cfg.micro_batch_size) if cfg is not None else 4
    items = [
        (list(other_prompt_ids[index]), list(trajectory.completion_ids))
        for index, trajectory in enumerate(trajectories)
    ]
    media = [trajectory.media for trajectory in trajectories]
    return score_sequences(policy, items, micro_batch_size=micro, grad=grad, media=media, text_mode=True)


def deltas(
    logps_teacher_context: Sequence[torch.Tensor],
    logps_student_context: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    """`Δ_t = ℓ_{teacher context} − ℓ_{student context}` for each trajectory.

    One convention for both roles (docs/04-implementation.md §4.1, docs/01-jolt.md §1.8):

    * teacher: `Δ^T = ℓ^{T} − ℓ^{T→S}` — `deltas(own, foreign)`;
    * student: `Δ^S = ℓ^{S→T} − ℓ^{S}` — `deltas(foreign, own)`.
    """
    if len(logps_teacher_context) != len(logps_student_context):
        raise ValueError("number of log-probability sets must match")
    return [
        teacher.detach() - student.detach()
        for teacher, student in zip(logps_teacher_context, logps_student_context, strict=True)
    ]


__all__ = [
    "FOREIGN_PASSES",
    "LOGIT_CHUNK",
    "PASS_STUDENT_FOREIGN",
    "PASS_STUDENT_OWN",
    "PASS_TEACHER_FOREIGN",
    "PASS_TEACHER_OWN",
    "REQUIRED_PASSES",
    "SpeechLosses",
    "StreamPacker",
    "add_stat",
    "deltas",
    "pack_rows",
    "required_passes",
    "score_role_pair",
    "score_sequences",
    "score_tokens",
]
