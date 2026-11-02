# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export MRT2 for host-synced tempo control (the `tempo-sync` fork of the AUv3 plugin).

Three functions next to the usual `<name>_state.safetensors`:

  <name>.mlxfn                 the streaming step of `export.py` (same inputs, two traces: generate / forced
                               tokens) with one more output: MRT2's own beat / downbeat read-out at temporal block 11
                               of the conditional (positive) CFG stream, as logits [1, 2]. Outputs:
                               [audio, readout, *state]. The read-out is a linear probe on the residual stream after
                               block 11 (it only sees generated tokens: an honest measurement of the music's beat).
  <name>_reprime.mlxfn         the 're-prime': rebuild the temporal self-attention KV ring buffers of every layer and
                               CFG stream from a fixed-length token context in ONE teacher-forced, all-frames pass, as
                               if the model had just heard that audio. Inputs: codes [1, T, 12] (raw 0..1023), then
                               the step's conditioning inputs (positive block, CFG scales, CFG negatives), then *state.
                               Output: *state. The conditioning, the decoder's audio state and the time step are kept,
                               so the audio continues without a reset. (~0.2 s instead of ~1 s for T one-frame steps.)
  <name>_encoder.mlxfn         the SpectroStream encoder for exactly T frames of 48 kHz stereo (T * 1920 samples),
                               to encode the time-stretched context.

plus `<name>_tempo.json` (window, KV layout, read-out layer, T, ...). Run with an MLX version matching the C++ runtime
(the fork's CMake pins MLX v0.31.1).

usage: python -m magenta_rt.mlx.export_tempo --output-name mrt2_base_tempo --readout readout_L11.npz [--frames 150]
"""

import argparse
import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np

import sequence_layers.mlx as sl
from sequence_layers.mlx import export as sl_export

from . import model, system, spectrostream
from .export import _flatten_state, _unflatten_state, _discretize_cfg_token
from .load_weights import load_weights
from magenta_rt import paths
from magenta_rt.config import DRUM_PIANOROLL, MUSICCOCA, PIANOROLL_WITH_ONSETS


def _attention_inner(residual):
  """the (materialized) attention layer inside an attention Residual"""
  return residual.body.layers[1].inner


def fold_readout(path):
  """standardised linear probe (mu, sd, beat_W/b, down_W/b) -> W [D, 2], b [2] on the raw residual stream"""
  R = np.load(path)
  W = np.stack([R['beat_W'], R['down_W']], 1)
  Wf = W / R['sd'][:, None]
  bf = np.array([R['beat_b'], R['down_b']]) - (R['mu'] / R['sd']) @ W
  return Wf.astype(np.float32), bf.astype(np.float32)


def build(model_name, checkpoint=None, bits=None, group_size=None):
  exp = model.get_model_class(model_name)()
  L = exp.decoder_temporal_size.num_layers
  W = 20 * 25 // L
  exp.encoder_max_past_horizon = W
  exp.decoder_temporal_self_attention_max_past_horizon = W
  exp.decoder_temporal_cross_attention_max_past_horizon = W
  rvq = exp.spectrostream.rvq_truncation_level
  cfg = system.MagentaRT2Sampler.Config(
      depthformer=exp.depthformer_config(num_active_codebooks=None),
      spectrostream=spectrostream.stft_spectrostream_40ms_generic_48khz_stereo_config(
          rvq_truncation_level=rvq, use_unique_codes=False),
      int16_outputs=False)
  s = cfg.make()
  ckpt = paths.resolve_checkpoint(checkpoint or system._CHECKPOINT_REGISTRY[model_name])
  print(f'Loading checkpoint: {ckpt}')
  load_weights(s, ckpt, num_input_channels=exp.input_num_channels)
  if bits and bits < 32:
    import mlx.nn as nn
    nn.quantize(s, group_size=group_size or (32 if bits == 4 else 64), bits=bits)
  return exp, s, W, rvq


def _unroll(a, t, W):
  """ring buffer [1, S, W, ...] with time step t -> entries oldest..newest (entry k sits in slot (t - W + k) % W)"""
  idx = mx.remainder(t - W + mx.arange(W), W)
  return mx.take(a, idx, axis=2)


def _check_reprime(streaming_step, reprime, args_for, flat_state, layout, prev_slots, block, scales, negs, T, rvq, W,
                   offs, n_live=30):
  """The parallel re-prime vs. the engine's sequential prefill (one forced step per frame, as
  MLXEngine::prefill_with_token_array) from the same live state and the same tokens: compare the self-attention KV
  ring buffers in time order (layer by layer, all CFG streams) and the previous-frame slots."""
  st = list(flat_state)
  for _ in range(n_live):                                  # some live generation first
    o = streaming_step(*args_for(mx.zeros((1, 0, rvq), dtype=mx.int32), st))
    st = list(o[2:])
  mx.eval(st)
  codes = mx.array(np.random.default_rng(0).integers(0, 1024, (1, T, rvq)).astype(np.int32))
  # sequential: seed the previous-frame slots with token k, one forced step (its KV gets token k); T-1 steps, then
  # seed token T-1 without a step
  seq_st = list(st)
  for k in range(T):
    tok = (codes[:, k:k + 1] + offs).astype(mx.int32)
    for pi in prev_slots:
      seq_st[pi] = mx.broadcast_to(tok[:, None], seq_st[pi].shape).astype(seq_st[pi].dtype)
    if k == T - 1:
      break
    o = streaming_step(*args_for(tok, seq_st))
    seq_st = list(o[2:])
    mx.eval(seq_st)
  par_st = list(reprime(codes, block, *scales, *negs, *st))
  mx.eval(par_st)
  rel = []
  for li, (ki, vi, mi, ti) in enumerate(layout):
    ta, tb = seq_st[ti][0, 0], par_st[ti][0, 0]
    errs = []
    for idx in (ki, vi):
      a = _unroll(seq_st[idx].astype(mx.float32), ta, W)
      b = _unroll(par_st[idx].astype(mx.float32), tb, W)
      errs.append(float(mx.linalg.norm(a - b) / (mx.linalg.norm(a) + 1e-9)))
    rel.append(max(errs))
  print('re-prime check: relative KV error (parallel vs sequential) per layer:', np.round(rel, 4).tolist())
  for pi in prev_slots:
    print(f'  previous-frame slot {pi} equal: {bool(mx.all(seq_st[pi] == par_st[pi]))}')
  # a shuffled-slot control: what a wrong slot order would cost
  ki, vi, mi, ti = layout[len(layout) // 2]
  a = _unroll(seq_st[ki].astype(mx.float32), seq_st[ti][0, 0], W)
  b = _unroll(par_st[ki].astype(mx.float32), par_st[ti][0, 0] + 1, W)
  print('  control (slots off by one):', round(float(mx.linalg.norm(a - b) / mx.linalg.norm(a)), 4))


def main(output_name, model_name=paths.DEFAULT_MODEL_NAME, readout=None, readout_layer=11, frames=150,
         bits=None, quantize_group_size=None, checkpoint=None, output_dir=None, temperature=1.3, top_k=40,
         cfg_musiccoca=3.0, cfg_notes=0.1, cfg_drums=1.0, probe=False, check=False):
  exp, s, W, rvq = build(model_name, checkpoint, bits, quantize_group_size)
  NR = system.NUM_RESERVED_TOKENS + 1                     # + the dropout token (conditioning vocabulary)
  dec = s.depthformer.decoder
  temporal = dec.temporal_body.layers[0]
  blocks = temporal.layers
  n_layers = len(blocks)
  print(f'{model_name}: {n_layers} temporal layers, attention window {W}, T = {frames} frames')

  # ---------------------------------------------------------------- conditioning (as export.py)
  musiccoca = [660, 1016, 295, 206, 857, 841, 391, 857, 619, 70, 401, 22][:exp.input_configs[0].rvq_truncation_level]
  cfg_keys = ['musiccoca', 'notes']

  def cond_block(mask_key=None):
    toks = []
    for c in exp.input_configs:
      if 'cfg' in c.key:
        continue
      if c.key == MUSICCOCA.key:
        toks.extend([-1] * c.rvq_truncation_level if mask_key == 'musiccoca' else list(musiccoca))
      else:
        toks.extend([-1] * c.rvq_truncation_level)
    return mx.array((np.array(toks, np.int32) + NR).reshape(1, 1, -1))

  block = cond_block()
  negs = {k: cond_block(k) for k in cfg_keys}
  t_val, k_val = mx.array([temperature]), mx.array([top_k], dtype=mx.int32)
  scales = []
  for c in exp.input_configs:
    if c.key == MUSICCOCA.key:
      scales.append(mx.array([cfg_musiccoca]))
    elif c.key == PIANOROLL_WITH_ONSETS.key:
      scales.append(mx.array([cfg_notes]))
    elif c.key == DRUM_PIANOROLL.key:
      scales.append(mx.array([cfg_drums]))
  num_scales, num_negs = len(scales), len(cfg_keys)
  has_cfg_tokens = any('cfg' in c.key for c in exp.input_configs)

  def seq(v):
    return sl.Sequence(v, mx.ones(v.shape[:2], dtype=mx.bool_))

  constants = {'temperature': t_val, 'top_k': k_val}
  for i, key in enumerate(cfg_keys):
    constants[f'classifier_free_guidance_scale_{key}'] = scales[0] if key == 'musiccoca' else scales[1]
    constants[f'classifier_free_guidance_negative_{key}'] = seq(negs[key])
  spec = sl.ChannelSpec(shape=(exp.input_num_channels,), dtype=mx.int32)
  state = s.get_initial_state(1, spec, constants=constants, training=False)
  flat_state, structure = _flatten_state(state)

  def with_cfg_tokens(x_values, negs_args, scales_args):
    if not has_cfg_tokens:
      return x_values, negs_args
    cfg_tokens = mx.concatenate([
        _discretize_cfg_token(scales_args[0], 0.2, 40, NR),
        _discretize_cfg_token(scales_args[1], 0.2, 40, NR),
        _discretize_cfg_token(scales_args[2], 1.0, 8, NR),
    ], axis=-1).reshape(1, 1, 3)
    return (mx.concatenate([x_values, cfg_tokens], axis=-1),
            [mx.concatenate([n, cfg_tokens], axis=-1) for n in negs_args])

  # ---------------------------------------------------------------- read-out at block `readout_layer`
  Wr = br = None
  if readout:
    Wf, bf = fold_readout(readout)
    Wr, br = mx.array(Wf), mx.array(bf)
  live = {}
  rblk = blocks[readout_layer]
  orig_step = rblk.step_with_emits

  def step_capture(x, st_, *a, **kw):
    y, s2, e = orig_step(x, st_, *a, **kw)
    live['h'] = y.values
    return y, s2, e
  object.__setattr__(rblk, 'step_with_emits', step_capture)

  def streaming_step(x_values, temperature_arg, top_k_arg, *flat_args):
    scales_args = list(flat_args[:num_scales])
    negs_args = list(flat_args[num_scales:num_scales + num_negs])
    forced_tokens = flat_args[num_scales + num_negs]
    state_flat = flat_args[num_scales + num_negs + 1:]
    x_values, negs_args = with_cfg_tokens(x_values, negs_args, scales_args)
    st_ = _unflatten_state(list(state_flat), structure)
    dyn = {'temperature': temperature_arg, 'top_k': top_k_arg}
    for i, key in enumerate(cfg_keys):
      dyn[f'classifier_free_guidance_scale_{key}'] = scales_args[i]
      dyn[f'classifier_free_guidance_negative_{key}'] = seq(negs_args[i])
    y, new_state, _ = s.step_with_emits(x=seq(x_values), state=st_, constants=dyn, forced_tokens=forced_tokens,
                                        training=False)
    new_flat, _ = _flatten_state(new_state)
    y = mx.swapaxes(y.values, -2, -1)
    y = mx.reshape(mx.flatten(y), (1, 2, -1))
    h = live['h']                                          # [streams, 1, D]; stream 0 = the positive conditioning
    h0 = mx.reshape(h, (-1, h.shape[-1]))[0:1].astype(mx.float32)
    ro = (h0 @ Wr + br) if Wr is not None else mx.zeros((1, 2), mx.float32)
    return (y, ro, *new_flat)

  def args_for(forced, st_list):
    return [block, t_val, k_val, *scales, *[negs[k] for k in cfg_keys], forced, *st_list]

  empty_forced = mx.zeros((1, 0, rvq), dtype=mx.int32)
  for _ in range(5):                                       # warm up (also fixes the traced dtypes)
    y, ro, *flat_state = streaming_step(*args_for(empty_forced, flat_state))
    mx.eval(y, ro, *flat_state)
  print('read-out stream layout:', live['h'].shape)

  # ---------------------------------------------------------------- locate the self-attention KV ring buffers
  # (forced tokens replace a step's *output*, which is the next step's input: two steps from the same state, the
  # first forced to different tokens, the second to the same -- only the self-attention buffers (and the
  # previous-frame slots) differ; the cross-attention buffers hold the conditioning)
  outs = []
  for code in (5, 700):
    f = mx.full((1, 1, rvq), code, dtype=mx.int32)
    o = streaming_step(*args_for(f, list(flat_state)))
    o = streaming_step(*args_for(mx.full((1, 1, rvq), 300, dtype=mx.int32), list(o[2:])))
    mx.eval(o)
    outs.append(o[2:])
  layout = []
  for i in range(len(flat_state) - 3):
    a = flat_state[i]
    if a.ndim == 5 and a.shape[2] == W and flat_state[i + 2].ndim == 3 and flat_state[i + 3].ndim == 2:
      if float(mx.abs(outs[0][i].astype(mx.float32) - outs[1][i].astype(mx.float32)).max()) > 0:
        layout.append((i, i + 1, i + 2, i + 3))
  assert len(layout) == n_layers, f'found {len(layout)} self-attention buffers, expected {n_layers}'
  prev_slots = [i for i, a in enumerate(flat_state) if a.ndim == 4 and a.shape[0] == 1 and a.shape[2] == 1
                and a.shape[3] == rvq]
  n_streams = flat_state[layout[0][0]].shape[1]
  print(f'KV layout: {len(layout)} layers (first {layout[0]}), streams {n_streams}, previous-frame slots {prev_slots}')
  if probe:
    return

  # ---------------------------------------------------------------- the re-prime (parallel teacher-forced pass)
  caps = {}
  patched = []
  for li, blk in enumerate(blocks):
    inner = _attention_inner(blk.layers[0])
    orig = inner._compute_attention

    def cap(queries, keys, values, mask, *a, _i=li, _orig=orig, **kw):
      if live.get('capture'):
        caps[_i] = (keys[:, -W:], values[:, -W:])
      return _orig(queries, keys, values, mask, *a, **kw)
    object.__setattr__(inner, '_compute_attention', cap)
    patched.append((inner, orig))
  enc_body = s.depthformer.encoder.body
  num_reserved_dec = system.NUM_RESERVED_TOKENS             # previous-frame / decoder embedding offset (no dropout)
  vocab = 1024
  offs = (mx.arange(rvq) * vocab + num_reserved_dec).astype(mx.int32)

  def reprime(codes, x_values, *flat_args):
    scales_args = list(flat_args[:num_scales])
    negs_args = list(flat_args[num_scales:num_scales + num_negs])
    state_flat = list(flat_args[num_scales + num_negs:])
    x_values, negs_args = with_cfg_tokens(x_values, negs_args, scales_args)
    T = codes.shape[1]
    conds = mx.concatenate([x_values, *negs_args], axis=0)                     # [streams, 1, C]
    conds = mx.broadcast_to(conds, (conds.shape[0], T, conds.shape[2]))
    source = enc_body.layer(seq(conds))
    uniq = codes.astype(mx.int32) + offs                                        # decoder ids
    uniq = mx.broadcast_to(uniq, (conds.shape[0], T, rvq))
    x = seq(uniq).pad_time(pad_left=1, pad_right=0, valid=True, pad_value=dec.config.sos_id)
    emb = dec.embedder.layer(x)
    tin = emb.apply_values(lambda v: mx.mean(v.astype(mx.float32), axis=-2).astype(v.dtype))[:, :-1]
    live['capture'] = True
    try:
      dec.temporal_body.layer(tin, training=False, constants={'source': source})
    finally:
      live['capture'] = False
    out = list(state_flat)
    t = state_flat[layout[0][3]][0, 0]                                          # time step (all streams equal)
    slots = mx.remainder(t - W + mx.arange(W), W)
    order = mx.argsort(slots)                                                   # slot s holds context frame order[s]
    for li, (ki, vi, mi, ti) in enumerate(layout):
      K, V = caps[li]                                                           # [streams, W, H, d]
      Kb = mx.take(K, order, axis=1)[None].astype(state_flat[ki].dtype)
      Vb = mx.take(V, order, axis=1)[None].astype(state_flat[vi].dtype)
      out[ki], out[vi] = Kb, Vb
      out[mi] = mx.ones(state_flat[mi].shape, dtype=state_flat[mi].dtype)
    last = (codes[:, -1:, :].astype(mx.int32) + offs)                           # [1, 1, rvq]
    for pi in prev_slots:
      shp = state_flat[pi].shape
      out[pi] = mx.broadcast_to(last[:, None], shp).astype(state_flat[pi].dtype)
    return tuple(out)

  if check:
    _check_reprime(streaming_step, reprime, args_for, flat_state, layout, prev_slots, block, scales,
                   [negs[k] for k in cfg_keys], frames, rvq, W, offs)
    if check == 'only':
      return

  # ---------------------------------------------------------------- the encoder for exactly T frames
  ss = s.spectrostream

  def encode(waveform_values):
    codes = ss.waveform_to_codes_layer.layer(seq(waveform_values))
    return codes.values[..., :rvq].astype(mx.int32)       # [1, T, rvq] raw codes, ready for the re-prime

  # ---------------------------------------------------------------- export
  out_dir = Path(output_dir or paths.models_dir()).resolve() / output_name
  os.makedirs(out_dir, exist_ok=True)
  step_path, rep_path = out_dir / f'{output_name}.mlxfn', out_dir / f'{output_name}_reprime.mlxfn'
  enc_path, state_path = out_dir / f'{output_name}_encoder.mlxfn', out_dir / f'{output_name}_state.safetensors'
  print('Exporting the step (+ read-out)...')
  with mx.exporter(str(step_path), streaming_step, shapeless=False) as ex:
    ex(*args_for(empty_forced, flat_state))
    ex(*args_for(mx.zeros((1, 1, rvq), dtype=mx.int32), flat_state))
  print('Exporting the re-prime...')
  codes0 = mx.zeros((1, frames, rvq), dtype=mx.int32)
  rep_args = [codes0, block, *scales, *[negs[k] for k in cfg_keys], *flat_state]
  mx.eval(reprime(*rep_args))
  mx.export_function(str(rep_path), reprime, *rep_args)
  print('Exporting the encoder...')
  mx.export_function(str(enc_path), encode, mx.zeros((1, frames * 1920, 2), dtype=mx.float32))
  traced = [a.dtype for a in flat_state]
  fresh, _ = _flatten_state(s.get_initial_state(1, spec, constants=constants, training=False))
  fresh = [a.astype(d) for a, d in zip(fresh, traced)]
  mx.eval(*fresh)
  mx.save_safetensors(str(state_path), {f'state_{i}': a for i, a in enumerate(fresh)})
  meta = {'model': model_name, 'window': W, 'layers': n_layers, 'streams': int(n_streams), 'reprime_frames': frames,
          'readout': bool(readout), 'readout_layer': readout_layer, 'kv_layout': layout, 'previous_frame_slots': prev_slots,
          'outputs': ['audio', 'readout', '*state'], 'mlx': mx.__version__, 'bits': bits}
  (out_dir / f'{output_name}_tempo.json').write_text(json.dumps(meta, indent=1))
  for inner, orig in patched:
    object.__setattr__(inner, '_compute_attention', orig)
  print(f'Exported to {out_dir}')


if __name__ == '__main__':
  ap = argparse.ArgumentParser('export_tempo')
  ap.add_argument('--output-name', required=True)
  ap.add_argument('--model', default=paths.DEFAULT_MODEL_NAME)
  ap.add_argument('--readout', default=None, help='read-out probe .npz (mu, sd, beat_W/b, down_W/b) for the block')
  ap.add_argument('--readout-layer', default=11, type=int)
  ap.add_argument('--frames', default=150, type=int, help='re-prime context length in 25-Hz frames (150 = 6 s)')
  ap.add_argument('--bits', default=None, type=int)
  ap.add_argument('--quantize-group-size', default=None, type=int)
  ap.add_argument('--checkpoint', default=None)
  ap.add_argument('--output-dir', default=None)
  ap.add_argument('--probe', action='store_true', help='only build, warm up and print the state layout')
  ap.add_argument('--check', choices=['yes', 'only'], default=None,
                  help='compare the parallel re-prime with the sequential prefill (only: and do not export)')
  a = ap.parse_args()
  main(a.output_name, a.model, a.readout, a.readout_layer, a.frames, a.bits, a.quantize_group_size, a.checkpoint,
       a.output_dir, probe=a.probe, check=a.check)
