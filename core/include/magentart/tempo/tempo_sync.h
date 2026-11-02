// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

/// @file tempo_sync.h
/// @brief Host tempo / downbeat sync for MRT2 by "re-priming" its memory.
///
/// MRT2 has no tempo input: it continues the tempo of what it has heard. A
/// clean tempo change is therefore a *re-prime*: the last 6 s of its own
/// output are time-stretched to the new tempo (Signalsmith Stretch, pitch
/// kept) and its attention memory is rebuilt from them in one parallel pass
/// (`MLXEngine::reprime_from_audio`, ~0.25 s). The stretched audio is never
/// played; MRT2 just continues its own music at the new tempo from the next
/// frame on.
///
/// Downbeat lock: MRT2's own beat / downbeat read-out (an output of the tempo
/// export) drives a causal bar-pointer tracker (beat_tracker.h). When MRT2's
/// bar is off the host's grid, a *phase re-prime* rebuilds its memory from a
/// context that ends where its own music was at the host's current bar
/// position ((-offset mod one bar) back: always a rewind, the music repeats up
/// to one bar), and the tempo goes to the host's in the same step.
///
/// The reference grid is the host DAW's transport (tempo, beat position,
/// downbeat of the current measure), mapped onto the engine's 48 kHz output
/// stream: render call k reads stream sample `cursor_k` while the host is at
/// beat `b_k`; with a reported latency L the host places that output at
/// `b_k - L * tempo / 60` (both cases, with and without PDC). No wall clock.
///
/// SyncConductor is a port of the Director's reprime / downbeat-lock /
/// external-clock logic in the mrt2-mech-interp project
/// (rhythm/conductor/director.py, the "Clean" tempo control + Ableton Link
/// sync), with the same constants. TempoSyncController wires it to an
/// MLXEngine and a host transport.

#include <magentart/tempo/beat_tracker.h>

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace magentart {
namespace core {
class MLXEngine;
}
namespace tempo {

inline constexpr int kSampleRate = 48000;
inline constexpr int kHop = 1920;  ///< samples per MRT2 frame
inline constexpr int kFps = 25;

/// Time-stretch (and optionally transpose) stereo audio to exactly `n_out`
/// samples with Signalsmith Stretch (Geraint Luff, MIT): output length =
/// input / rate, time-aligned. Level-matched to the input (RMS), clipped.
void stretch_stereo(const float *in_L, const float *in_R, int n_in,
                    float *out_L, float *out_R, int n_out,
                    float semitones = 0.0f);

/// Reference beat grid on the output stream's timeline.
struct Grid {
  double bpm = 120.0;
  double t0 = 0.0;  ///< stream time (s) of one of its downbeats
  int bpb = 4;      ///< beats per bar
};

struct SyncEvent {
  int frame = 0;
  std::string kind;        ///< "reprime" (tempo) | "phase" (bar + tempo)
  double tempo = 0;        ///< the tempo it re-primed to
  double ratio = 1;        ///< vs the tempo it was playing at
  double rewind = 0;       ///< frames of rewind (phase re-prime)
  double shift = 0;        ///< beats MRT2 was off the grid (> 0 late)
  double ms = 0;           ///< blocking time of the re-prime
};

struct SyncParams {
  bool lock = true;            ///< keep MRT2's downbeats on the grid's bars
  bool resync = true;          ///< re-prime again when it drifts off
  int steps = 1;               ///< split big changes into this many re-primes
  double prime_gain_db = 6.0;  ///< louder context (MRT2 continues re-encoded audio quieter)
  double min_track_s = 3.0;    ///< tracker warm-up before its tempo is trusted
};

struct SyncFrameInfo {
  int frame = 0;
  float p_beat = 0, p_down = 0;
  std::optional<double> tempo;  ///< MRT2's tempo (median of 1 s, octave-folded)
  double beat_phase = 0;
  int bpb = 4;
  std::vector<double> beats, downs;  ///< tracker times (s)
};

class SyncConductor {
public:
  /// Re-prime the model from exactly ctx_frames * kHop stereo samples.
  using ReprimeFn = std::function<bool(const float *, const float *, int)>;

  explicit SyncConductor(int ctx_frames = 150);
  void reset();
  void set_reprime(ReprimeFn fn) { reprime_fn_ = std::move(fn); }
  SyncParams &params() { return p_; }
  int ctx_frames() const { return ctx_frames_; }

  /// The target tempo (nullopt = free). plan=false: no re-prime.
  void set_target(std::optional<double> bpm, bool plan = true);
  /// The reference grid (nullopt = off). The same grid refreshed (jitter,
  /// whole bars moved) keeps the lock's evidence and any pending correction.
  void set_clock(std::optional<Grid> g);
  /// A new reference tempo / position: set the grid and change MRT2's tempo
  /// with one phase re-prime on its next beat that also lands on the grid's
  /// bar position (or a plain tempo change if its bar is not measurable).
  void retime_to_clock(const Grid &g);
  bool same_grid(const Grid &a, const Grid &b) const;
  std::optional<Grid> clock() const { return clock_; }

  /// Before generating frame f: plans and executes re-primes. can_block:
  /// there is enough buffered audio to block for a re-prime now.
  void before_frame(bool can_block);
  /// After frame f: its audio and MRT2's read-out logits for it.
  SyncFrameInfo after_frame(const float *L, const float *R, float beat_logit,
                            float down_logit);

  int frame() const { return f_; }
  std::optional<double> measured() const;           ///< MRT2's tempo now
  std::optional<double> target() const { return target_; }
  std::optional<double> bar_offset();                ///< beats off the grid (> 0 late)
  bool busy() const { return pending_phase_.has_value() || !plan_steps_.empty(); }
  int reprime_count() const { return reprimes_; }
  std::vector<SyncEvent> pop_events();

private:
  struct Seg {
    int frame;
    double tempo;
  };
  // --- the Steer (honest tracker of MRT2's own beat) ---
  std::optional<double> steer_measured() const;
  double music_phase_next() const;
  // --- Director port ---
  std::optional<double> beat_tempo(double window_s = 6.0, int min_beats = 6) const;
  void plan_reprime(double target);
  bool on_beat();
  bool reprime(double target, double rewind_s, std::optional<double> measured,
               const char *kind, double shift);
  std::optional<std::pair<int, bool>> bar_fold(double a, double per, int bpb,
                                               double lo) const;
  std::optional<int> which_bar(const std::vector<int> &ks,
                               const std::optional<std::pair<int, bool>> &fold,
                               int bpb, bool trust) const;
  std::optional<double> phase_error(const Grid &g, bool trust = true) const;
  std::optional<double> phase_error_at(const Grid &g, double t) const;
  void lock_check();
  bool phase_reprime(const Grid &g, bool whole, bool set_tgt, bool at);
  /// MRT2's beat phase vs the grid (beats, -0.5..0.5, > 0 late) from the
  /// circular mean of its recent heard beats, or nullopt.
  std::optional<double> beat_phase_error(const Grid &g) const;
  /// MRT2's tempo from the drift of its beat phase against the grid since the
  /// last re-prime: T - 60 * d(lag)/dt. Far more precise than the tracker's
  /// beat-interval estimates (a 1 % error is 0.12 beat in 6 s), or nullopt.
  std::optional<double> drift_tempo() const;
  /// The tempo MRT2 plays at now, best estimate first: drift vs the grid,
  /// beat-time fit, the last re-prime's target (recent), the tracker's.
  std::optional<double> current_tempo() const;
  /// After a re-prime MRT2 plays within a few % of the context's tempo: an
  /// estimate > 8 % off the last re-prime's target (octave-folded) is the
  /// tracker on another metrical level, not a tempo change.
  bool plausible(double bpm) const;
  void frame_audio(int frame, const float *&L, const float *&R) const;

  int ctx_frames_;
  SyncParams p_;
  ReprimeFn reprime_fn_;
  BarPointer dbn_;
  OnlineBeatTracker trk_;
  std::optional<TrackerFrame> last_;
  int f_obs_ = 0;
  std::deque<double> shown_;
  std::optional<double> target_;

  int f_ = 0;
  int ring_frames_;
  std::vector<float> ring_L_, ring_R_;
  std::vector<Seg> segs_;
  std::vector<double> plan_steps_;
  int next_step_f_ = 0;
  std::optional<double> pending_target_, plan_measured_;
  int last_reprime_f_ = -1000000000;
  int bad_ = 0;
  std::optional<int> wait_since_;
  std::deque<double> beat_log_, down_log_;
  std::deque<std::pair<int, float>> pd_log_;
  bool bar_trust_ = false;
  std::optional<Grid> clock_;
  int lock_bad_ = 0;
  std::optional<Grid> pending_phase_;
  bool phase_at_ = false;
  int last_phase_f_ = -1000000000;
  std::deque<std::pair<double, double>> drift_;  ///< (t, unwrapped beat lag) since the last re-prime
  int reprimes_ = 0;
  std::vector<SyncEvent> events_;
};

/// Host transport as seen at one render call, mapped onto the output stream.
struct HostTransport {
  bool valid = false;
  double tempo = 0;           ///< BPM (host beats, usually quarter notes)
  double beat = 0;            ///< host beat the stream sample `cursor` carries
  double bar_start_beat = 0;  ///< host beat of a downbeat (current measure)
  int beats_per_bar = 4;
  std::uint64_t cursor = 0;   ///< stream position (48 kHz samples)
  bool playing = false;
};

struct SyncStatus {
  bool enabled = false;
  bool available = false;   ///< tempo-sync models loaded and read-out present
  bool host_valid = false;
  double host_bpm = 0;
  int host_bpb = 4;
  bool playing = false;
  double mrt_bpm = 0;       ///< 0 = unknown
  bool offset_valid = false;
  double offset_beats = 0;  ///< MRT2's bar vs the host's (> 0 late)
  bool busy = false;        ///< a re-prime is pending
  int reprimes = 0;
  int frame = 0;
  std::string last_event;
};

/// Runs a SyncConductor around an MLXEngine's frames: always tracks MRT2's
/// beat (when the step export has the read-out); when enabled, follows the
/// host transport. All methods except set_* / status() run on the inference
/// thread.
class TempoSyncController {
public:
  explicit TempoSyncController(core::MLXEngine &engine);
  void reset();  ///< a new stream (engine state reset, ring buffers cleared)
  void set_enabled(bool on) { enabled_.store(on, std::memory_order_relaxed); }
  bool enabled() const { return enabled_.load(std::memory_order_relaxed); }
  void set_bar_lock(bool on) { lock_.store(on, std::memory_order_relaxed); }
  bool bar_lock() const { return lock_.load(std::memory_order_relaxed); }
  void set_keep_cfg(bool on) { keep_cfg_.store(on, std::memory_order_relaxed); }
  bool keep_cfg() const { return keep_cfg_.load(std::memory_order_relaxed); }
  bool available() const;

  /// One frame: (follow the host,) maybe re-prime, generate, track.
  bool step(float *L, float *R, const HostTransport &host, bool can_block,
            std::int32_t *tokens_out = nullptr);

  SyncStatus status() const;
  std::vector<SyncEvent> pop_events();
  SyncConductor &conductor() { return cond_; }
  /// Per-frame info of the last frame (inference thread).
  const SyncFrameInfo &last_info() const { return info_; }

private:
  void follow_host(const HostTransport &host);
  core::MLXEngine &engine_;
  SyncConductor cond_;
  std::atomic<bool> enabled_{false}, lock_{true}, keep_cfg_{true};
  bool was_enabled_ = false;
  bool cfg_overridden_ = false;
  SyncFrameInfo info_;
  mutable std::mutex status_mutex_;
  SyncStatus status_;
  std::vector<SyncEvent> pending_events_;
};

}  // namespace tempo
}  // namespace magentart
