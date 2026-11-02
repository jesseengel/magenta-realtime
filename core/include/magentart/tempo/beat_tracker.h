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

/// @file beat_tracker.h
/// @brief Causal beat / downbeat / tempo / meter tracking of MRT2's own music.
///
/// A bar-pointer dynamic Bayesian network (after Krebs, Böck & Widmer 2015, as
/// in madmom's DBNDownBeatTracker) with meter switching and sudden tempo-jump
/// transitions, run as a causal forward filter at 100 fps on MRT2's per-frame
/// beat / downbeat read-out (25 fps, the tempo export's second output).
///
/// State = (meter template j, beat period P in DBN frames, position within the
/// bar). Within a beat the position advances deterministically; at beat
/// boundaries the period may change (Laplace prior on the tempo ratio plus a
/// small uniform jump floor); at bar boundaries the meter may switch.
///
/// Port of `rhythm/rhythmlib/dbn.py` (BarPointer forward filter +
/// OnlineTracker) of the mrt2-mech-interp project; same parameters.

#include <cstdint>
#include <vector>

namespace magentart {
namespace tempo {

struct BarPointerConfig {
  std::vector<int> beats_per_bar{3, 4};
  double min_bpm = 55.0;
  double max_bpm = 200.0;
  int fps = 100;
  double obs_lambda = 16.0;
  double tempo_lambda = 100.0;
  double jump_prob = 1e-4;
  double meter_switch_prob = 1e-4;
  double obs_width = 0.04;  ///< beat region: a fixed duration (s)
};

class BarPointer {
public:
  explicit BarPointer(const BarPointerConfig &cfg = BarPointerConfig{});

  int fps() const { return fps_; }
  int n_states() const { return n_; }
  int n_meters() const { return static_cast<int>(bpb_.size()); }
  int n_tempi() const { return static_cast<int>(periods_.size()); }
  int bpb(int j) const { return bpb_[j]; }
  int period(int ti) const { return periods_[ti]; }
  int offset(int j, int ti) const { return offsets_[j * n_tempi() + ti]; }
  int meter_of(int s) const { return meter_of_[s]; }
  int tempo_of(int s) const { return tempo_of_[s]; }
  int pos_of(int s) const { return pos_of_[s]; }
  int region(int s) const { return region_[s]; }
  double obs_lambda() const { return obs_lambda_; }

  /// One step of the bar-pointer transition (sum-product): out = T alpha.
  void transition(const std::vector<double> &alpha,
                  std::vector<double> &out) const;

  /// Tempo (BPM) from a tempo marginal: posterior-weighted log-BPM in a
  /// +-half-bin window around its mode.
  double marg_tempo(const std::vector<double> &tm, int half = 2) const;

private:
  int fps_;
  std::vector<int> bpb_, periods_, offsets_;
  int n_ = 0;
  std::vector<int> meter_of_, tempo_of_, pos_of_;
  std::vector<std::uint8_t> region_, is_end_;
  std::vector<double> ttempo_;  ///< [nt x nt] row-normalized
  double obs_lambda_;
  double psw_;
};

/// Tracker output after one 25-fps input frame. Times are in seconds of
/// tracker time (input frames / 25 since the last reset).
struct TrackerFrame {
  std::vector<double> beats;      ///< beats decided in this frame
  std::vector<double> downbeats;  ///< ... of which downbeats
  double tempo = 0;               ///< smoothed tempo marginal (BPM)
  int meter = 4;                  ///< beats per bar (marginal mode)
  double beat_phase = 0;          ///< MAP beat phase [0, 1)
  int beat_in_bar = 0;
  double bar_phase = 0;
  int map_bpb = 4;
  double map_tempo = 120;         ///< MAP state's tempo (BPM)
  double next_beat = 0;           ///< predicted time of the next beat (s)
  double conf = 0;
};

/// Streaming wrapper around BarPointer's forward filter: feed MRT2's 25-fps
/// p(beat), p(downbeat); get beats as they are predicted (each beat is placed
/// where the MAP state at the middle of the previous beat predicted it), plus
/// tempo / meter / phase.
class OnlineBeatTracker {
public:
  explicit OnlineBeatTracker(const BarPointer &dbn, int fps_in = 25);
  void reset();
  TrackerFrame push(double p_beat, double p_down);
  /// Move the bar-position belief by `beats` (negative = back) in every
  /// tempo state, so the tracker stays in step with music whose bar phase a
  /// re-prime just moved.
  void shift_beats(double beats);
  bool started() const { return has_alpha_; }

private:
  const BarPointer &d_;
  int up_;
  std::vector<double> alpha_, scratch_, tm_, tm_smooth_, mm_;
  bool has_alpha_ = false, has_tm_ = false;
  long long t_ = 0;  ///< DBN frames consumed
  bool has_prev_ = false;
  double prev_ = 0;
  bool has_mid_ = false;
  double mid_pred_ = 0;
  bool mid_down_ = false;
  double last_beat_ = -1e9;
};

}  // namespace tempo
}  // namespace magentart

