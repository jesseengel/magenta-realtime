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

#include <magentart/tempo/tempo_sync.h>

#include <magentart/mlx_engine.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <set>

#include "signalsmith-stretch/signalsmith-stretch.h"

namespace magentart {
namespace tempo {

namespace {

constexpr double kResyncTol = 0.02;        // re-sync when > 2 % off the target (twice, 1 s apart)
constexpr double kClockResyncTol = 0.012;  // ... 1.2 % with an external clock: tempo drift is phase drift
constexpr double kLockTol = 0.12;          // downbeat lock: re-align when > 0.12 beat off the grid
constexpr double kDecLat = 1960.0 / 48000; // MRT2's audio lags its token frames by one frame + 40 samples
constexpr int kRingSeconds = 16;           // output history: 6 s of context x stretch + a bar of rewind
constexpr double kPi = 3.14159265358979323846;

double octave_near(double bpm, double ref) {
  return bpm * std::pow(2.0, std::round(std::log2(ref / bpm)));
}

double pymod(double a, double m) { return a - m * std::floor(a / m); }
int pymod_i(long long a, int m) { return static_cast<int>(((a % m) + m) % m); }
// Python's round() (half to even) for the integer roundings of the port
long long pyround(double x) { return static_cast<long long>(std::nearbyint(x)); }

double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// least-squares line y = slope * x + icpt
void polyfit1(const std::vector<double> &x, const std::vector<double> &y,
              double &slope, double &icpt) {
  const double n = static_cast<double>(x.size());
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    sx += x[i];
    sy += y[i];
    sxx += x[i] * x[i];
    sxy += x[i] * y[i];
  }
  const double den = n * sxx - sx * sx;
  slope = den != 0 ? (n * sxy - sx * sy) / den : 0.0;
  icpt = (sy - slope * sx) / n;
}

double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-std::clamp(z, -50.0, 50.0))); }

void prime_gain(std::vector<float> &x, double gain_db, double knee = 0.8) {
  const float g = static_cast<float>(std::pow(10.0, gain_db / 20.0));
  const float k = static_cast<float>(knee);
  for (float &v : x) {
    float y = v * g;
    const float a = std::abs(y);
    if (a > k) y = std::copysign(k + (1 - k) * std::tanh((a - k) / (1 - k)), y);
    v = y;
  }
}

}  // namespace

// ─── Signalsmith Stretch ────────────────────────────────────────────────────

void stretch_stereo(const float *in_L, const float *in_R, int n_in,
                    float *out_L, float *out_R, int n_out, float semitones) {
  signalsmith::stretch::SignalsmithStretch<float> st;
  st.presetDefault(2, static_cast<float>(kSampleRate));
  if (semitones != 0.0f)
    st.setTransposeSemitones(semitones, 8000.0f / kSampleRate);
  const float *ins[2] = {in_L, in_R};
  float *outs[2] = {out_L, out_R};
  if (!st.exact(ins, n_in, outs, n_out)) {
    // too short for an exact stretch: plain linear resampling (never hit
    // for the re-prime's multi-second contexts)
    for (int i = 0; i < n_out; ++i) {
      const double x = n_out > 1 ? double(i) * (n_in - 1) / (n_out - 1) : 0.0;
      const int i0 = static_cast<int>(x);
      const int i1 = std::min(i0 + 1, n_in - 1);
      const float w = static_cast<float>(x - i0);
      out_L[i] = in_L[i0] * (1 - w) + in_L[i1] * w;
      out_R[i] = in_R[i0] * (1 - w) + in_R[i1] * w;
    }
  }
  // level-match to the input (RMS) and clip, as the Python Conductor's 'ss'
  double ein = 0, eout = 0;
  for (int i = 0; i < n_in; ++i) ein += double(in_L[i]) * in_L[i] + double(in_R[i]) * in_R[i];
  for (int i = 0; i < n_out; ++i) eout += double(out_L[i]) * out_L[i] + double(out_R[i]) * out_R[i];
  const double rin = std::sqrt(ein / std::max(1, 2 * n_in));
  const double rout = std::sqrt(eout / std::max(1, 2 * n_out));
  const float g = static_cast<float>(rin / (rout + 1e-9));
  for (int i = 0; i < n_out; ++i) {
    out_L[i] = std::clamp(out_L[i] * g, -1.0f, 1.0f);
    out_R[i] = std::clamp(out_R[i] * g, -1.0f, 1.0f);
  }
}

// ─── SyncConductor ──────────────────────────────────────────────────────────

namespace {
BarPointerConfig conductor_dbn_config() {
  BarPointerConfig c;
  c.beats_per_bar = {3, 4};
  c.min_bpm = 55.0;
  c.max_bpm = 200.0;
  c.meter_switch_prob = 1e-4;
  return c;
}
}  // namespace

SyncConductor::SyncConductor(int ctx_frames)
    : ctx_frames_(ctx_frames), dbn_(conductor_dbn_config()), trk_(dbn_),
      ring_frames_(kRingSeconds * kFps) {
  ring_L_.assign(static_cast<size_t>(ring_frames_) * kHop, 0.0f);
  ring_R_.assign(static_cast<size_t>(ring_frames_) * kHop, 0.0f);
  reset();
}

void SyncConductor::reset() {
  trk_.reset();
  last_.reset();
  f_obs_ = 0;
  shown_.clear();
  // target / clock / params are settings and survive a new stream
  f_ = 0;
  segs_.clear();
  plan_steps_.clear();
  next_step_f_ = 0;
  plan_measured_.reset();
  last_reprime_f_ = -1000000000;
  bad_ = 0;
  wait_since_.reset();
  beat_log_.clear();
  down_log_.clear();
  pd_log_.clear();
  bar_trust_ = false;
  lock_bad_ = 0;
  last_phase_f_ = -1000000000;
  drift_.clear();
  if (target_) pending_target_ = target_;  // re-plan once the tracker has locked
  if (clock_) {
    pending_phase_.reset();
    phase_at_ = false;
  }
}

// --- Steer ------------------------------------------------------------------

std::optional<double> SyncConductor::steer_measured() const {
  if (!last_ || f_obs_ < p_.min_track_s * kFps) return std::nullopt;
  return last_->tempo;
}

double SyncConductor::music_phase_next() const {
  return pymod(last_->beat_phase + last_->map_tempo / 60.0 / kFps, 1.0);
}

std::optional<double> SyncConductor::measured() const {
  auto m = steer_measured();
  if (!m) return std::nullopt;
  if (shown_.empty()) return m;
  return median(std::vector<double>(shown_.begin(), shown_.end()));
}

// --- tempo ------------------------------------------------------------------

std::optional<double> SyncConductor::beat_tempo(double window_s, int min_beats) const {
  const double lo = std::max(double(f_) / kFps - window_s, (last_reprime_f_ + 1.5 * kFps) / kFps);
  std::vector<double> b;
  for (double x : beat_log_)
    if (x >= lo) b.push_back(x);
  if (static_cast<int>(b.size()) < min_beats) return std::nullopt;
  std::vector<double> d;
  for (size_t i = 1; i < b.size(); ++i) d.push_back(b[i] - b[i - 1]);
  const double med = median(d);
  if (med <= 0) return std::nullopt;
  std::vector<double> idx;
  std::set<long long> uniq;
  for (double x : b) {
    const long long k = pyround((x - b[0]) / med);
    idx.push_back(double(k));
    uniq.insert(k);
  }
  if (static_cast<int>(uniq.size()) < min_beats) return std::nullopt;
  double slope, icpt;
  polyfit1(idx, b, slope, icpt);
  if (slope <= 0) return std::nullopt;
  double bpm = 60.0 / slope;
  auto m = measured();
  if (!m) return bpm;
  bpm = octave_near(bpm, *m);
  if (std::abs(std::log(bpm / *m)) < 0.06) return bpm;
  return std::nullopt;  // disagrees with the tracker: don't trust either
}

void SyncConductor::set_target(std::optional<double> bpm, bool plan) {
  target_ = bpm;
  plan_steps_.clear();
  bad_ = 0;
  if (!bpm || !plan) {
    pending_target_.reset();
    return;
  }
  plan_reprime(*bpm);
}

void SyncConductor::plan_reprime(double target) {
  auto m = measured();
  if (!m) {
    pending_target_ = target;  // no music yet: re-prime once the tracker has locked
    return;
  }
  pending_target_.reset();
  std::optional<double> bt = clock_ ? drift_tempo() : std::nullopt;
  if (!bt || !plausible(*bt)) bt = beat_tempo(ctx_frames_ / double(kFps));
  if (bt && !plausible(*bt)) bt.reset();
  if (!bt && !segs_.empty() && (f_ - last_reprime_f_ < 6 * kFps || last_reprime_f_ >= 0)) bt = segs_.back().tempo;
  plan_measured_ = bt;  // the first step stretches by target / what is heard
  const double cur = octave_near(bt ? *bt : *m, target);  // (bt: plausible or the last target)
  const double r = target / cur;
  if (std::abs(std::log(r)) < 0.006) return;
  if (segs_.empty()) segs_.push_back({0, cur});
  int n = std::max(1, p_.steps);
  if (std::abs(std::log(r)) < std::log(1.06)) n = 1;
  plan_steps_.clear();
  for (int k = 0; k < n; ++k) plan_steps_.push_back(cur * std::pow(r, double(k + 1) / n));
  next_step_f_ = f_;
}

bool SyncConductor::on_beat() {
  // switch right after a beat (the predicted phase for this frame is just
  // past 0), or after waiting one beat at most
  if (!last_) return true;
  if (!wait_since_) wait_since_ = f_;
  const double ph = music_phase_next();
  const double per = 60.0 * kFps / std::max(last_->map_tempo, 30.0);
  return ph < 1.5 / per || (f_ - *wait_since_) > per + 1;
}

void SyncConductor::frame_audio(int frame, const float *&L, const float *&R) const {
  const size_t slot = static_cast<size_t>(frame % ring_frames_) * kHop;
  L = ring_L_.data() + slot;
  R = ring_R_.data() + slot;
}

bool SyncConductor::reprime(double target, double rewind_s, std::optional<double> measured_now,
                            const char *kind, double shift) {
  // Rebuild MRT2's memory from the music so far, time-stretched so that ALL of
  // the context plays at `target` (each stretch of output since an earlier
  // re-prime by target / its own tempo: MRT2 follows the bulk of its
  // context). measured_now: the tempo actually heard now, replaces the
  // latest segment's nominal tempo. rewind_s: drop that much from the end
  // first, to land on a given beat phase (sample-precise).
  if (!reprime_fn_) return false;
  const auto t_start = std::chrono::steady_clock::now();
  const long long rw_samp = std::max<long long>(0, pyround(rewind_s * kSampleRate));
  const int rewind = static_cast<int>(rw_samp / kHop);
  const int rw_rem = static_cast<int>(rw_samp % kHop);
  const int f_end = f_ - rewind;
  const int hist_start = std::max(0, f_ - ring_frames_);
  const long long end_sample = static_cast<long long>(f_end) * kHop - rw_rem;
  const long long start_sample = static_cast<long long>(hist_start) * kHop;
  if (end_sample - start_sample < 2LL * kFps * kHop) return false;
  std::vector<Seg> segs = segs_;
  if (segs.empty()) segs.push_back({0, measured_now ? *measured_now : target});
  if (measured_now) segs.back().tempo = *measured_now;
  const int need = ctx_frames_ * kHop;
  const int margin = kSampleRate / 4;  // the stretcher's cold start falls outside the window
  std::vector<std::vector<float>> pieces_L, pieces_R;  // newest first
  long long got = 0;
  auto copy_span = [&](long long a, long long b, std::vector<float> &L, std::vector<float> &R) {
    L.resize(static_cast<size_t>(b - a));
    R.resize(static_cast<size_t>(b - a));
    for (long long s = a; s < b;) {
      const int fr = static_cast<int>(s / kHop), off = static_cast<int>(s % kHop);
      const int len = static_cast<int>(std::min<long long>(kHop - off, b - s));
      const float *fl, *frr;
      frame_audio(fr, fl, frr);
      std::copy(fl + off, fl + off + len, L.begin() + (s - a));
      std::copy(frr + off, frr + off + len, R.begin() + (s - a));
      s += len;
    }
  };
  for (int k = static_cast<int>(segs.size()) - 1; k >= 0 && got < need; --k) {
    const long long fs = segs[k].frame;
    const long long fe = k + 1 < static_cast<int>(segs.size()) ? segs[k + 1].frame : f_end;
    const long long a = std::max(fs, static_cast<long long>(hist_start)) * kHop;
    const long long b = std::min(fe * kHop, end_sample);
    if (b - a < kHop) continue;
    const double r = target / octave_near(segs[k].tempo, target);
    const long long want_src = static_cast<long long>(std::ceil((need - got) * r)) + margin;
    const long long src_len = std::min(b - a, want_src);
    std::vector<float> sL, sR;
    copy_span(b - src_len, b, sL, sR);
    if (std::abs(r - 1.0) > 1e-3) {
      const int n_out = static_cast<int>(std::lround(src_len / r));
      std::vector<float> oL(n_out), oR(n_out);
      stretch_stereo(sL.data(), sR.data(), static_cast<int>(src_len), oL.data(), oR.data(), n_out);
      pieces_L.push_back(std::move(oL));
      pieces_R.push_back(std::move(oR));
    } else {
      pieces_L.push_back(std::move(sL));
      pieces_R.push_back(std::move(sR));
    }
    got += static_cast<long long>(pieces_L.back().size());
  }
  if (got < need) return false;  // not enough history yet
  std::vector<float> ctx_L(need), ctx_R(need);
  long long pos = need;  // fill from the end
  for (size_t i = 0; i < pieces_L.size() && pos > 0; ++i) {
    const auto &pl = pieces_L[i];
    const auto &pr = pieces_R[i];
    const long long take = std::min<long long>(pos, static_cast<long long>(pl.size()));
    std::copy(pl.end() - take, pl.end(), ctx_L.begin() + (pos - take));
    std::copy(pr.end() - take, pr.end(), ctx_R.begin() + (pos - take));
    pos -= take;
  }
  prime_gain(ctx_L, p_.prime_gain_db);
  prime_gain(ctx_R, p_.prime_gain_db);
  if (!reprime_fn_(ctx_L.data(), ctx_R.data(), need)) return false;
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t_start).count();
  const double played = segs.back().tempo;
  last_reprime_f_ = f_;
  drift_.clear();
  std::vector<Seg> keep;
  for (const auto &s : segs)
    if (s.frame < f_) keep.push_back(s);
  keep.push_back({f_, target});
  if (keep.size() > 8) keep.erase(keep.begin(), keep.end() - 8);
  segs_ = keep;
  SyncEvent ev;
  ev.frame = f_;
  ev.kind = kind;
  ev.tempo = target;
  ev.ratio = target / octave_near(played, target);
  ev.rewind = rw_samp / double(kHop);
  ev.shift = shift;
  ev.ms = ms;
  events_.push_back(ev);
  ++reprimes_;
  return true;
}

// --- downbeat lock ------------------------------------------------------------

bool SyncConductor::same_grid(const Grid &a, const Grid &b) const {
  return a.bpb == b.bpb && std::abs(std::log(b.bpm / a.bpm)) < 1e-5 &&
         std::abs(pymod((b.t0 - a.t0) * b.bpm / 60.0 + b.bpb / 2.0, b.bpb) - b.bpb / 2.0) < 0.06;
}

void SyncConductor::set_clock(std::optional<Grid> g) {
  if (clock_ && g && same_grid(*clock_, *g)) {
    clock_ = g;  // the same grid, refreshed: keep the evidence and any pending correction
    if (pending_phase_) pending_phase_ = g;
    return;
  }
  const bool keep = g && phase_at_ && pending_phase_ &&
                    std::abs(std::log(g->bpm / pending_phase_->bpm)) < 1e-5;
  clock_ = g;
  lock_bad_ = 0;
  drift_.clear();
  if (!g) bar_trust_ = false;
  if (keep) {
    pending_phase_ = g;  // a pending retime to this tempo: keep it, on the fresh grid
  } else {
    pending_phase_.reset();
    phase_at_ = false;
    wait_since_.reset();
  }
}

std::optional<std::pair<int, bool>> SyncConductor::bar_fold(double a, double per, int bpb,
                                                            double lo) const {
  // which beat of MRT2's beat grid (beat j heard at a + j per) is its
  // downbeat, from its own p(downbeat): the peak around each beat heard since
  // `lo`, averaged per position in the bar -> (k, clear: 25 % above the next,
  // >= 3 bars)
  if (pd_log_.empty()) return std::nullopt;
  const int f0 = pd_log_.front().first;
  const int n = static_cast<int>(pd_log_.size());
  const double t_hi = (f_ - 1) / double(kFps) + kDecLat;
  std::vector<double> sums(bpb, 0.0), cnt(bpb, 0.0);
  const long long j0 = static_cast<long long>(std::ceil((lo - a) / per));
  const long long j1 = static_cast<long long>(std::floor((t_hi - a) / per));
  for (long long j = j0; j <= j1; ++j) {
    const long long fc = pyround((a + j * per - kDecLat) * kFps) - f0;
    if (fc - 1 < 0 || fc + 1 >= n) continue;
    const double mx = std::max({double(pd_log_[fc - 1].second), double(pd_log_[fc].second),
                                double(pd_log_[fc + 1].second)});
    const int pos = pymod_i(j, bpb);
    sums[pos] += mx;
    cnt[pos] += 1;
  }
  if (*std::min_element(cnt.begin(), cnt.end()) < 3) return std::nullopt;
  std::vector<double> sc(bpb);
  for (int i = 0; i < bpb; ++i) sc[i] = sums[i] / cnt[i];
  const int k = static_cast<int>(std::max_element(sc.begin(), sc.end()) - sc.begin());
  double second = -1e9;
  for (int i = 0; i < bpb; ++i)
    if (i != k) second = std::max(second, sc[i]);
  return std::make_pair(k, sc[k] >= 1.25 * second);
}

std::optional<int> SyncConductor::which_bar(const std::vector<int> &ks,
                                            const std::optional<std::pair<int, bool>> &fold,
                                            int bpb, bool trust) const {
  // a clear long-window fold wins; else the tracker's recent downbeats if they
  // agree -- except a half-bar jump of a bar already on the grid's (trust)
  if (fold && fold->second) return fold->first;
  std::set<int> u(ks.begin(), ks.end());
  if (u.size() != 1) return std::nullopt;
  const int k = *u.begin();
  if (trust && bar_trust_ && bpb % 2 == 0 && k == bpb / 2) return 0;
  return k;
}

std::optional<double> SyncConductor::phase_error(const Grid &g, bool trust) const {
  // how many beats MRT2's *heard* beats lag the grid (mod bpb; > 0 = late)
  const double per = 60.0 / g.bpm;
  const double lo = std::max(double(f_) / kFps - std::max(4.0, 1.5 * g.bpb * per),
                             (last_reprime_f_ + 1.5 * kFps) / kFps);
  std::vector<double> b, d;
  for (double x : beat_log_)
    if (x >= lo) b.push_back(x + kDecLat);
  for (double x : down_log_)
    if (x >= lo) d.push_back(x + kDecLat);
  if (b.size() < 4 || d.empty()) return std::nullopt;
  std::complex<double> z(0, 0);
  for (double x : b) z += std::polar(1.0, 2 * kPi * (x - g.t0) / per);
  z /= double(b.size());
  if (std::abs(z) < 0.6) return std::nullopt;  // not on one steady beat grid (yet)
  const double frac = std::arg(z) / (2 * kPi);  // its beats are this many beats late
  std::vector<int> ks;
  for (double x : d) ks.push_back(pymod_i(pyround((x - g.t0) / per - frac), g.bpb));
  const double lo_fold = std::max(double(f_) / kFps - 10.0, (last_reprime_f_ + 1.5 * kFps) / kFps);
  auto fold = bar_fold(g.t0 + frac * per, per, g.bpb, lo_fold);
  auto k = which_bar(ks, fold, g.bpb, trust);
  if (!k) return std::nullopt;  // its downbeats disagree
  return pymod(*k + frac, g.bpb);
}

std::optional<double> SyncConductor::phase_error_at(const Grid &g, double t) const {
  // phase_error for a grid at another tempo: MRT2's bar position heard at
  // stream time t, extrapolated from a least-squares fit of its recent beats
  // (at the grid's metrical level), against the grid's bar position at t
  const double lo = std::max(double(f_) / kFps - std::max(4.0, 1.5 * g.bpb * 60.0 / g.bpm),
                             (last_reprime_f_ + 1.5 * kFps) / kFps);
  std::vector<double> b, d;
  for (double x : beat_log_)
    if (x >= lo) b.push_back(x + kDecLat);
  for (double x : down_log_)
    if (x >= lo) d.push_back(x + kDecLat);
  if (b.size() < 5 || d.empty()) return std::nullopt;
  std::vector<double> diffs;
  for (size_t i = 1; i < b.size(); ++i) diffs.push_back(b[i] - b[i - 1]);
  const double med = median(diffs);
  if (med <= 0) return std::nullopt;
  std::vector<double> idx;
  std::set<long long> uniq;
  for (double x : b) {
    const long long k = pyround((x - b[0]) / med);
    idx.push_back(double(k));
    uniq.insert(k);
  }
  if (uniq.size() < 5) return std::nullopt;
  double per, a;
  polyfit1(idx, b, per, a);
  if (per <= 0) return std::nullopt;
  std::vector<double> res;
  for (size_t i = 0; i < b.size(); ++i) res.push_back(std::abs(b[i] - (a + idx[i] * per)));
  if (1.4826 * median(res) > 0.07 * per) return std::nullopt;  // not one steady grid (yet)
  per = 60.0 / octave_near(60.0 / per, g.bpm);
  std::vector<int> ks;
  for (double x : d) ks.push_back(pymod_i(pyround((x - a) / per), g.bpb));
  const double lo_fold = std::max(double(f_) / kFps - 10.0, (last_reprime_f_ + 1.5 * kFps) / kFps);
  auto fold = bar_fold(a, per, g.bpb, lo_fold);
  auto k = which_bar(ks, fold, g.bpb, true);
  if (!k) return std::nullopt;
  const double pos_m = pymod((t - a) / per - *k, g.bpb);
  const double pos_r = pymod((t - g.t0) * g.bpm / 60.0, g.bpb);
  return pymod(pos_r - pos_m, g.bpb);
}

std::optional<double> SyncConductor::beat_phase_error(const Grid &g) const {
  const double per = 60.0 / g.bpm;
  const double lo = std::max(double(f_) / kFps - std::max(4.0, 1.5 * g.bpb * per),
                             (last_reprime_f_ + 1.5 * kFps) / kFps);
  std::complex<double> z(0, 0);
  int n = 0;
  for (double x : beat_log_)
    if (x >= lo) {
      z += std::polar(1.0, 2 * kPi * (x + kDecLat - g.t0) / per);
      ++n;
    }
  if (n < 4) return std::nullopt;
  z /= double(n);
  if (std::abs(z) < 0.6) return std::nullopt;
  return std::arg(z) / (2 * kPi);
}

std::optional<double> SyncConductor::drift_tempo() const {
  if (!clock_ || drift_.size() < 6) return std::nullopt;
  const double span = drift_.back().first - drift_.front().first;
  if (span < 2.0) return std::nullopt;
  std::vector<double> t, e;
  for (const auto &d : drift_) {
    t.push_back(d.first);
    e.push_back(d.second);
  }
  double slope, icpt;
  polyfit1(t, e, slope, icpt);  // beats of lag per second
  const double tm = clock_->bpm - 60.0 * slope;
  if (!(tm > 0) || std::abs(std::log(tm / clock_->bpm)) > 0.1) return std::nullopt;
  return tm;
}

bool SyncConductor::plausible(double bpm) const {
  if (segs_.empty() || last_reprime_f_ < 0) return true;  // free-running: only the tracker knows
  const double nominal = segs_.back().tempo;
  return std::abs(std::log(octave_near(bpm, nominal) / nominal)) < 0.08;
}

std::optional<double> SyncConductor::current_tempo() const {
  if (auto d = drift_tempo(); d && plausible(*d)) return d;
  if (auto b = beat_tempo(4.0); b && plausible(*b)) return b;
  // the tracker's own (smoothed) tempo -- lags, but it does see a drift; the
  // last re-prime's target would hide one and re-prime at the old tempo
  if (auto m = measured(); m && plausible(*m) && f_ - last_reprime_f_ >= 3 * kFps) return m;
  if (!segs_.empty() && last_reprime_f_ >= 0) return segs_.back().tempo;
  return measured();
}

void SyncConductor::lock_check() {
  // every 5 frames: if MRT2 is off the grid by more than kLockTol twice in a
  // row, plan a phase re-prime for its next beat
  if (!clock_) return;
  if (auto fr = beat_phase_error(*clock_)) {  // the drift of its beat phase -> its tempo
    const double t = double(f_) / kFps;
    double v = *fr;
    if (!drift_.empty()) v = drift_.back().second + (pymod(*fr - drift_.back().second + 0.5, 1.0) - 0.5);
    drift_.emplace_back(t, v);
    while (!drift_.empty() && t - drift_.front().first > 8.0) drift_.pop_front();
  }
  if (pending_phase_ || f_ - last_reprime_f_ < 3 * kFps) return;
  const Grid g = *clock_;
  auto e = phase_error(g);
  if (!e) return;
  const double w = pymod(*e + g.bpb / 2.0, g.bpb) - g.bpb / 2.0;
  // a small offset can wait until the drift gives MRT2's tempo (a phase
  // re-prime also sets the tempo, from current_tempo())
  if (std::abs(w) < 0.5 && !drift_tempo() && f_ - last_reprime_f_ < 6 * kFps) return;
  lock_bad_ = std::abs(w) > kLockTol ? lock_bad_ + 1 : 0;
  // once locked, a jump of a beat or more needs longer evidence: the
  // tracker's downbeat can flip between two hypotheses half a bar apart
  const bool recent = f_ - last_phase_f_ < 30 * kFps;
  const bool half = std::abs(std::abs(w) - g.bpb / 2.0) < 0.3;
  const int need = (recent && std::abs(w) >= 0.75) ? (half ? 20 : 6) : 2;
  if (lock_bad_ >= need) {
    lock_bad_ = 0;
    pending_phase_ = g;
  }
}

void SyncConductor::retime_to_clock(const Grid &g) {
  const auto tm_prev = current_tempo();  // (the drift vs the old grid, before set_clock clears it)
  set_clock(g);
  auto tm = tm_prev;
  auto e = phase_error_at(g, double(f_) / kFps + kDecLat);
  if (p_.lock && !e && f_ - last_reprime_f_ < 3 * kFps) {
    // just re-primed (e.g. the grid moved by an underrun): MRT2's beat is not
    // measurable yet and its measured tempo is stale -- take the target now,
    // and let the lock / resync checks correct bar and tempo once it is
    target_ = g.bpm;
    plan_steps_.clear();
    pending_target_.reset();
    bad_ = 0;
    return;
  }
  if (!p_.lock || !e || !tm) {
    set_target(g.bpm);
    return;
  }
  target_ = g.bpm;
  plan_steps_.clear();
  pending_target_.reset();
  bad_ = 0;
  const double w = pymod(*e + g.bpb / 2.0, g.bpb) - g.bpb / 2.0;
  if (std::abs(w) < kLockTol && std::abs(std::log(octave_near(*tm, g.bpm) / g.bpm)) < 0.006) {
    bar_trust_ = true;
    return;  // already there: nothing to re-prime
  }
  pending_phase_ = g;
  phase_at_ = true;
}

bool SyncConductor::phase_reprime(const Grid &g, bool whole, bool set_tgt, bool at) {
  // re-align MRT2 to the grid: rebuild its memory from a context that ends
  // where its own music was at the grid's current bar position, (-e mod bpb)
  // of its beats back (always a rewind); the rebuilt memory ends one decoder
  // latency after its audio, hence - kDecLat. The tempo goes to the grid's in
  // the same step. at: the grid runs at another tempo; within kLockTol it is
  // a plain tempo re-prime.
  auto e = at ? phase_error_at(g, double(f_) / kFps + kDecLat) : phase_error(g, !whole);
  if (!e) return false;
  double err = *e;
  if (whole) {
    err = double(pymod_i(pyround(err), g.bpb));
    if (err == 0) return false;
  }
  auto tm = current_tempo();
  if (!tm) return false;
  const double per_m = 60.0 / octave_near(*tm, g.bpm);  // MRT2's own beat period
  const double late = kDecLat * g.bpm / 60.0;           // a plain re-prime leaves MRT2 one decoder latency later
  if (at && std::abs(pymod(err + late + g.bpb / 2.0, g.bpb) - g.bpb / 2.0) < kLockTol) {
    const bool ok = reprime(g.bpm, 0.0, octave_near(*tm, g.bpm), "reprime", 0.0);
    if (ok) {
      bar_trust_ = true;
      if (set_tgt) target_ = g.bpm;
    }
    return ok;
  }
  double rw = pymod(-err, g.bpb) * per_m - kDecLat;
  if (rw < 0) rw += g.bpb * per_m;
  const double w = pymod(err + g.bpb / 2.0, g.bpb) - g.bpb / 2.0;
  const bool ok = reprime(g.bpm, rw, octave_near(*tm, g.bpm), "phase", w);
  if (ok) {
    bar_trust_ = !whole || clock_.has_value();
    if (set_tgt) target_ = g.bpm;
    trk_.shift_beats(-(rw + kDecLat) / per_m);
    last_phase_f_ = f_;
  }
  return ok;
}

std::optional<double> SyncConductor::bar_offset() {
  if (!clock_) return std::nullopt;
  auto e = phase_error(*clock_);
  if (!e) return std::nullopt;
  return pymod(*e + clock_->bpb / 2.0, clock_->bpb) - clock_->bpb / 2.0;
}

std::vector<SyncEvent> SyncConductor::pop_events() {
  std::vector<SyncEvent> out;
  out.swap(events_);
  return out;
}

// --- frames -------------------------------------------------------------------

void SyncConductor::before_frame(bool can_block) {
  if (pending_target_ && measured()) plan_reprime(*pending_target_);
  if (p_.lock && f_ % 5 == 2) lock_check();
  else if (!p_.lock && clock_ && f_ % 5 == 2) {  // tempo only: still measure the drift
    if (auto fr = beat_phase_error(*clock_)) {
      const double t = double(f_) / kFps;
      double v = *fr;
      if (!drift_.empty()) v = drift_.back().second + (pymod(*fr - drift_.back().second + 0.5, 1.0) - 0.5);
      drift_.emplace_back(t, v);
      while (!drift_.empty() && t - drift_.front().first > 8.0) drift_.pop_front();
    }
  }
  if (pending_phase_ && plan_steps_.empty()) {
    if (can_block && on_beat()) {
      const Grid g = *pending_phase_;
      const bool at = phase_at_;
      pending_phase_.reset();
      phase_at_ = false;
      wait_since_.reset();
      if (!phase_reprime(g, false, true, at) && at)
        set_target(g.bpm);  // bar not measurable after all: the tempo now, bar later
    }
  } else if (!plan_steps_.empty()) {
    if (f_ >= next_step_f_ && can_block && on_beat()) {
      const double tg = plan_steps_.front();
      plan_steps_.erase(plan_steps_.begin());
      reprime(tg, 0.0, plan_measured_, "reprime", 0.0);
      plan_measured_.reset();
      next_step_f_ = f_ + static_cast<int>(1.5 * kFps);
      wait_since_.reset();
    }
  } else if (p_.resync && target_ && f_ % kFps == 0 && f_ - last_reprime_f_ >= 3 * kFps) {
    auto bt = (clock_ && p_.lock) ? drift_tempo() : beat_tempo(4.0);
    if (!bt) bt = beat_tempo(4.0);
    if (bt && !plausible(*bt)) bt.reset();
    if (bt) {
      const double e = std::log(octave_near(*bt, *target_) / *target_);
      const bool clk = clock_.has_value() && p_.lock;
      bad_ = std::abs(e) > (clk ? kClockResyncTol : kResyncTol) ? bad_ + 1 : 0;
      if (bad_ >= 2) {  // drifted: re-sync with one more small re-prime (on a beat)
        bad_ = 0;
        if (clk) {      // grid: the tempo and its bar position at the switch in one re-prime
          pending_phase_ = clock_;
          phase_at_ = true;
        } else {
          plan_steps_ = {*target_};
          plan_measured_ = octave_near(*bt, *target_);
          next_step_f_ = f_;
        }
      }
    }
  }
}

SyncFrameInfo SyncConductor::after_frame(const float *L, const float *R, float beat_logit,
                                         float down_logit) {
  const double pb = sigmoid(beat_logit), pd = sigmoid(down_logit);
  TrackerFrame o = trk_.push(pb, std::min(pd, pb));
  last_ = o;
  ++f_obs_;
  auto m = steer_measured();
  if (m) {
    // display value: folded to the target's octave (else to the recent
    // median) and median-filtered over 1 s
    double ref = *m;
    if (target_) ref = *target_;
    else if (!shown_.empty()) ref = median(std::vector<double>(shown_.begin(), shown_.end()));
    shown_.push_back(octave_near(*m, ref));
    while (static_cast<int>(shown_.size()) > kFps) shown_.pop_front();
  }
  pd_log_.emplace_back(f_, static_cast<float>(pd));
  while (static_cast<int>(pd_log_.size()) > 14 * kFps) pd_log_.pop_front();
  for (double x : o.beats) beat_log_.push_back(x);
  while (beat_log_.size() > 96) beat_log_.pop_front();
  for (double x : o.downbeats) down_log_.push_back(x);
  while (down_log_.size() > 32) down_log_.pop_front();
  const size_t slot = static_cast<size_t>(f_ % ring_frames_) * kHop;
  std::copy(L, L + kHop, ring_L_.begin() + slot);
  std::copy(R, R + kHop, ring_R_.begin() + slot);
  SyncFrameInfo info;
  info.frame = f_;
  info.p_beat = static_cast<float>(pb);
  info.p_down = static_cast<float>(pd);
  info.tempo = measured();
  info.beat_phase = o.beat_phase;
  info.bpb = o.map_bpb;
  info.beats = o.beats;
  info.downs = o.downbeats;
  ++f_;
  return info;
}

// ─── TempoSyncController ────────────────────────────────────────────────────

TempoSyncController::TempoSyncController(core::MLXEngine &engine)
    : engine_(engine), cond_(150) {
  cond_.set_reprime([this](const float *L, const float *R, int n) {
    return engine_.reprime_from_audio(L, R, n);
  });
}

bool TempoSyncController::available() const {
  return engine_.has_tempo_models() && engine_.step_has_readout() &&
         engine_.reprime_context_frames() == cond_.ctx_frames();
}

void TempoSyncController::reset() {
  cond_.reset();
  info_ = SyncFrameInfo{};
}

void TempoSyncController::follow_host(const HostTransport &h) {
  if (!h.valid || !(h.tempo > 1.0) || !std::isfinite(h.beat)) return;
  Grid g;
  g.bpm = h.tempo;
  g.bpb = std::clamp(h.beats_per_bar, 1, 16);
  g.t0 = double(h.cursor) / kSampleRate + (h.bar_start_beat - h.beat) * 60.0 / h.tempo;
  auto c = cond_.clock();
  if (!c) {
    cond_.retime_to_clock(g);
    return;
  }
  const bool tempo_changed = std::abs(std::log(g.bpm / c->bpm)) > 1e-5 || g.bpb != c->bpb;
  if (!h.playing) {
    // stopped: the host's beat position stands still -- free-run the last
    // grid, following tempo changes only (the next downbeat stays put)
    if (tempo_changed) {
      Grid g2 = g;
      g2.t0 = c->t0;
      cond_.retime_to_clock(g2);
    }
    return;
  }
  if (tempo_changed || !cond_.same_grid(*c, g)) {
    cond_.retime_to_clock(g);  // new tempo or a transport jump
  } else {
    cond_.set_clock(g);        // the same grid, refreshed
  }
}

bool TempoSyncController::step(float *L, float *R, const HostTransport &host, bool can_block,
                               std::int32_t *tokens_out) {
  const bool avail = available();
  const bool on = enabled() && avail;
  cond_.params().lock = bar_lock();
  if (on) {
    if (!was_enabled_) {
      cond_.set_clock(std::nullopt);
      cond_.set_target(std::nullopt);
    }
    follow_host(host);
  } else if (was_enabled_) {
    cond_.set_clock(std::nullopt);  // released: MRT2 free-runs at its tempo
    cond_.set_target(std::nullopt);
  }
  was_enabled_ = on;
  // optional: no explicit musiccoca CFG while synced (keep_cfg = false). In
  // the closed loop the explicit CFG's pull toward the prompt's own tempo is
  // corrected by the next re-prime, and it keeps the style (and its beat)
  // steadier, so it stays on by default
  const bool cfg_off = on && !keep_cfg();
  if (cfg_off != cfg_overridden_) {
    engine_.set_explicit_cfg_musiccoca(!cfg_off);
    cfg_overridden_ = cfg_off;
  }
  if (on) cond_.before_frame(can_block);
  const bool ok = engine_.generate_frame(L, R, tokens_out);
  if (!ok) return false;
  if (engine_.step_has_readout()) {
    float ro[2];
    engine_.last_readout(ro);
    info_ = cond_.after_frame(L, R, ro[0], ro[1]);
  }
  // status snapshot for the UI (cheap; the bar offset every 5 frames)
  auto evs = cond_.pop_events();
  std::lock_guard<std::mutex> lk(status_mutex_);
  status_.enabled = enabled();
  status_.available = available();
  status_.host_valid = host.valid;
  status_.host_bpm = host.tempo;
  status_.host_bpb = host.beats_per_bar;
  status_.playing = host.playing;
  auto m = cond_.measured();
  status_.mrt_bpm = m ? *m : 0.0;
  if (cond_.frame() % 5 == 0) {
    auto off = on ? cond_.bar_offset() : std::nullopt;
    status_.offset_valid = off.has_value();
    status_.offset_beats = off ? *off : 0.0;
  }
  status_.busy = on && cond_.busy();
  status_.reprimes = cond_.reprime_count();
  status_.frame = cond_.frame();
  for (const auto &e : evs) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%s -> %.1f BPM (x%.3f%s, %.0f ms)", e.kind.c_str(), e.tempo,
                  e.ratio, e.kind == "phase" ? ", bar" : "", e.ms);
    status_.last_event = buf;
    pending_events_.push_back(e);
    if (pending_events_.size() > 64) pending_events_.erase(pending_events_.begin());
  }
  return true;
}

SyncStatus TempoSyncController::status() const {
  std::lock_guard<std::mutex> lk(status_mutex_);
  return status_;
}

std::vector<SyncEvent> TempoSyncController::pop_events() {
  std::lock_guard<std::mutex> lk(status_mutex_);
  std::vector<SyncEvent> out;
  out.swap(pending_events_);
  return out;
}

}  // namespace tempo
}  // namespace magentart
