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

#include <magentart/tempo/beat_tracker.h>

#include <algorithm>
#include <cmath>

namespace magentart {
namespace tempo {

namespace {
constexpr double kTempoEmaS = 1.0;  // time constant of the causal tempo estimate
}

BarPointer::BarPointer(const BarPointerConfig &cfg)
    : fps_(cfg.fps), bpb_(cfg.beats_per_bar), obs_lambda_(cfg.obs_lambda) {
  const int pmin = static_cast<int>(std::floor(60.0 * fps_ / cfg.max_bpm));
  const int pmax = static_cast<int>(std::ceil(60.0 * fps_ / cfg.min_bpm));
  for (int p = pmin; p <= pmax; ++p) periods_.push_back(p);
  const int nm = n_meters(), nt = n_tempi();
  offsets_.assign(nm * nt, 0);
  n_ = 0;
  for (int j = 0; j < nm; ++j)
    for (int ti = 0; ti < nt; ++ti) {
      offsets_[j * nt + ti] = n_;
      n_ += bpb_[j] * periods_[ti];
    }
  meter_of_.assign(n_, 0);
  tempo_of_.assign(n_, 0);
  pos_of_.assign(n_, 0);
  region_.assign(n_, 0);
  is_end_.assign(n_, 0);
  const int width = std::max(1, static_cast<int>(std::lround(cfg.obs_width * fps_)));
  for (int j = 0; j < nm; ++j)
    for (int ti = 0; ti < nt; ++ti) {
      const int P = periods_[ti], b = bpb_[j], o = offset(j, ti);
      for (int pos = 0; pos < b * P; ++pos) {
        const int s = o + pos, pib = pos % P, beat = pos / P;
        meter_of_[s] = j;
        tempo_of_[s] = ti;
        pos_of_[s] = pos;
        region_[s] = pib < width ? (beat == 0 ? 2 : 1) : 0;
        is_end_[s] = pib == P - 1;
      }
    }
  // tempo transitions: Laplace on the period ratio + a uniform jump floor
  ttempo_.assign(nt * nt, 0.0);
  for (int ti = 0; ti < nt; ++ti) {
    double sum = 0;
    for (int tj = 0; tj < nt; ++tj) {
      const double r = static_cast<double>(periods_[tj]) / periods_[ti];
      const double v = std::exp(-cfg.tempo_lambda * std::abs(r - 1.0));
      ttempo_[ti * nt + tj] = v;
      sum += v;
    }
    for (int tj = 0; tj < nt; ++tj)
      ttempo_[ti * nt + tj] =
          (1 - cfg.jump_prob) * ttempo_[ti * nt + tj] / sum + cfg.jump_prob / nt;
  }
  psw_ = nm > 1 ? cfg.meter_switch_prob : 0.0;
}

void BarPointer::transition(const std::vector<double> &alpha,
                            std::vector<double> &out) const {
  const int nm = n_meters(), nt = n_tempi();
  out.assign(n_, 0.0);
  for (int s = 0; s < n_ - 1; ++s)
    if (!is_end_[s]) out[s + 1] = alpha[s];
  std::vector<double> a(nt), acc(nt);
  for (int j = 0; j < nm; ++j) {
    const int b = bpb_[j];
    for (int k = 0; k < b; ++k) {
      // sources: the last state of beat k in every tempo block of meter j
      for (int ti = 0; ti < nt; ++ti)
        a[ti] = alpha[offset(j, ti) + (k + 1) * periods_[ti] - 1];
      std::fill(acc.begin(), acc.end(), 0.0);
      for (int ti = 0; ti < nt; ++ti) {
        const double v = a[ti];
        if (v == 0.0) continue;
        const double *row = &ttempo_[ti * nt];
        for (int tj = 0; tj < nt; ++tj) acc[tj] += v * row[tj];
      }
      if (k < b - 1) {  // next beat of the same bar
        for (int tj = 0; tj < nt; ++tj)
          out[offset(j, tj) + (k + 1) * periods_[tj]] = acc[tj];
      } else {          // bar end -> downbeat of (possibly another) meter
        for (int j2 = 0; j2 < nm; ++j2) {
          const double pm = j2 == j ? 1.0 - psw_ : psw_ / std::max(nm - 1, 1);
          if (pm <= 0) continue;
          for (int tj = 0; tj < nt; ++tj) out[offset(j2, tj)] += acc[tj] * pm;
        }
      }
    }
  }
}

double BarPointer::marg_tempo(const std::vector<double> &tm, int half) const {
  const int nt = n_tempi();
  const int k = static_cast<int>(std::max_element(tm.begin(), tm.end()) - tm.begin());
  double ws = 0, wl = 0;
  for (int d = -half; d <= half; ++d) {
    const int i = std::clamp(k + d, 0, nt - 1);
    const double w = tm[i] + 1e-12;
    ws += w;
    wl += w * std::log(60.0 * fps_ / periods_[i]);
  }
  return std::exp(wl / ws);
}

OnlineBeatTracker::OnlineBeatTracker(const BarPointer &dbn, int fps_in)
    : d_(dbn), up_(static_cast<int>(std::lround(double(dbn.fps()) / fps_in))) {
  reset();
}

void OnlineBeatTracker::reset() {
  alpha_.clear();
  has_alpha_ = has_tm_ = false;
  t_ = 0;
  has_prev_ = has_mid_ = false;
  prev_ = mid_pred_ = 0;
  mid_down_ = false;
  last_beat_ = -1e9;
}

TrackerFrame OnlineBeatTracker::push(double p_beat, double p_down) {
  const BarPointer &d = d_;
  const int n = d.n_states(), nt = d.n_tempi(), nm = d.n_meters();
  p_down = std::min(p_down, p_beat);
  const double pb = std::clamp(p_beat, 1e-4, 1 - 1e-4);
  const double pd = std::clamp(p_down, 1e-4, 1 - 1e-4);
  const double om[3] = {(1 - pb) / (d.obs_lambda() - 1), std::max(pb - pd, 1e-4), pd};
  const double a_ema = 1.0 / (kTempoEmaS * d.fps());
  TrackerFrame out;
  int st = 0;
  double conf = 0;
  for (int i = 0; i < up_; ++i) {
    if (has_alpha_) {
      d.transition(alpha_, scratch_);
      alpha_.swap(scratch_);
    } else {
      alpha_.assign(n, 1.0 / n);
      has_alpha_ = true;
    }
    double tot = 0;
    for (int s = 0; s < n; ++s) {
      alpha_[s] *= om[d.region(s)];
      tot += alpha_[s];
    }
    tm_.assign(nt, 0.0);
    mm_.assign(nm, 0.0);
    double best = -1;
    const double inv = tot > 0 ? 1.0 / tot : 0.0;
    for (int s = 0; s < n; ++s) {
      const double v = alpha_[s] * inv;
      alpha_[s] = v;
      tm_[d.tempo_of(s)] += v;
      mm_[d.meter_of(s)] += v;
      if (v > best) {
        best = v;
        st = s;
      }
    }
    conf = best;
    // beat decisions (decode_online): place each beat where the MAP state at
    // the middle of the previous beat predicted it
    const int j = d.meter_of(st), P = d.period(d.tempo_of(st)), pos = d.pos_of(st);
    const int pib = pos % P, bidx = pos / P;
    const double tg = static_cast<double>(t_ + i);
    const double ph = double(pib) / P;
    if (has_prev_) {
      if (!has_mid_ && ph >= 0.5) {
        mid_pred_ = tg + P - pib;
        mid_down_ = (bidx + 1) % d.bpb(j) == 0;
        has_mid_ = true;
      }
      if (ph < prev_ - 0.5) {
        const double bt = has_mid_ ? mid_pred_ : tg - pib;
        const bool isd = has_mid_ ? mid_down_ : bidx == 0;
        if (bt > last_beat_) {
          out.beats.push_back(bt / d.fps());
          if (isd) out.downbeats.push_back(bt / d.fps());
          last_beat_ = bt;
        }
        has_mid_ = false;
      }
    }
    prev_ = ph;
    has_prev_ = true;
    if (!has_tm_) {
      tm_smooth_ = tm_;
      has_tm_ = true;
    } else {
      for (int k = 0; k < nt; ++k) tm_smooth_[k] += a_ema * (tm_[k] - tm_smooth_[k]);
    }
  }
  t_ += up_;
  const int j = d.meter_of(st), P = d.period(d.tempo_of(st)), pos = d.pos_of(st);
  const int pib = pos % P, bidx = pos / P;
  out.tempo = d.marg_tempo(tm_smooth_);
  out.meter = d.bpb(static_cast<int>(std::max_element(mm_.begin(), mm_.end()) - mm_.begin()));
  out.beat_phase = double(pib) / P;
  out.beat_in_bar = bidx;
  out.bar_phase = (bidx + double(pib) / P) / d.bpb(j);
  out.map_bpb = d.bpb(j);
  out.map_tempo = 60.0 * d.fps() / P;
  out.next_beat = double(t_ - 1 + P - pib) / d.fps();
  out.conf = conf;
  return out;
}

void OnlineBeatTracker::shift_beats(double beats) {
  if (!has_alpha_) return;
  const BarPointer &d = d_;
  std::vector<double> tmp;
  for (int j = 0; j < d.n_meters(); ++j)
    for (int ti = 0; ti < d.n_tempi(); ++ti) {
      const int P = d.period(ti), o = d.offset(j, ti), L = d.bpb(j) * P;
      long sh = std::lround(beats * P) % L;
      if (sh < 0) sh += L;
      if (sh == 0) continue;
      tmp.assign(alpha_.begin() + o, alpha_.begin() + o + L);
      for (int i = 0; i < L; ++i) alpha_[o + (i + sh) % L] = tmp[i];  // np.roll
    }
  has_prev_ = false;  // no spurious beat from the jump
  has_mid_ = false;
}

}  // namespace tempo
}  // namespace magentart

