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

// tempo_tracker_test — runs the C++ OnlineBeatTracker over a text file of
// 25-fps "p_beat p_down" rows and prints, per frame,
// "frame tempo beat_phase map_bpb n_beats beat_times... | downbeat_times...",
// for a bit-level comparison with the Python reference
// (rhythm/rhythmlib/dbn.py OnlineTracker; rhythm/au_sync/scripts/check_tracker_port.py).

#include <magentart/tempo/beat_tracker.h>
#include <magentart/tempo/tempo_sync.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

// --stretch: time alignment of stretch_stereo (Signalsmith exact()) on a
// click track -- output click k should sit at input click k / rate
static int stretch_test() {
  const int sr = 48000, n_in = 7 * sr;
  const double per = 0.5;  // clicks every 0.5 s
  for (double rate : {0.88, 1.0 / 0.95, 1.15}) {
    std::vector<float> L(n_in, 0.0f), R(n_in, 0.0f);
    std::vector<double> clicks;
    for (double t = 0.25; t < 7.0; t += per) {
      const int i0 = static_cast<int>(t * sr);
      for (int k = 0; k < 96; ++k) L[i0 + k] = R[i0 + k] = 0.8f * std::sin(3.14159265 * k / 96.0);
      clicks.push_back(t + 48.0 / sr);  // click centre
    }
    const int n_out = static_cast<int>(std::lround(n_in / rate));
    std::vector<float> oL(n_out), oR(n_out);
    magentart::tempo::stretch_stereo(L.data(), R.data(), n_in, oL.data(), oR.data(), n_out);
    // energy centroid around each expected position
    double sum_err = 0, max_err = 0;
    int n = 0;
    for (double c : clicks) {
      const double exp_t = c / rate;
      const int lo = static_cast<int>((exp_t - 0.08) * sr), hi = static_cast<int>((exp_t + 0.08) * sr);
      if (lo < 0 || hi >= n_out) continue;
      double w = 0, wt = 0;
      for (int i = lo; i < hi; ++i) {
        const double e = double(oL[i]) * oL[i];
        w += e;
        wt += e * i;
      }
      if (w <= 0) continue;
      const double err = (wt / w / sr - exp_t) * 1000.0;
      sum_err += err;
      max_err = std::max(max_err, std::abs(err));
      ++n;
    }
    std::printf("rate %.4f: %d clicks, mean offset %+.2f ms (output - expected), max |offset| %.2f ms\n", rate, n,
                sum_err / std::max(n, 1), max_err);
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc >= 2 && std::strcmp(argv[1], "--stretch") == 0) return stretch_test();
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s probs.txt\n", argv[0]);
    return 1;
  }
  magentart::tempo::BarPointerConfig cfg;
  cfg.beats_per_bar = {3, 4};
  cfg.min_bpm = 55;
  cfg.max_bpm = 200;
  cfg.meter_switch_prob = 1e-4;
  magentart::tempo::BarPointer dbn(cfg);
  magentart::tempo::OnlineBeatTracker trk(dbn);
  std::ifstream in(argv[1]);
  double pb, pd;
  int f = 0;
  while (in >> pb >> pd) {
    auto o = trk.push(pb, pd);
    std::printf("%d %.6f %.6f %d %zu", f, o.tempo, o.beat_phase, o.map_bpb, o.beats.size());
    for (double b : o.beats) std::printf(" %.4f", b);
    std::printf(" |");
    for (double b : o.downbeats) std::printf(" %.4f", b);
    std::printf("\n");
    ++f;
  }
  std::fprintf(stderr, "n_states %d, %d frames\n", dbn.n_states(), f);
  return 0;
}
