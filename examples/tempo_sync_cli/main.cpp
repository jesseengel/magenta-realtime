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

// tempo_sync_cli — offline test harness for host tempo / downbeat sync.
//
// Runs the engine + TempoSyncController exactly as RealtimeRunner does in the
// AU, against a simulated DAW transport, with a simulated real-time output
// clock: every step advances a virtual wall clock by its measured duration,
// the virtual audio device consumes the ring buffer at 48 kHz (underruns
// stall the stream, as in the AU), and the ring holds at most `--lead`
// seconds. Re-primes only run when the buffered audio covers them.
//
// Writes <out>.wav (the output stream, 48 kHz stereo float) and <out>.json
// (per-frame read-out, tracker beats, host snapshots, events) for
// rhythm/au_sync/scripts/eval_tempo_sync.py.
//
//   tempo_sync_cli <model_dir> [--resources DIR] [--prompt TEXT] [--seconds 60]
//       [--host T:BPM[:BPB]]... [--jump T:BEATS]... [--sync-at 8] [--bar-offset B]
//       [--no-lock] [--keep-cfg | --drop-cfg] [--lead 0.5] [--cfg-musiccoca 3] [--temperature 1.3]
//       [--top-k 40] [--seed-rotation N] [--latency S] [--out PREFIX]

#include <magentart/mlx_engine.h>
#include <magentart/tempo/tempo_sync.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace magentart;

namespace {

bool write_wav(const std::string &path, const std::vector<float> &inter, int sr) {
  const uint16_t nc = 2, bits = 32, fmt = 3;
  const uint32_t frames = static_cast<uint32_t>(inter.size() / nc);
  const uint32_t data = frames * nc * bits / 8, chunk = 36 + data, sub1 = 16;
  const uint32_t byte_rate = sr * nc * bits / 8;
  const uint16_t align = nc * bits / 8;
  const uint32_t usr = sr;
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  f.write("RIFF", 4); f.write((const char *)&chunk, 4); f.write("WAVE", 4);
  f.write("fmt ", 4); f.write((const char *)&sub1, 4); f.write((const char *)&fmt, 2);
  f.write((const char *)&nc, 2); f.write((const char *)&usr, 4);
  f.write((const char *)&byte_rate, 4); f.write((const char *)&align, 2);
  f.write((const char *)&bits, 2); f.write("data", 4); f.write((const char *)&data, 4);
  f.write((const char *)inter.data(), inter.size() * sizeof(float));
  return f.good();
}

// Simulated DAW transport: piecewise-constant tempo on a wall-clock timeline
// (s since playback start), plus locate jumps.
struct TempoPoint { double t, bpm; int bpb; };
struct Jump { double t, beats; };

struct HostSim {
  std::vector<TempoPoint> tempo{{0.0, 120.0, 4}};
  std::vector<Jump> jumps;
  double bar_offset = 0.0;  // beats: where the host's bars start relative to beat 0
  double beat_at(double t) const {
    double b = 0.0;
    for (size_t i = 0; i < tempo.size(); ++i) {
      const double t0 = tempo[i].t;
      const double t1 = i + 1 < tempo.size() ? tempo[i + 1].t : 1e18;
      if (t <= t0) break;
      b += (std::min(t, t1) - t0) * tempo[i].bpm / 60.0;
    }
    for (const auto &j : jumps)
      if (t >= j.t) b += j.beats;
    return b;
  }
  const TempoPoint &at(double t) const {
    size_t k = 0;
    for (size_t i = 0; i < tempo.size(); ++i)
      if (t >= tempo[i].t) k = i;
    return tempo[k];
  }
};

std::string jnum(double v, int prec = 4) {
  if (!std::isfinite(v)) return "null";
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.*f", prec, v);
  return buf;
}

std::string jstr(const std::string &s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    o += c;
  }
  return o + "\"";
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model_dir> [options] (see the header of main.cpp)\n", argv[0]);
    return 1;
  }
  std::filesystem::path model_dir = argv[1];
  const char *mh = std::getenv("MAGENTA_HOME");
  std::string resources = mh ? std::string(mh) + "/resources" : "";
  std::string prompt = "funk groove with slap bass, clavinet and tight drums";
  std::string out = "tempo_sync";
  double seconds = 60.0, sync_at = 8.0, lead_s = 0.5, latency_s = 0.0;
  float cfg_mc = 3.0f, temperature = 1.3f;
  int top_k = 40, seed_rotation = 0;
  bool lock = true, keep_cfg = true;
  HostSim host;
  bool tempo_given = false;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(1); }
      return argv[++i];
    };
    if (a == "--resources") resources = next();
    else if (a == "--prompt") prompt = next();
    else if (a == "--seconds") seconds = std::atof(next().c_str());
    else if (a == "--sync-at") sync_at = std::atof(next().c_str());
    else if (a == "--lead") lead_s = std::atof(next().c_str());
    else if (a == "--latency") latency_s = std::atof(next().c_str());
    else if (a == "--cfg-musiccoca") cfg_mc = std::stof(next());
    else if (a == "--temperature") temperature = std::stof(next());
    else if (a == "--top-k") top_k = std::atoi(next().c_str());
    else if (a == "--seed-rotation") seed_rotation = std::atoi(next().c_str());
    else if (a == "--bar-offset") host.bar_offset = std::atof(next().c_str());
    else if (a == "--no-lock") lock = false;
    else if (a == "--keep-cfg") keep_cfg = true;
    else if (a == "--drop-cfg") keep_cfg = false;
    else if (a == "--out") out = next();
    else if (a == "--host") {
      TempoPoint p{0, 120, 4};
      std::string v = next();
      std::sscanf(v.c_str(), "%lf:%lf:%d", &p.t, &p.bpm, &p.bpb);
      if (!tempo_given) host.tempo.clear();
      tempo_given = true;
      host.tempo.push_back(p);
    } else if (a == "--jump") {
      Jump j{0, 0};
      std::sscanf(next().c_str(), "%lf:%lf", &j.t, &j.beats);
      host.jumps.push_back(j);
    } else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      return 1;
    }
  }
  std::sort(host.tempo.begin(), host.tempo.end(), [](auto &x, auto &y) { return x.t < y.t; });
  const std::string name = model_dir.filename().string();
  const std::string step_path = (model_dir / (name + ".mlxfn")).string();
  const std::string reprime_path = (model_dir / (name + "_reprime.mlxfn")).string();
  const std::string encoder_path = (model_dir / (name + "_encoder.mlxfn")).string();

  core::MLXEngine engine;
  if (!engine.init_assets(resources.c_str(), "musiccoca")) {
    std::fprintf(stderr, "init_assets failed (%s)\n", resources.c_str());
    return 1;
  }
  if (!engine.load_model(step_path.c_str())) return 1;
  if (!engine.load_tempo_models(reprime_path.c_str(), encoder_path.c_str(), 150)) return 1;
  engine.set_temperature(temperature);
  engine.set_top_k(top_k);
  engine.set_cfg_musiccoca(cfg_mc);
  engine.set_cfg_notes(1.0f);
  engine.set_cfg_drums(1.0f);
  engine.set_seed_rotation(seed_rotation);
  engine.reset_state();
  engine.set_text_prompt(prompt);
  while (engine.get_text_encoder_status() == 1 || engine.get_quantizer_status() == 1)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

  tempo::TempoSyncController sync(engine);
  sync.set_bar_lock(lock);
  sync.set_keep_cfg(keep_cfg);

  const int SR = tempo::kSampleRate, H = tempo::kHop;
  const int n_frames = static_cast<int>(seconds * tempo::kFps);
  const double cap = lead_s * SR;            // ring capacity in sync mode
  const double need_block = 0.42 * SR;       // buffered audio a re-prime may block for
  std::vector<float> inter;
  inter.reserve(static_cast<size_t>(n_frames) * H * 2);
  std::vector<float> L(H), R(H);
  std::ostringstream js_frames, js_events;
  double play = 0.0;      // stream samples the virtual device has played
  double wall = 0.0;      // virtual wall clock (s since playback start)
  long long gen = 0;      // stream samples generated
  double underrun = 0.0;  // samples of stalls
  int n_underruns = 0;
  std::vector<double> step_ms;
  bool first_frame = true, first_event = true;
  for (int f = 0; f < n_frames; ++f) {
    const bool priming = f < 3;  // as RealtimeRunner::start_locked
    sync.set_enabled(wall >= sync_at);
    tempo::HostTransport ht;
    const auto &tp = host.at(wall);
    ht.valid = true;
    ht.tempo = tp.bpm;
    ht.beats_per_bar = tp.bpb;
    ht.cursor = static_cast<std::uint64_t>(play);
    ht.beat = host.beat_at(wall) - latency_s * tp.bpm / 60.0;
    ht.bar_start_beat = host.bar_offset + std::floor((ht.beat - host.bar_offset) / tp.bpb) * tp.bpb;
    ht.playing = true;
    const double avail = double(gen) - play;
    const bool can_block = avail >= need_block;
    const auto t0 = std::chrono::steady_clock::now();
    if (!sync.step(L.data(), R.data(), ht, can_block)) {
      std::fprintf(stderr, "step failed at frame %d\n", f);
      return 1;
    }
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    step_ms.push_back(dt * 1000.0);
    if (!priming) {
      const double consumed = dt * SR;
      if (consumed > avail) {  // the device ran dry: the stream stalls
        underrun += consumed - std::max(avail, 0.0);
        ++n_underruns;
        play = double(gen);
      } else {
        play += consumed;
      }
      wall += dt;
    }
    gen += H;
    if (!priming && double(gen) - play > cap) {  // the inference thread waits for ring space
      const double wait = (double(gen) - play - cap) / SR;
      wall += wait;
      play = double(gen) - cap;
    }
    for (int i = 0; i < H; ++i) {
      inter.push_back(L[i]);
      inter.push_back(R[i]);
    }
    const auto &info = sync.last_info();
    const auto st = sync.status();
    auto clk = sync.conductor().clock();
    js_frames << (first_frame ? "" : ",\n") << "[" << f << "," << jnum(info.p_beat) << "," << jnum(info.p_down)
              << "," << (info.tempo ? jnum(*info.tempo, 2) : "null") << "," << ht.cursor << ","
              << jnum(ht.beat, 5) << "," << jnum(ht.tempo, 3) << "," << ht.beats_per_bar << ","
              << jnum(wall, 4) << "," << (st.enabled ? 1 : 0) << ","
              << (clk ? jnum(clk->t0, 5) : "null") << "," << (st.offset_valid ? jnum(st.offset_beats, 3) : "null")
              << "," << jnum(dt * 1000.0, 1) << "]";
    first_frame = false;
    for (const auto &e : sync.pop_events()) {
      js_events << (first_event ? "" : ",\n") << "{\"f\":" << e.frame << ",\"t\":" << jnum(e.frame / 25.0, 2)
                << ",\"kind\":" << jstr(e.kind) << ",\"tempo\":" << jnum(e.tempo, 2)
                << ",\"ratio\":" << jnum(e.ratio) << ",\"rewind\":" << jnum(e.rewind, 2)
                << ",\"shift\":" << jnum(e.shift, 3) << ",\"ms\":" << jnum(e.ms, 0)
                << ",\"lead_s\":" << jnum(avail / SR, 3) << "}";
      std::printf("  [%6.2f s] %s -> %.2f BPM (x%.4f) rewind %.2f fr, shift %+.2f, %.0f ms, lead %.2f s\n",
                  e.frame / 25.0, e.kind.c_str(), e.tempo, e.ratio, e.rewind, e.shift, e.ms, avail / SR);
      first_event = false;
    }
    if ((f + 1) % 250 == 0) {
      std::printf("frame %d: MRT2 %.1f BPM, host %.1f, offset %s, re-primes %d, underruns %d (%.0f ms)\n", f + 1,
                  st.mrt_bpm, st.host_bpm, st.offset_valid ? jnum(st.offset_beats, 2).c_str() : "-", st.reprimes,
                  n_underruns, 1000.0 * underrun / SR);
      std::fflush(stdout);
    }
  }
  write_wav(out + ".wav", inter, SR);
  std::vector<double> ms = step_ms;
  std::sort(ms.begin(), ms.end());
  std::ofstream js(out + ".json");
  js << "{\"model\":" << jstr(name) << ",\"prompt\":" << jstr(prompt) << ",\"seconds\":" << seconds
     << ",\"sync_at\":" << sync_at << ",\"lead_s\":" << lead_s << ",\"latency_s\":" << latency_s
     << ",\"lock\":" << (lock ? "true" : "false") << ",\"keep_cfg\":" << (keep_cfg ? "true" : "false")
     << ",\"cfg_musiccoca\":" << cfg_mc << ",\"temperature\":" << temperature << ",\"seed_rotation\":" << seed_rotation
     << ",\"bar_offset\":" << host.bar_offset << ",\"host\":[";
  for (size_t i = 0; i < host.tempo.size(); ++i)
    js << (i ? "," : "") << "[" << host.tempo[i].t << "," << host.tempo[i].bpm << "," << host.tempo[i].bpb << "]";
  js << "],\"jumps\":[";
  for (size_t i = 0; i < host.jumps.size(); ++i)
    js << (i ? "," : "") << "[" << host.jumps[i].t << "," << host.jumps[i].beats << "]";
  js << "],\"underruns\":" << n_underruns << ",\"underrun_ms\":" << jnum(1000.0 * underrun / SR, 1)
     << ",\"step_ms_median\":" << jnum(ms[ms.size() / 2], 2)
     << ",\"frame_cols\":[\"f\",\"p_beat\",\"p_down\",\"mrt_bpm\",\"cursor\",\"host_beat\",\"host_bpm\",\"bpb\","
        "\"wall\",\"enabled\",\"grid_t0\",\"offset\",\"ms\"],\"frames\":[\n"
     << js_frames.str() << "],\"events\":[\n" << js_events.str() << "]}\n";
  std::printf("wrote %s.wav / %s.json: %d frames, %d re-primes, %d underruns (%.0f ms), step median %.1f ms\n",
              out.c_str(), out.c_str(), n_frames, sync.status().reprimes, n_underruns, 1000.0 * underrun / SR,
              ms[ms.size() / 2]);
  return 0;
}
