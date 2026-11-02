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

/// @file realtime_runner.h
/// @brief Audio-thread-safe wrapper around `MLXEngine` for real-time hosts.
///
/// `magentart::core::RealtimeRunner` adds to `MLXEngine`:
///   - An inference thread that calls `generate_frame` on a 25 Hz cadence.
///   - Stereo ring buffers so the audio thread can pull arbitrary block sizes.
///   - Volume / mute / bypass smoothing (one-pole) applied on the audio thread.
///   - A MIDI-gate envelope that attenuates output when no notes are held.
///   - PromptSurface controls: a 2D plane blends MusicCoCa prompts by inverse-distance
///     weighting of user-controlled (x, y) coordinates per prompt slot.
///
/// Threading model
///   - Lifecycle methods (`load_model`, `unload`, `start`, `stop`) are serialized
///     by an internal mutex; call from the UI or controller thread.
///   - `read_audio_stereo` is the audio callback entry point. Lock-free except
///     for its internal ring buffer's atomics.
///   - Parameter setters (sampling, MIDI, prompt surface, volume, prompts) are
///     atomic and safe from any thread.
///   - Most setters are thin forwarders to the embedded `MLXEngine`; see
///     `mlx_engine.h` for semantics.
///
/// This header only declares the public interface; all non-trivial bodies live
/// in `core/src/realtime_runner.cpp` to keep compile times reasonable and to
/// avoid forcing Objective-C++ on consumers (the MLX inference loop needs an
/// Objective-C autorelease pool per iteration, which we wrap portably in the
/// .cpp via `magentart::detail::AutoreleasePool`).

#include <magentart/mlx_engine.h>
#include <magentart/ring_buffer.h>
#include <magentart/tempo/tempo_sync.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace magentart {
namespace core {

/// Snapshot of runner state surfaced to UI (populated by the inference thread).
struct EngineMetrics {
    float transformer_ms = 0;
    float total_ms = 0;
    std::size_t buffer_available = 0;
    std::size_t buffer_capacity = RingBuffer::kCapacity;
    int transport_flags = -1; ///< -1 uninitialized, -2 no host blocks, -3 block returned NO
    std::uint64_t dropped_frames = 0; ///< Cumulative real-time ring-buffer underruns since last reset.
};

/// Simple attack/release one-pole envelope. Lock-free via atomic `value`.
struct ExponentialEnvelope {
    std::atomic<float> value{0.0f};
    float alpha_attack = 0.0f;
    float alpha_release = 0.0f;

    void set_attack_samples(float samples) {
        alpha_attack = 1.0f - std::exp(-4.60517f / samples);
    }
    void set_release_samples(float samples) {
        alpha_release = 1.0f - std::exp(-4.60517f / samples);
    }
    float tick(float target) {
        float current = value.load(std::memory_order_relaxed);
        float alpha, next;
        do {
            alpha = (target > current) ? alpha_attack : alpha_release;
            next = current + (target - current) * alpha;
        } while (!value.compare_exchange_weak(current, next, std::memory_order_relaxed));
        return next;
    }
};

class RealtimeRunner {
public:
    RealtimeRunner();
    ~RealtimeRunner();

    RealtimeRunner(const RealtimeRunner&) = delete;
    RealtimeRunner& operator=(const RealtimeRunner&) = delete;

    /// @name Lifecycle
    /// Controller-thread only; serialized internally by `lifecycle_mutex_`.
    /// @{
    bool init_assets(const char* resource_dir) {
        return engine_.init_assets(resource_dir);
    }
    bool load_musiccoca_model(const char* resource_dir, const char* subfolder);
    bool load_model(const char* mlxfn_path);
    bool load_prefill_model(const char* spectrostream_mlxfn_path,
                            const char* prefill_mlxfn_path);
    /// Stop the inference loop, encode + prefill the supplied audio (with
    /// 1 s trim each side), then restart the loop
    /// from the post-prefill state. The engine checkpoints on success —
    /// subsequent `reset()` calls return to this prefilled context, which
    /// makes "prefill once, try many prompts" trivial. The runner does
    /// NOT pre-load the ring buffer with prefill-loop audio (which is
    /// simply the reconstruction of the input: audio => tokens => audio);
    /// instead the standard 3-frame priming kicks in on restart.
    bool prefill_state(const float* audio_samples, int num_samples,
                       std::function<void(const std::string&)> log_callback = nullptr);
    /// Stop the inference loop, reset the model state, mask MusicCoCa, prefill
    /// `duration_frames` of cached silent RVQ tokens, and restart. Default
    /// duration is enough to fully saturate every layer's attention
    /// window so any prior generation no longer influences output. Like
    /// `prefill_state`, this checkpoints — `reset()` returns to the silent
    /// state. See `MLXEngine::prefill_silence` for the cached-silent-token
    /// implementation details.
    bool prefill_silence(int duration_frames = 550,
                         std::function<void(const std::string&)> log_callback = nullptr);
    void unload();
    void start();
    void stop();
    /// @}

    /// @name Audio output (audio thread) @{

    /// Pull `count` stereo samples into `destL` / `destR`. Applies bypass,
    /// volume / mute smoothing, reset envelope, and (if enabled) the MIDI
    /// gate envelope. Returns `false` if the ring buffer underran (the caller
    /// still gets `count` samples, zero-padded).
    ///
    /// `blocking=true` waits up to one ring-buffer worth of samples — intended
    /// for offline render only. Never pass `true` from the audio callback.
    bool read_audio_stereo(float* destL, float* destR, std::size_t count,
                           bool blocking = false);
    /// @}

    /// @name Sampling parameters (forward to `MLXEngine`)
    /// Atomic — safe from any thread. @{
    void set_temperature(float t) { engine_.set_temperature(t); }
    float get_temperature() const { return engine_.get_temperature(); }
    void set_top_k(int k) { engine_.set_top_k(k); }
    int get_top_k() const { return engine_.get_top_k(); }

    void set_cfg_musiccoca(float v) { engine_.set_cfg_musiccoca(v); }
    float get_cfg_musiccoca() const { return engine_.get_cfg_musiccoca(); }
    void set_cfg_notes(float v) { engine_.set_cfg_notes(v); }
    float get_cfg_notes() const { return engine_.get_cfg_notes(); }
    void set_cfg_drums(float v) { engine_.set_cfg_drums(v); }
    float get_cfg_drums() const { return engine_.get_cfg_drums(); }
    void set_unmask_width(int w) { engine_.set_unmask_width(w); }
    int get_unmask_width() const { return engine_.get_unmask_width(); }
    void set_seed_rotation(int r) { engine_.set_seed_rotation(r); }
    int get_seed_rotation() const { return engine_.get_seed_rotation(); }
    /// @}

    /// @name MIDI notes
    /// Additionally tracks active-note count for the MIDI-gate envelope. @{
    void set_note_on(int n);
    void set_note_off(int n);
    /// @}

    /// @name Drumless (thin forwarders) @{
    void set_drumless(bool on) { engine_.set_drumless(on); }
    bool get_drumless() const { return engine_.get_drumless(); }
    void set_onset_mode(int mode) { engine_.set_onset_mode(mode); }
    int get_onset_mode() const { return engine_.get_onset_mode(); }
    /// @}

    /// @name Prompts (forwarders — set_audio_* also triggers prompt surface re-blend) @{
    void set_text_prompt(const std::string& t) { engine_.set_text_prompt(t); }
    void set_text_prompts(const std::vector<std::string>& t, const std::vector<float>& w) { engine_.set_text_prompts(t, w); }

    // TODO(public-release): this is a placeholder — when `path` is non-empty it
    // writes a deterministic fake embedding rather than decoding the file.
    // Either wire this to `set_audio_prompt_samples` (after loading + resampling
    // the audio) or remove the stub so callers don't silently get fake data.
    void set_audio_prompt(int index, const std::string& path);
    void set_audio_embedding(int index, const float* embedding);
    void set_audio_prompt_samples(int index, const std::string& filename,
                                  const float* samples, std::size_t count) {
        engine_.set_audio_prompt_samples(index, filename, samples, count);
    }
    std::string get_cached_text(int index) { return engine_.get_cached_text(index); }
    int get_text_encoder_status() const { return engine_.get_text_encoder_status(); }
    int get_prompt_status(int index) const { return engine_.get_prompt_status(index); }
    int get_quantizer_status() const { return engine_.get_quantizer_status(); }
    std::vector<std::string> get_logs() { return engine_.get_logs(); }
    bool get_audio_embedding(int index, float* out) const {
        return engine_.get_audio_embedding(index, out);
    }
    bool reblend_musiccoca_tokens(const float* weights, int count, const float* pca_coeffs = nullptr, int pca_count = 0) {
        return engine_.reblend_musiccoca_tokens(weights, count, pca_coeffs, pca_count);
    }
    /// @}

    /// @name Output control @{
    void set_volume_db(float v) { volume_db_.store(v, std::memory_order_relaxed); }
    float get_volume_db() const { return volume_db_.load(std::memory_order_relaxed); }
    void set_mute(bool m) { mute_.store(m, std::memory_order_relaxed); }
    bool get_mute() const { return mute_.load(std::memory_order_relaxed); }
    void set_latency_comp(bool c) { latency_comp_.store(c, std::memory_order_relaxed); }
    bool get_latency_comp() const { return latency_comp_.load(std::memory_order_relaxed); }
    void set_midi_gate_enabled(bool e) { midi_gate_enabled_.store(e, std::memory_order_relaxed); }
    bool get_midi_gate_enabled() const { return midi_gate_enabled_.load(std::memory_order_relaxed); }
    /// @}

    /// @name Blend weights
    /// Explicit per-prompt blend weights (0–1, should sum to 1). Callers are
    /// responsible for computing these — either from IDW (surface mode in the
    /// frontend) or from user sliders (list mode). Bumps
    /// `position_generation_` so the inference thread picks up the change on
    /// its next frame. All setters are automatable. @{
    void set_blend_weight(int i, float w) {
        if (i >= 0 && i < (int)kMaxPrompts) {
            blend_weights_[i].store(w, std::memory_order_relaxed);
            position_generation_.fetch_add(1, std::memory_order_relaxed);
        }
    }
    float get_blend_weight(int i) const {
        return (i >= 0 && i < (int)kMaxPrompts) ? blend_weights_[i].load(std::memory_order_relaxed) : 0.0f;
    }
    /// Batch-set all weights in one call (bumps generation once).
    void set_blend_weights(const float* weights, int count) {
        for (int i = 0; i < (int)kMaxPrompts; ++i) {
            blend_weights_[i].store(i < count ? weights[i] : 0.0f, std::memory_order_relaxed);
        }
        position_generation_.fetch_add(1, std::memory_order_relaxed);
    }
    /// @}

    /// @name PCA controls
    /// Signed [-1, +1] coefficients per PCA component; applied to prompt slots
    /// whose text is `"pca"`. @{
    void set_pca_coeff(int i, float v) {
        if (i >= 0 && i < (int)kMaxPCAComponents) {
            pca_coeff_[i].store(v, std::memory_order_relaxed);
            position_generation_.fetch_add(1, std::memory_order_relaxed);
        }
    }
    float get_pca_coeff(int i) const {
        return (i >= 0 && i < (int)kMaxPCAComponents) ? pca_coeff_[i].load(std::memory_order_relaxed) : 0.0f;
    }

    bool is_loaded() const { return engine_.is_loaded(); }
    bool load_pca_file(const char* path) { return engine_.load_pca_file(path); }
    bool is_pca_loaded() const { return engine_.is_pca_loaded(); }
    int pca_component_count() const { return engine_.pca_component_count(); }
    int pca_centroid_count() const { return engine_.pca_centroid_count(); }
    /// @}

    /// @name Bypass & reset @{
    void set_bypass(bool b) { bypass_.store(b, std::memory_order_relaxed); }
    bool get_bypass() const { return bypass_.load(std::memory_order_relaxed); }
    void set_host_bypass(bool b) { host_bypass_.store(b, std::memory_order_relaxed); }
    bool get_host_bypass() const { return host_bypass_.load(std::memory_order_relaxed); }

    /// Rising-edge **user-initiated** reset trigger consumed by the inference
    /// loop. The reset envelope fades the next frame in to avoid a click.
    /// Use this for any reset whose source is the user's intent — clicking
    /// the Reset Model parameter, hitting Play in a host app, etc.
    /// **Never suppressed by prefill state**, even if a prefill just
    /// completed.
    void trigger_reset() {
        reset_request_.store(ResetRequest::User, std::memory_order_relaxed);
        reset_env_.value.store(0.0f, std::memory_order_relaxed);
    }
    /// Rising-edge **transport-initiated** reset trigger (e.g. DAW transport
    /// rewinds to beat 0). Same effect as `trigger_reset()` *unless* the
    /// engine is in the post-prefill grace window: prefill arms a one-shot
    /// "skip the next transport reset" so the user's freshly-prefilled
    /// context survives a DAW re-cue. Has no effect on a pending user
    /// reset (won't downgrade `User` to `Transport`).
    void trigger_transport_reset() {
        ResetRequest expected = ResetRequest::None;
        reset_request_.compare_exchange_strong(expected, ResetRequest::Transport,
                                                std::memory_order_relaxed);
        reset_env_.value.store(0.0f, std::memory_order_relaxed);
    }

    /// The user's buffer size (latency). With host sync on, the ring may hold
    /// more (kSyncLeadSamples) so a re-prime never runs it dry; the reported
    /// latency stays this value -- the sync maps the host's beat onto the
    /// samples as they are read, so the lead does not shift the alignment.
    void set_buffer_size(std::size_t cap) {
        user_buffer_size_.store(cap, std::memory_order_relaxed);
        apply_ring_capacity();
    }
    std::size_t get_buffer_size() const { return user_buffer_size_.load(std::memory_order_relaxed); }
    std::size_t get_latency_samples() const {
        return get_latency_comp() ? get_buffer_size() : 0;
    }

    /// Audio-thread safe: called on DAW stopped→playing transition. Drains
    /// the ring buffers without disturbing the inference thread's write cursor.
    /// With host sync on the buffered lead is kept (a re-prime may need it to
    /// re-align to the new transport position right away).
    void reset_for_playback() {
        if (get_host_sync()) return;
        ring_L_.drain();
        ring_R_.drain();
    }

    /// Full reset: stops the inference thread, resets model state, clears the
    /// ring buffers, then restarts if we were running.
    void reset();
    /// @}

    /// @name Host tempo / downbeat sync (tempo-sync fork)
    /// MRT2 follows the host's tempo and puts its downbeats on the host's bars
    /// by re-priming its memory from time-stretched copies of its own recent
    /// output (see magentart/tempo/tempo_sync.h). Needs the tempo export
    /// (`magenta_rt.mlx.export_tempo`): a step with MRT2's beat read-out plus
    /// `<model>_reprime.mlxfn` and `<model>_encoder.mlxfn`. @{
    bool load_tempo_models(const char* reprime_mlxfn_path, const char* encoder_mlxfn_path,
                           int context_frames = 150);
    /// Models loaded and the step reports MRT2's beat read-out.
    bool tempo_sync_available() const { return sync_->available(); }
    void set_host_sync(bool on) {
        sync_->set_enabled(on);
        apply_ring_capacity();
    }
    bool get_host_sync() const { return sync_->enabled(); }
    /// Keep MRT2's downbeats on the host's bars (else: tempo only).
    void set_bar_lock(bool on) { sync_->set_bar_lock(on); }
    bool get_bar_lock() const { return sync_->bar_lock(); }
    /// Keep the explicit musiccoca CFG while synced (default on; off = only
    /// the distilled CFG tokens, less pull toward the prompt's own tempo).
    void set_sync_keep_cfg(bool on) { sync_->set_keep_cfg(on); }
    bool get_sync_keep_cfg() const { return sync_->keep_cfg(); }
    /// Audio thread, once per render call before reading audio: the host's
    /// transport at the first sample of this buffer (`beat_position` =
    /// AUHostMusicalContextBlock's currentBeatPosition, `measure_downbeat` =
    /// currentMeasureDownbeatPosition). The output the host receives now is
    /// placed `latency_seconds` earlier on its timeline (PDC), so it carries
    /// beat `beat_position - latency * tempo / 60`. Lock-free.
    void set_host_transport(double tempo_bpm, double beat_position, double measure_downbeat,
                            int beats_per_bar, bool playing, double latency_seconds);
    tempo::SyncStatus get_sync_status() const { return sync_->status(); }
    /// Buffered audio the ring keeps with host sync on, and the part of it a
    /// re-prime may block for.
    static constexpr std::size_t kSyncLeadSamples = 24000;     // 0.5 s
    static constexpr std::size_t kReprimeLeadSamples = 20160;  // 0.42 s
    /// @}

    /// @name State persistence @{
    bool save_state(const char* path);
    bool load_state(const char* path);
    /// Restore the engine's reset target to the model's factory initial state
    /// (the `<model>_state.safetensors` payload), undoing any prefill-induced
    /// or `load_state`-induced checkpoint, and apply it to the live state.
    void reset_to_factory();
    /// @}

    /// @name Audio Recording @{
    void start_recording();
    void stop_recording();
    void clear_recording();
    bool get_recorded_audio(float* destL, float* destR, std::size_t start_idx, std::size_t count) const;
    std::size_t get_recorded_sample_count() const { return recorded_samples_.load(std::memory_order_relaxed); }
    /// Calculates a reduced visual representation of the recorded audio.
    /// Divides the recorded samples into 'num_buckets' chunks and retrieves the maximum
    /// absolute amplitude (peak) for each chunk. Used to efficiently stream waveform data
    /// to the React WebView without overloading the JavaScript IPC bridge.
    std::vector<float> get_waveform_peaks(int num_buckets) const;
    /// @}

    /// @name Host integration & metrics @{
    void set_transport_flags(int flags) { transport_flags_.store(flags, std::memory_order_relaxed); }
    void set_offline(bool v) { is_offline_.store(v, std::memory_order_relaxed); }
    EngineMetrics get_metrics() const;
    /// Clear the cumulative dropped-frame (real-time underrun) tally. Call from
    /// the UI/controller thread after surfacing the count to the user.
    void reset_dropped_frames() { dropped_frame_count_.store(0, std::memory_order_relaxed); }
    /// @}

private:
    void start_locked(bool skip_reset = false);
    void stop_locked();
    void inference_loop();
    /// One frame through the tempo-sync controller (tracks MRT2's beat; when
    /// host sync is on, may re-prime first). Inference thread.
    bool generate_one();
    void apply_ring_capacity();
    tempo::HostTransport read_host_transport() const;

    MLXEngine engine_{};
    float audio_buf_L_[kFrameSamples]{};
    float audio_buf_R_[kFrameSamples]{};

    std::vector<float> recording_buf_L_;
    std::vector<float> recording_buf_R_;
    std::atomic<bool> is_recording_{false};
    std::atomic<std::size_t> recorded_samples_{0};

    RingBuffer ring_L_;
    RingBuffer ring_R_;
    std::atomic<float> volume_db_{0.0f};
    std::atomic<bool> mute_{false};
    std::atomic<bool> latency_comp_{false};
    float smoothed_gain_{1.0f};
    std::atomic<bool> midi_gate_enabled_{false};
    std::array<std::atomic<bool>, 128> note_states_{};
    std::atomic<int> active_note_count_{0};
    ExponentialEnvelope reset_env_;
    ExponentialEnvelope midi_env_;
    std::atomic<float> frame_total_ms_{0};
    std::atomic<std::uint64_t> dropped_frame_count_{0};
    std::atomic<int> transport_flags_{-1};
    std::atomic<bool> is_offline_{false};
    std::atomic<bool> running_{false};
    std::thread thread_;
    std::mutex lifecycle_mutex_;

    // Blend weights (set by frontend, consumed by inference loop)
    std::atomic<float> blend_weights_[kMaxPrompts] = {};
    std::atomic<float> pca_coeff_[kMaxPCAComponents] = {};
    mutable std::atomic<std::uint32_t> position_generation_{0};
    std::uint32_t last_blended_generation_ = 0;

    std::atomic<bool> bypass_{false};
    std::atomic<bool> host_bypass_{false};

    // Host tempo sync. The host transport is published by the audio thread
    // through a seqlock (every field atomic; `host_seq_` odd while writing).
    std::unique_ptr<tempo::TempoSyncController> sync_;
    std::atomic<std::size_t> user_buffer_size_{2048};
    std::atomic<std::uint32_t> host_seq_{0};
    std::atomic<bool> host_valid_{false};
    std::atomic<double> host_tempo_{0.0};
    std::atomic<double> host_beat_{0.0};
    std::atomic<double> host_bar_{0.0};
    std::atomic<int> host_bpb_{4};
    std::atomic<std::uint64_t> host_cursor_{0};
    std::atomic<bool> host_playing_{false};

    /// Pending reset request. Packed into a single atomic so the inference
    /// loop reads the request kind in one operation (no TOCTOU window
    /// between checking "is something pending?" and "what kind?"). The
    /// inference loop exchanges this back to `None` when consuming.
    /// Producers: `trigger_reset()` writes `User` unconditionally;
    /// `trigger_transport_reset()` writes `Transport` only if currently
    /// `None` (so it can't downgrade a pending user reset).
    enum class ResetRequest : int { None = 0, User = 1, Transport = 2 };
    std::atomic<ResetRequest> reset_request_{ResetRequest::None};
    /// Armed by `prefill_state` / `prefill_silence` on success. Consumed
    /// by the inference loop's reset handler **only when the request kind
    /// is `Transport`** — user-initiated resets always fire regardless.
    /// One-shot: the inference loop clears it when it suppresses a
    /// transport reset.
    std::atomic<bool> skip_next_transport_reset_{false};
};

}  // namespace core
}  // namespace magentart
