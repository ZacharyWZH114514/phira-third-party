//! `android.media.AudioTrack` output backend.
//!
//! The other Android backend in this crate goes through oboe, which drags in a
//! C++ dependency (and AAudio / OpenSL ES underneath it). This one speaks to
//! the framework directly over JNI: it builds a streaming `AudioTrack` with
//! float PCM, feeds it from a dedicated render thread with blocking writes and
//! keeps the mixer's sample rate in sync with the device, so the game clock
//! stays tied to what the speaker actually plays.
//!
//! The track is created on the caller's thread (which is attached to the JVM
//! already) so that a broken device reports through [`Backend::start`] instead
//! of vanishing into the render thread.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use anyhow::{anyhow, bail, Result};
use jni::{
    jni_sig, jni_str,
    objects::{JFloatArray, JObject},
    refs::Global,
    vm::JavaVM,
    Env, JValue,
};
use parking_lot::Mutex;

use super::{BackendSetup, State};
use crate::Backend;

// ---- android.media.AudioTrack / AudioFormat / AudioAttributes constants ----

/// `AudioFormat.ENCODING_PCM_FLOAT`
const ENCODING_PCM_FLOAT: i32 = 4;
/// `AudioFormat.CHANNEL_OUT_STEREO`
const CHANNEL_OUT_STEREO: i32 = 0xc;
/// `AudioTrack.MODE_STREAM`
const MODE_STREAM: i32 = 1;
/// `AudioManager.STREAM_MUSIC`
const STREAM_MUSIC: i32 = 3;
/// `AudioAttributes.USAGE_GAME`
const USAGE_GAME: i32 = 14;
/// `AudioAttributes.CONTENT_TYPE_MUSIC`
const CONTENT_TYPE_MUSIC: i32 = 2;
/// `AudioTrack.WRITE_BLOCKING`
const WRITE_BLOCKING: i32 = 0;
/// `AudioTrack.PERFORMANCE_MODE_LOW_LATENCY` (API 26+)
const PERFORMANCE_MODE_LOW_LATENCY: i32 = 1;
/// `AudioManager.AUDIO_SESSION_ID_GENERATE`
const AUDIO_SESSION_ID_GENERATE: i32 = 0;
/// `AudioTrack.STATE_INITIALIZED`
const STATE_INITIALIZED: i32 = 1;
/// `android.os.Process.THREAD_PRIORITY_URGENT_AUDIO`
const THREAD_PRIORITY_URGENT_AUDIO: i32 = -19;

static DEVICE_RATE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Output rate the framework negotiated for the last track we opened.
pub fn device_sample_rate() -> u32 {
    DEVICE_RATE.load(std::sync::atomic::Ordering::Relaxed)
}

const DEFAULT_SAMPLE_RATE: u32 = 48_000;
/// Frames rendered per `write` when the user did not pick a buffer size.
const DEFAULT_CHUNK_FRAMES: u32 = 256;
/// Anything below this starves the thread between two blocking writes.
const MIN_CHUNK_FRAMES: u32 = 64;
/// `f32` stereo
const BYTES_PER_FRAME: i32 = std::mem::size_of::<f32>() as i32 * 2;
/// How many chunks the track's internal buffer holds. More chunks absorb
/// scheduler hiccups, fewer lower the latency.
const BUFFER_CHUNKS: i32 = 3;
/// How much of that buffer we keep in use for the low latency path. One chunk
/// would be the minimum, but then a single late wakeup is an audible dropout.
const LOW_LATENCY_BUFFER_CHUNKS: i32 = 2;
/// How long a stop waits for the render thread before leaving it detached.
const STOP_TIMEOUT: Duration = Duration::from_millis(120);
const STOP_POLL_INTERVAL: Duration = Duration::from_millis(2);

#[derive(Debug, Clone, Default)]
pub struct AudioTrackSettings {
    /// Frames per rendered chunk. `None` uses [`DEFAULT_CHUNK_FRAMES`].
    pub buffer_size: Option<u32>,
    /// Ask the framework for `PERFORMANCE_MODE_LOW_LATENCY` (API 26+).
    pub low_latency: bool,
    /// Force an output sample rate. `None` follows the device native rate.
    pub sample_rate: Option<u32>,
}

pub struct AudioTrackBackend {
    settings: AudioTrackSettings,
    state: Option<Arc<Mutex<State>>>,
    broken: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl AudioTrackBackend {
    pub fn new(settings: AudioTrackSettings) -> Self {
        Self {
            settings,
            state: None,
            broken: Arc::default(),
            running: Arc::default(),
            thread: None,
        }
    }

    /// Signal the render thread and wait for it to release the track.
    ///
    /// The join is bounded: the thread can be parked inside a blocking
    /// `write`, and blocking the caller (which may be the UI/game loop) while
    /// the audio device decides to answer is not acceptable. If it did not
    /// come back in time we leave it detached --it owns its own stop flag, so
    /// it will notice, release its track and exit on its own.
    fn stop_thread(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        let Some(thread) = self.thread.take() else {
            return;
        };
        let deadline = Instant::now() + STOP_TIMEOUT;
        while !thread.is_finished() && Instant::now() < deadline {
            thread::sleep(STOP_POLL_INTERVAL);
        }
        if thread.is_finished() {
            let _ = thread.join();
        }
    }
}

impl Drop for AudioTrackBackend {
    fn drop(&mut self) {
        self.stop_thread();
    }
}

impl Backend for AudioTrackBackend {
    fn setup(&mut self, setup: BackendSetup) -> Result<()> {
        self.state = Some(Arc::new(Mutex::new(setup.into())));
        Ok(())
    }

    fn start(&mut self) -> Result<()> {
        self.stop_thread();

        let state = Arc::clone(
            self.state
                .as_ref()
                .ok_or_else(|| anyhow!("audio backend was not set up"))?,
        );

        // Build the track here rather than on the render thread: this thread is
        // attached to the JVM already, and any failure is reported as an `Err`
        // from `start` (which the caller turns into a visible error message).
        let vm = JavaVM::singleton().map_err(|err| anyhow!("cannot get the JavaVM: {err:?}"))?;
        let (track, sample_rate) =
            vm.attach_current_thread(|env| -> Result<_> { create_track(env, &self.settings) })?;

        DEVICE_RATE.store(sample_rate, std::sync::atomic::Ordering::Relaxed);
    state.lock().mixer.sample_rate = sample_rate;

        // A fresh flag per run: a thread that is still winding down (see
        // `stop_thread`) must not be resurrected by the next `start`.
        let running = Arc::new(AtomicBool::new(true));
        self.running = Arc::clone(&running);
        let broken = Arc::clone(&self.broken);
        let spawned = thread::Builder::new()
            .name("audiotrack".into())
            .spawn(move || {
                let result = vm.attach_current_thread(|env| -> Result<()> {
                    let result = render_loop(env, state, track, &running);
                    if let Err(err) = &result {
                        log_info(env, &format!("render loop failed: {err:#}"));
                    }
                    result
                });
                if result.is_err() {
                    broken.store(true, Ordering::Relaxed);
                }
            })?;
        self.thread = Some(spawned);
        Ok(())
    }

    fn consume_broken(&self) -> bool {
        self.broken.fetch_and(false, Ordering::Relaxed)
    }
}

/// Everything the render thread owns. All Java objects are global references so
/// they survive the thread that created them.
struct Track {
    track: Global<JObject<'static>>,
    samples: Global<JFloatArray<'static>>,
    /// Frames per rendered chunk.
    chunk_frames: usize,
    sample_rate: u32,
}

impl Track {
    /// Push one chunk of interleaved stereo samples. Returns the number of
    /// frames the framework accepted.
    fn write(&self, env: &mut Env, data: &[f32]) -> Result<usize> {
        // `set_region` borrows the env immutably, the call below needs it mutably.
        self.samples.set_region(env, 0, data)?;

        let track: &JObject = &self.track;
        let samples: &JObject = &self.samples;
        let written = env
            .call_method(
                track,
                jni_str!("write"),
                jni_sig!("([FIII)I"),
                &[
                    JValue::Object(samples),
                    JValue::Int(0),
                    JValue::Int(data.len().min(i32::MAX as usize) as i32),
                    JValue::Int(WRITE_BLOCKING),
                ],
            )?
            .i()?;
        if written < 0 {
            bail!("AudioTrack.write failed with {written}");
        }
        // `write` counts floats, we hand back frames.
        Ok(written as usize / 2)
    }

    /// Frames played since the track was created.
    fn playback_head(&self, env: &mut Env) -> Result<i32> {
        let track: &JObject = &self.track;
        Ok(env
            .call_method(track, jni_str!("getPlaybackHeadPosition"), jni_sig!("()I"), &[])?
            .i()?)
    }

    fn stop(&self, env: &mut Env) {
        let track: &JObject = &self.track;
        let _ = env.call_method(track, jni_str!("stop"), jni_sig!("()V"), &[]);
        let _ = env.call_method(track, jni_str!("release"), jni_sig!("()V"), &[]);
    }
}

fn render_loop(
    env: &mut Env,
    state: Arc<Mutex<State>>,
    track: Track,
    running: &AtomicBool,
) -> Result<()> {
    // This thread is what feeds the speaker: keep the scheduler off our back.
    let _ = env.call_static_method(
        jni_str!("android/os/Process"),
        jni_str!("setThreadPriority"),
        jni_sig!("(I)V"),
        &[JValue::Int(THREAD_PRIORITY_URGENT_AUDIO)],
    );

    let result = (|| -> Result<()> {
        let mut samples = vec![0f32; track.chunk_frames * 2];
        let mut frames_written: u64 = 0;
        let mut first_chunk = true;
        while running.load(Ordering::Relaxed) {
            {
                let mut state = state.lock();
                // Keep the mixer pinned to the rate the track actually runs at;
                // anything that resets it would make every renderer play at the
                // wrong speed (fast + pitch shifted).
                state.mixer.sample_rate = track.sample_rate;
                state.mixer.render_stereo(&mut samples);
            }

            let mut rest = &samples[..];
            while !rest.is_empty() {
                let frames = track.write(env, rest)?.min(rest.len() / 2);
                if frames == 0 {
                    // The framework asked us to back off.
                    thread::sleep(std::time::Duration::from_millis(1));
                    continue;
                }
                frames_written += frames as u64;
                rest = &rest[frames * 2..];
            }

            if first_chunk {
                first_chunk = false;
                log_info(env, "first chunk queued");
            }

            // What is queued but not played yet is the delay the player hears.
            // Both counters are 32-bit on the Java side and wrap, so compare
            // them modulo 2^32.
            if let Ok(head) = track.playback_head(env) {
                let queued = (frames_written as u32).wrapping_sub(head as u32) as f32;
                state
                    .lock()
                    .recorder
                    .push(queued / track.sample_rate as f32);
            }
        }
        Ok(())
    })();

    track.stop(env);
    result
}

/// Report through `android.util.Log`, because stderr from native code is
/// dropped on Android and a silent audio backend is impossible to debug.
fn log_info(env: &mut Env, message: &str) {
    let Ok(tag) = env.new_string("PhiraAudio") else {
        return;
    };
    let Ok(msg) = env.new_string(message) else {
        return;
    };
    let tag: &JObject = &tag;
    let msg: &JObject = &msg;
    let _ = env.call_static_method(
        jni_str!("android/util/Log"),
        jni_str!("i"),
        jni_sig!("(Ljava/lang/String;Ljava/lang/String;)I"),
        &[JValue::Object(tag), JValue::Object(msg)],
    );
}

fn create_track(env: &mut Env, settings: &AudioTrackSettings) -> Result<(Track, u32)> {
    let sample_rate = match settings.sample_rate {
        Some(rate) => rate,
        None => native_sample_rate(env).unwrap_or(DEFAULT_SAMPLE_RATE),
    };
    let chunk_frames = settings
        .buffer_size
        .unwrap_or(DEFAULT_CHUNK_FRAMES)
        .max(MIN_CHUNK_FRAMES) as usize;

    let min_bytes = env
        .call_static_method(
            jni_str!("android/media/AudioTrack"),
            jni_str!("getMinBufferSize"),
            jni_sig!("(III)I"),
            &[
                JValue::Int(sample_rate as i32),
                JValue::Int(CHANNEL_OUT_STEREO),
                JValue::Int(ENCODING_PCM_FLOAT),
            ],
        )?
        .i()?;
    if min_bytes <= 0 {
        bail!("AudioTrack.getMinBufferSize rejected {sample_rate} Hz float stereo ({min_bytes})");
    }
    let chunk_bytes = chunk_frames as i32 * BYTES_PER_FRAME;
    let buffer_bytes = min_bytes.max(chunk_bytes * BUFFER_CHUNKS);

    let attributes = build_attributes(env)?;
    let format = build_format(env, sample_rate)?;

    let attributes_obj: &JObject = &attributes;
    let format_obj: &JObject = &format;
    let track = env.new_object(
        jni_str!("android/media/AudioTrack"),
        jni_sig!("(Landroid/media/AudioAttributes;Landroid/media/AudioFormat;III)V"),
        &[
            JValue::Object(attributes_obj),
            JValue::Object(format_obj),
            JValue::Int(buffer_bytes),
            JValue::Int(MODE_STREAM),
            JValue::Int(AUDIO_SESSION_ID_GENERATE),
        ],
    )?;

    let state = env
        .call_method(&track, jni_str!("getState"), jni_sig!("()I"), &[])?
        .i()?;
    if state != STATE_INITIALIZED {
        bail!("AudioTrack did not initialize (state = {state})");
    }

    let track_obj: &JObject = &track;
    if settings.low_latency {
        // Both calls are API 23 / 26 and are best-effort: older devices simply
        // keep the default performance mode and buffer.
        let _ = env.call_method(
            track_obj,
            jni_str!("setPerformanceMode"),
            jni_sig!("(I)I"),
            &[JValue::Int(PERFORMANCE_MODE_LOW_LATENCY)],
        );
        let _ = env.call_method(
            track_obj,
            jni_str!("setBufferSizeInFrames"),
            jni_sig!("(I)I"),
            &[JValue::Int(chunk_frames as i32 * LOW_LATENCY_BUFFER_CHUNKS)],
        );
    }

    let buffer_frames = env
        .call_method(track_obj, jni_str!("getBufferSizeInFrames"), jni_sig!("()I"), &[])?
        .i()?;

    env.call_method(track_obj, jni_str!("play"), jni_sig!("()V"), &[])?;

    let samples = env.new_float_array(chunk_frames * 2)?;
    let message = format!(
        "init: {sample_rate} Hz, {chunk_frames} frames per write, \
         {buffer_frames} frames buffered, low_latency={}, \
         performance_mode={:?}",
        settings.low_latency,
        env.call_method(track_obj, jni_str!("getPerformanceMode"), jni_sig!("()I"), &[])
            .ok()
            .and_then(|v| v.i().ok()),
    );
    log_info(env, &message);
    Ok((
        Track {
            track: env.new_global_ref(&track)?,
            samples: env.new_global_ref(&samples)?,
            chunk_frames,
            sample_rate,
        },
        sample_rate,
    ))
}

/// The device's native output rate, so the framework does not have to resample
/// everything we write. Deprecated upstream but still the only way to ask
/// without a `Context`.
fn native_sample_rate(env: &mut Env) -> Result<u32> {
    let rate = env
        .call_static_method(
            jni_str!("android/media/AudioTrack"),
            jni_str!("getNativeOutputSampleRate"),
            jni_sig!("(I)I"),
            &[JValue::Int(STREAM_MUSIC)],
        )?
        .i()?;
    if rate <= 0 {
        bail!("AudioTrack.getNativeOutputSampleRate returned {rate}");
    }
    Ok(rate as u32)
}

fn build_attributes<'local>(env: &mut Env<'local>) -> Result<JObject<'local>> {
    let builder = env.new_object(
        jni_str!("android/media/AudioAttributes$Builder"),
        jni_sig!("()V"),
        &[],
    )?;
    env.call_method(
        &builder,
        jni_str!("setUsage"),
        jni_sig!("(I)Landroid/media/AudioAttributes$Builder;"),
        &[JValue::Int(USAGE_GAME)],
    )?;
    env.call_method(
        &builder,
        jni_str!("setContentType"),
        jni_sig!("(I)Landroid/media/AudioAttributes$Builder;"),
        &[JValue::Int(CONTENT_TYPE_MUSIC)],
    )?;
    Ok(env
        .call_method(
            &builder,
            jni_str!("build"),
            jni_sig!("()Landroid/media/AudioAttributes;"),
            &[],
        )?
        .l()?)
}

fn build_format<'local>(env: &mut Env<'local>, sample_rate: u32) -> Result<JObject<'local>> {
    let builder = env.new_object(
        jni_str!("android/media/AudioFormat$Builder"),
        jni_sig!("()V"),
        &[],
    )?;
    env.call_method(
        &builder,
        jni_str!("setEncoding"),
        jni_sig!("(I)Landroid/media/AudioFormat$Builder;"),
        &[JValue::Int(ENCODING_PCM_FLOAT)],
    )?;
    env.call_method(
        &builder,
        jni_str!("setSampleRate"),
        jni_sig!("(I)Landroid/media/AudioFormat$Builder;"),
        &[JValue::Int(sample_rate as i32)],
    )?;
    env.call_method(
        &builder,
        jni_str!("setChannelMask"),
        jni_sig!("(I)Landroid/media/AudioFormat$Builder;"),
        &[JValue::Int(CHANNEL_OUT_STEREO)],
    )?;
    Ok(env
        .call_method(
            &builder,
            jni_str!("build"),
            jni_sig!("()Landroid/media/AudioFormat;"),
            &[],
        )?
        .l()?)
}
