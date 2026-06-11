# WhoSpoke

WhoSpoke is a multimodal speaker identification and transcription library for video. It is designed to answer the practical research question: **who spoke, when, and what did they say?**

The package combines automatic speech recognition, speaker diarization, face recognition, active-speaker detection, voice matching, and evidence fusion to produce named, timestamped transcripts with transparent confidence scores.

> **Current status:** WhoSpoke currently provides a working package skeleton, validated configuration models, CLI commands, export schemas, and a backend registry. The heavy machine-learning backends are represented as configurable options and can be integrated behind the existing interfaces. The default pipeline returns a mock result so users can test installation, configuration, exports, and downstream workflows before installing GPU-heavy dependencies.

---

## Why WhoSpoke?

Most diarization tools answer only **“who spoke when?”** by assigning anonymous labels such as `SPEAKER_00` and `SPEAKER_01`. They do not know whether `SPEAKER_00` is Emmanuel Macron, Olaf Scholz, a journalist, or a moderator.

Face recognition can identify visible people, but it does not know whether the visible person is currently speaking. Voice matching can identify known voices, but it can struggle with noisy audio, overlapping speech, and short segments. Active-speaker detection can tell which face is speaking, but it needs identity labels. WhoSpoke is designed to fuse all of these signals:

- ASR: what was said?
- Diarization: when did speaker turns occur?
- Face recognition: who is visible?
- Voice matching: whose voice does this sound like?
- Active-speaker detection: which visible face is speaking?
- Fusion: which identity is most strongly supported by the evidence?

---

## Supported algorithm families

WhoSpoke exposes a registry of algorithm families so users can choose the most appropriate solution for their hardware, data quality, language coverage, and accuracy requirements.

### 1. ASR / transcription backends

Select with `asr.backend` or the CLI flag `--asr-backend`.

| Backend | Suggested use |
| --- | --- |
| `whisper` | Multilingual transcription and translation; strong default for noisy, accented, or code-switched speech. |
| `faster_whisper` | Faster Whisper inference through CTranslate2-style deployments. |
| `kaldi` | Highly configurable research-grade ASR pipelines. |
| `vosk` | Lightweight offline and streaming ASR for CPU/embedded use. |
| `deepspeech` | RNN/CTC-based ASR for simpler or legacy workflows. |
| `wav2letter` | Fast convolutional end-to-end ASR. |
| `open_seq2seq` | Sequence-to-sequence toolkit for ASR and related tasks. |
| `wav2vec` | Self-supervised speech representation and ASR finetuning. |
| `mock` | Default placeholder backend for tests and demos. |

Important adjustable ASR parameters include:

```yaml
asr:
  backend: whisper
  model_name: large
  language: null
  task: transcribe
  word_timestamps: true
  beam_size: 5
  best_of: 5
  temperature: [0.0, 0.2, 0.4, 0.6]
  compression_ratio_threshold: 2.4
  log_prob_threshold: -1.0
  no_speech_threshold: 0.6
```

### 2. Speaker diarization algorithms

Select with `diarization.algorithm` or the CLI flag `--diarization-algorithm`.

| Algorithm | Suggested use |
| --- | --- |
| `pyannote` | Modular neural segmentation + embedding + clustering pipeline. |
| `uis_rnn` | Supervised recurrent speaker diarization / online clustering. |
| `eend` | End-to-end neural diarization, useful for overlapping speech. |
| `vb_hmm` | Variational Bayes HMM re-clustering for classical diarization pipelines. |
| `spectral` | Embedding-based diarization with spectral clustering. |
| `kmeans` | Simple clustering when the number of speakers is known or estimated. |
| `affinity` | Affinity propagation clustering without pre-setting the number of speakers. |
| `mock` | Default no-op placeholder. |

Diarization has several configurable stages:

```yaml
diarization:
  algorithm: pyannote
  overlap: true
  segmentation:
    threshold: 0.55
    min_duration_on: 0.30
    min_duration_off: 0.20
    window_size: 5.0
    step_size: 0.5
    max_speakers_per_window: 4
  embedding:
    model: xvector
    normalize: true
    metric: cosine
  clustering:
    method: ahc
    threshold: 0.75
    num_speakers: null
    min_speakers: 2
    max_speakers: 6
    plda: false
  aggregation:
    gap: 0.30
    min_segment_duration: 0.20
    collar: 0.0
```

Supported clustering methods are `ahc`, `kmeans`, `spectral`, `affinity`, `uis_rnn`, `vb_hmm`, and `mock`.

### 3. Face recognition backbones

WhoSpoke is designed to integrate with DeepFace-style backbones. Select with `face.algorithm` or the CLI flag `--face-algorithm`.

| Backbone | Notes |
| --- | --- |
| `arcface` | Strong default; additive angular margin embedding model. |
| `facenet` / `facenet512` | Efficient face embeddings. |
| `vgg_face` | Heavier VGG-based face recognition. |
| `openface` | Lightweight real-time oriented face embeddings. |
| `deepface` | Original DeepFace model family. |
| `deepid` | Early deep CNN face verification family. |
| `dlib` | Dlib face recognition backend. |
| `sface` | Mobile/efficient face recognition. |
| `ghostfacenet` | Lightweight face-recognition model. |
| `buffalo_l` | High-performing InsightFace-style model. |
| `mock` | Placeholder face matcher. |

Example:

```yaml
face:
  algorithm: arcface
  detector_backend: retinaface
  align: true
  normalization: base
  distance_metric: cosine
  similarity_threshold: 0.67
  enforce_detection: false
  max_faces_per_frame: 5
```

### 4. Voice matching models

Select with `voice.algorithm` or `--voice-algorithm`.

| Backend | Notes |
| --- | --- |
| `resemblyzer` | GE2E-style 256-dimensional speaker embeddings; good lightweight default. |
| `xvector` | Widely used speaker embedding family. |
| `ecapa_tdnn` | Strong speaker verification architecture with channel attention. |
| `mock` | Placeholder voice matcher. |

Example:

```yaml
voice:
  algorithm: resemblyzer
  similarity_threshold: 0.75
  min_sample_duration: 2.0
  min_segment_duration: 1.0
  metric: cosine
```

### 5. Active-speaker detection

Select with `active_speaker.algorithm` or `--active-speaker-algorithm`.

| Backend | Notes |
| --- | --- |
| `fast_asd` | Optimized TalkNet-style active-speaker detection. |
| `talknet` | Audio-visual active-speaker detection model. |
| `mock` | Placeholder active-speaker detection. |

Example:

```yaml
active_speaker:
  algorithm: fast_asd
  threshold: 0.60
  frame_rate: 25
  face_detection_threshold: 0.70
```

### 6. Evidence fusion

Evidence fusion combines transcript timing, diarization clusters, face identity scores, voice identity scores, and active-speaker tracks.

```yaml
fusion:
  mode: balanced
  confidence_threshold: 0.60
  ambiguity_margin: 0.10
  face_similarity_threshold: 0.67
  voice_similarity_threshold: 0.78
  weights:
    asr: 1.0
    diarization: 1.0
    face: 1.0
    voice: 1.5
    active_speaker: 1.0
```

Use `voice` and `face` weights to control how strongly each modality contributes. For example, if the camera often cuts away from speakers, increase the `voice` weight; if the audio is noisy but faces are clear, increase the `face` and `active_speaker` weights.

---

## Installation

Base installation:

```bash
pip install WhoSpoke
```

Install optional backend families as needed:

```bash
pip install "WhoSpoke[asr-whisper]"
pip install "WhoSpoke[diarization-pyannote]"
pip install "WhoSpoke[face-deepface]"
pip install "WhoSpoke[voice-resemblyzer]"
pip install "WhoSpoke[active-speaker]"
```

Development installation:

```bash
git clone https://github.com/YOUR_USERNAME/WhoSpoke.git
cd WhoSpoke
pip install -e ".[dev]"
```

---

## Tutorial: diplomatic summit example

Imagine a video of a multilingual press conference where two leaders switch between French, German, and English. We want a timestamped transcript with named speakers.

### Our Subjects

For demonstration purposes, WhoSpoke can be initialized with reference portraits and short voice samples for each known speaker. In this example, we use two public political figures in a multilingual diplomatic summit scenario.

<p align="center">
  <img src="assets/macron.jpg" alt="Emmanuel Macron" width="240"/>
  <img src="assets/scholz.jpg" alt="Olaf Scholz" width="240"/>
</p>

<p align="center">
  <strong>Emmanuel Macron</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <strong>Olaf Scholz</strong>
</p>

### Step 1: Prepare `people.yaml`

```yaml
people:
  - id: emmanuel_macron
    name: Emmanuel Macron
    portraits:
      - assets/macron_1.jpg
      - assets/macron_2.jpg
    voice_samples:
      - assets/macron_voice_1.wav
      - assets/macron_voice_2.wav

  - id: olaf_scholz
    name: Olaf Scholz
    portraits:
      - assets/scholz_1.jpg
      - assets/scholz_2.jpg
    voice_samples:
      - assets/scholz_voice_1.wav
      - assets/scholz_voice_2.wav
```

### Step 2: Prepare `analysis.yaml`

```yaml
asr:
  backend: whisper
  model_name: large
  language: null
  task: transcribe
  beam_size: 5
  best_of: 5
  temperature: [0.0, 0.2, 0.4, 0.6]
  compression_ratio_threshold: 2.4
  log_prob_threshold: -1.0

diarization:
  algorithm: pyannote
  overlap: true
  segmentation:
    threshold: 0.55
    min_duration_on: 0.30
    min_duration_off: 0.20
  embedding:
    model: xvector
    normalize: true
    metric: cosine
  clustering:
    method: ahc
    threshold: 0.75
    min_speakers: 2
    max_speakers: 6
  aggregation:
    gap: 0.30

face:
  algorithm: arcface
  detector_backend: retinaface
  align: true
  distance_metric: cosine
  similarity_threshold: 0.67

voice:
  algorithm: resemblyzer
  similarity_threshold: 0.78
  min_sample_duration: 2.0
  min_segment_duration: 1.0

active_speaker:
  algorithm: fast_asd
  threshold: 0.60
  frame_rate: 25

fusion:
  mode: balanced
  confidence_threshold: 0.60
  ambiguity_margin: 0.10
  weights:
    asr: 1.0
    diarization: 1.0
    face: 1.0
    voice: 1.5
    active_speaker: 1.0

device: cuda
batch_size: 4
num_workers: 2
```

### Step 3: Inspect available algorithms

```bash
WhoSpoke list-backends
WhoSpoke list-backends --category asr
WhoSpoke list-backends --category face
```

### Step 4: Run from the command line

```bash
WhoSpoke analyze g7_summit.mp4 \
  --people people.yaml \
  --config analysis.yaml \
  --output output_dir
```

You can also override algorithms directly without editing the YAML file:

```bash
WhoSpoke analyze g7_summit.mp4 \
  --people people.yaml \
  --config analysis.yaml \
  --asr-backend whisper \
  --asr-model-name medium \
  --diarization-algorithm pyannote \
  --clustering-method spectral \
  --face-algorithm facenet \
  --voice-algorithm ecapa_tdnn \
  --active-speaker-algorithm fast_asd \
  --output output_dir
```

### Step 5: Run from Python

```python
from WhoSpoke import AnalysisConfig, PersonProfile, SpeakerVideoAnalyzer

people = [
    PersonProfile(
        id="emmanuel_macron",
        name="Emmanuel Macron",
        portraits=["assets/macron_1.jpg"],
        voice_samples=["assets/macron_voice_1.wav"],
    ),
    PersonProfile(
        id="olaf_scholz",
        name="Olaf Scholz",
        portraits=["assets/scholz_1.jpg"],
        voice_samples=["assets/scholz_voice_1.wav"],
    ),
]

config = AnalysisConfig.from_yaml("analysis.yaml")
analyzer = SpeakerVideoAnalyzer(config=config)
result = analyzer.analyze("g7_summit.mp4", people=people)

result.to_json("output_dir/result.json")
result.to_csv("output_dir/result.csv")
```

### Step 6: Expected output shape

The current skeleton emits mock results but preserves the intended schema:

```text
0:00.0 – 0:05.0  UNKNOWN: [DUMMY] This is a placeholder transcript segment.
```

Future full-backend results will look like this:

```text
0:00.0 – 0:06.5  Emmanuel Macron: Il est impératif que nous renforcions notre coopération bilatérale pour la sécurité énergétique.
0:06.5 – 0:12.2  Olaf Scholz: Das ist absolut richtig, Emmanuel. Wir müssen gemeinsam handeln, um diese Krise zu bewältigen.
0:12.2 – 0:18.0  Emmanuel Macron: Therefore, we are proposing a joint framework moving forward...
```

The JSON output includes per-segment fields such as `speaker_id`, `speaker_name`, `confidence`, `evidence_scores`, and `algorithm_trace`.
```

---

## License

This project is licensed under the MIT License.
