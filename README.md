# NaijaVoices & FLEURS Data Loader

This repository provides a flexible PyTorch-based data loading pipeline tailored for training speech recognition and text-to-speech (TTS) models on NaijaVoices dataset. It supports both custom datasets and the multilingual [Google FLEURS](https://huggingface.co/datasets/google/fleurs) dataset.

---


## 🛠️ Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/iambusayor/naijavoices-dataloader.git
   cd naijavoices-dataloader
   ```

2. **Create a Virtual Environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   *Note: Ensure you have [FFmpeg](https://ffmpeg.org/download.html) installed for audio processing.*

---

## 📁 Directory Structure

```
naijavoices-dataloader/
├── dataset.py           # Main dataset classes
├── preprocessing.py     # Audio and text preprocessing utilities
├── collators.py         # Custom data collators for batching
├── requirements.txt     # Python dependencies
└── README.md            # This documentation
```

---

## 📄 Usage

### 1. **Custom Dataset: NaijaVoices**

To load your own dataset:

```python
from dataset import NaijaVoices
from transformers import Wav2Vec2Processor

# Initialize processor (example with Wav2Vec2)
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')

dataset = NaijaVoices(
    data_file='path/to/your_dataset.csv',
    max_audio_len_secs=30,
    audio_dir='path/to/audio_files',
    processor=processor,
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    language_iso='ha'  # Optional: filter by language code
)
```

**CSV File Format**:

Ensure your CSV file has the following columns:

- `audio_path`: Relative path to the audio file.
- `text`: Text label.
- `duration`: Duration of the audio in seconds.
- `language`: Language code (e.g., 'ha' for Hausa).

### 2. **Google FLEURS Dataset**

To load the FLEURS dataset:

```python
from dataset import FleursDataset
from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')

dataset = FleursDataset(
    split='train',
    processor=processor,
    language_iso='ha'  # Options: 'ha', 'ig', 'yo', or 'all'
)
```

---

## 🧹 Preprocessing

The `preprocessing.py` module offers utilities for:

- **Audio Loading**: Load and resample audio files to 16kHz.
- **Text Cleaning**: Normalize transcriptions by removing unwanted characters and formatting.
- **Vocabulary Creation**: Generate a vocabulary file (`vocab.json`) from your dataset.

**Example: Clean Text**

```python
from preprocessing import clean_text

raw_text = "Hello,   World!   "
cleaned_text = clean_text(raw_text)
print(cleaned_text)  # Output: "hello, world!"
```

---

## 🧩 Data Collators

Custom collators in `collators.py` handle batching for different model types:

- **CTC Models**:

  ```python
  from collators import DataCollatorCTCWithPadding

  collator = DataCollatorCTCWithPadding(processor=processor)
  ```

- **Sequence-to-Sequence Models**:

  ```python
  from collators import DataCollatorSpeechSeq2SeqWithPadding

  collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
  ```

These collators ensure that inputs and labels are appropriately padded and formatted for training.

---

## 🚀 Training Integration

Integrate the dataset and collator into your training loop or framework.

**Example with PyTorch DataLoader**:

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collator
)
```

Use `train_loader` in your training loop to feed data to your model.

---

## ⚙️ Configuration

Adjust parameters as needed:

- **Maximum Audio Length**: Set `max_audio_len_secs` to filter out longer audio files.
- **Language Filtering**: Use `language_iso` to focus on a specific language.
- **Processor Components**: Customize `feature_extractor` and `tokenizer` based on your model requirements.

---

## 📝 Notes

- **Audio Files**: Ensure all audio files are in WAV format and match the paths specified in your CSV.
- **Dependencies**: Some functionalities rely on external libraries like `librosa` and `transformers`. Ensure they are installed.
- **Error Handling**: The data loader includes basic error handling for missing or corrupted audio files.

---

