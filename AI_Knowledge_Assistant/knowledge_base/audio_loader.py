from pathlib import Path
from faster_whisper import WhisperModel
from langchain_core.documents import Document

_model = None

def get_whisper_model():
    global _model

    if _model is None:
        _model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8",
        )

    return _model

def load_wav(file_path: str) -> list[Document]:
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    if path.suffix.lower() != ".wav":
        raise ValueError("Only WAV files are supported.")

    model = get_whisper_model()
    segments, info = model.transcribe(str(path))

    transcription = " ".join(
        segment.text.strip() for segment in segments
    ).strip()

    if not transcription:
        raise ValueError("No speech could be transcribed from the audio.")

    return [
        Document(
            page_content=transcription,
            metadata={
                "source": str(path.resolve()),
                "file_type": "wav",
                "language": info.language,
            },
        )
    ]
