"""Shared, validated requests for the browser, CLI, and inference worker."""
import secrets
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    title: str = Field(default="Untitled session", max_length=160)
    model: Literal["turbo", "xl"] = "turbo"
    mode: Literal["music", "speech"] = "music"
    task: Literal["text2music", "cover", "repaint"] = "text2music"
    caption: str = Field(default="Warm soul, brushed drums, Rhodes piano, intimate vocals", max_length=4000)
    lyrics: str = Field(default="", max_length=12000)
    instrumental: bool = False
    duration: float = Field(default=30, ge=0.5, le=600)
    bpm: int | None = Field(default=None, ge=30, le=300)
    keyscale: str = Field(default="", max_length=40)
    timesignature: Literal["", "2", "3", "4", "6"] = ""
    language: str = Field(default="en", pattern=r"^(unknown|[a-z]{2,3})$")
    steps: int = Field(default=8, ge=1, le=20)
    seed: int = Field(default=-1, ge=-1, le=2147483647)
    thinking: bool = False
    lm_temperature: float = Field(default=0.85, ge=0, le=2)
    lm_top_p: float = Field(default=0.9, gt=0, le=1)
    shift: float = Field(default=1, gt=0, le=10)
    offload: bool = True
    source_audio: str | None = None
    reference_audio: str | None = None
    cover_strength: float = Field(default=1, ge=0, le=1)
    repaint_start: float = Field(default=0, ge=0)
    repaint_end: float = Field(default=10, gt=0)
    voice_style: str = Field(default="Warm, clear British narrator, natural conversational delivery", max_length=2000)
    text: str = Field(default="", max_length=2000)
    session_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    sequence: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def valid_combination(self):
        if self.mode == "speech":
            if not self.text.strip():
                raise ValueError("Enter text for experimental speech")
            if self.task != "text2music" or self.thinking or self.instrumental:
                raise ValueError("Speech uses text2music with thinking and instrumental disabled")
        elif self.duration < 10:
            raise ValueError("Music duration must be at least 10 seconds")
        if self.task in {"cover", "repaint"} and not self.source_audio:
            raise ValueError("Cover and repaint require source audio")
        if self.task == "repaint" and self.repaint_end <= self.repaint_start:
            raise ValueError("Repaint end must be after its start")
        return self

    def resolved(self):
        return self.model_copy(update={"seed": secrets.randbelow(2147483648)}) if self.seed < 0 else self.model_copy()

    def prompt(self):
        if self.mode == "speech":
            return {
                "caption": f"Solo spoken word, dry close-mic speech, no music, no singing, no instruments. {self.voice_style}",
                "lyrics": "[spoken word]\n" + self.text.strip(),
            }
        return {"caption": self.caption, "lyrics": "[Instrumental]" if self.instrumental else self.lyrics}

    def inference_params(self):
        return dict(**self.prompt(), task_type=self.task, duration=self.duration,
                    instrumental=self.instrumental, bpm=self.bpm, keyscale=self.keyscale,
                    timesignature=self.timesignature, vocal_language=self.language,
                    inference_steps=self.steps, seed=self.seed, thinking=self.thinking,
                    use_cot_metas=self.thinking, use_cot_caption=False, use_cot_language=False,
                    lm_temperature=self.lm_temperature, lm_top_p=self.lm_top_p,
                    shift=self.shift, reference_audio=self.reference_audio, src_audio=self.source_audio,
                    audio_cover_strength=self.cover_strength, repainting_start=self.repaint_start,
                    repainting_end=self.repaint_end)


PRESETS = {
    "after-hours": dict(title="After hours", caption="Intimate neo-soul, warm Rhodes, brushed drums, rounded bass, smoky alto vocals, late-night atmosphere", bpm=82, keyscale="D minor", lyrics="[Verse]\nThe city settles into blue\nA quiet room, a softer view\n[Chorus]\nLet the night move slow\nThere is nowhere else to go"),
    "neon": dict(title="Neon avenue", caption="Instrumental synthwave, analog arpeggios, driving electronic drums, warm bass, cinematic night drive", instrumental=True, bpm=112, keyscale="A minor"),
    "daybreak": dict(title="Daybreak", caption="Acoustic indie folk, fingerpicked guitar, gentle percussion, warm conversational vocals, hopeful morning light", bpm=96, keyscale="G major", lyrics="[Verse]\nLight is falling through the trees\nMorning carries us with ease\n[Chorus]\nHere we go, a brand new day\nLet the daylight lead the way"),
}
