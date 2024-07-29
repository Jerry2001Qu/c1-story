from pydantic import BaseModel, Field, HttpUrl
from typing import List, Union, Optional, Tuple
from enum import Enum

class SoundEffectType(str, Enum):
    FADE_IN = "fadeIn"
    FADE_OUT = "fadeOut"
    FADE_IN_FADE_OUT = "fadeInFadeOut"

class SoundEffect(BaseModel):
    type: SoundEffectType
    duration: float = Field(1.0, ge=0)

class Soundtrack(BaseModel):
    src: HttpUrl
    effects: Optional[List[SoundEffect]] = None
    volume: float = Field(1.0, ge=0)

class Font(BaseModel):
    src: HttpUrl

class Track(BaseModel):
    clips: List["Clip"]

class FitOption(str, Enum):
    CROP = "crop"
    COVER = "cover"
    CONTAIN = "contain"
    NONE = "none"

class TransitionType(str, Enum):
    FADE_IN = "fadeIn"
    FADE_OUT = "fadeOut"

class Transition(BaseModel):
    type: TransitionType
    duration: float = Field(1.0, ge=0)

class Clip(BaseModel):
    asset: "Asset"
    start: float
    length: float = Field(ge=0)
    fit: FitOption = FitOption.CROP
    transitions: Optional[List[Transition]] = None
    opacity: Optional[float] = Field(1.0, ge=0, le=1)

class AssetType(str, Enum):
    VIDEO = "video"
    IMAGE = "image"
    HTML = "html"
    AUDIO = "audio"
    PREBUILT = "prebuilt"

class Crop(BaseModel):
    top: Optional[float] = None
    bottom: Optional[float] = None
    left: Optional[float] = None
    right: Optional[float] = None

class VideoAsset(BaseModel):
    type: AssetType = AssetType.VIDEO
    src: HttpUrl
    trim: float = Field(0.0, ge=0)
    volume: float = Field(1.0, ge=0)
    volumeEffects: Optional[List[SoundEffect]] = None
    speed: float = Field(1.0, ge=0)
    crop: Optional[Crop] = None

class ImageAsset(BaseModel):
    type: AssetType = AssetType.IMAGE
    src: HttpUrl
    crop: Optional[Crop] = None

class HtmlAsset(BaseModel):
    type: AssetType = AssetType.HTML
    html: str
    css: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    background: Optional[Tuple[int, int, int, float]] = Field(default=(0, 0, 0, 0.0))
    position: Optional[str] = Field(None, pattern="^(top|topRight|right|bottomRight|bottom|bottomLeft|left|topLeft|center)$")

class AudioAsset(BaseModel):
    type: AssetType = AssetType.AUDIO
    src: HttpUrl
    trim: float = Field(0.0, ge=0)
    volume: float = Field(1.0, ge=0)
    speed: float = Field(1.0, ge=0)
    effects: Optional[List[SoundEffect]] = None

class PrebuiltAsset(BaseModel):
    type: AssetType = AssetType.PREBUILT
    key: str
    data: Optional[List] = None

Asset = Union[VideoAsset, ImageAsset, HtmlAsset, AudioAsset, PrebuiltAsset]

class Section(BaseModel):
    text: Optional[str] = None
    data: Optional[List] = None
    tracks: Optional[List[Track]] = None

    offset: float = Field(0.0, ge=0)
    trim: float = Field(0.0, ge=0)
    length: Optional[float] = Field(None, ge=0)
    volume: float = Field(1.0, ge=0)
    speed: float = Field(1.0, ge=0)
    volumeEffects: Optional[List[SoundEffect]] = None

class Overlay(BaseModel):
    data: Optional[List] = None
    tracks: Optional[List[Track]] = None

class Timeline(BaseModel):
    soundtrack: Optional[Soundtrack] = None
    background: Tuple[int, int, int] = (0, 0, 0)
    fonts: Optional[List[Font]] = None
    overlay: Optional[Overlay] = None
    sections: Optional[List[Section]] = None

class OutputFormat(str, Enum):
    MP4 = "mp4"

class OutputResolution(str, Enum):
    PREVIEW = "preview"
    SD = "sd"
    HD = "hd"
    FHD = "1080"
    UHD = "4k"

class OutputQuality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "veryhigh"

class OutputSize(BaseModel):
    width: int
    height: int

class Output(BaseModel):
    format: OutputFormat = OutputFormat.MP4
    resolution: OutputResolution = OutputResolution.FHD
    size: Optional[OutputSize] = None
    fps: float = 29.97
    quality: OutputQuality = OutputQuality.MEDIUM
    bitrate: Optional[str] = None
    mute: bool = False

class VideoSchema(BaseModel):
    timeline: Timeline
    output: Output

Clip.model_rebuild()
