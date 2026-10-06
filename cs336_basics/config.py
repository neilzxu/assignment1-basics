from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 10_000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 8
    d_ff: int = 1344
    rope_theta: float = 10_000.0


@dataclass
class DataConfig:
    train_path: str
    valid_path: str


@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    min_learning_rate: float = 1e-4
    warmup_steps: int = 10
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    grad_clip: float = 1.0


@dataclass
class TrainingConfig:
    batch_size: int = 32
    max_steps: int = 10_000
    device: str = "cpu"
    seed: int = 42
    log_interval: int = 10
    overfit_one_batch: bool = False


@dataclass
class ValidationConfig:
    interval: int = 200
    batches: int = 20


@dataclass
class CheckpointConfig:
    save_dir: str = "artifacts/runs/tinystories"
    resume_from: str | None = None
    interval: int = 1000


@dataclass
class TrainConfig:
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def validate(self) -> None:
        if self.model.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.model.d_model % self.model.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.training.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.training.max_steps <= 0:
            raise ValueError("max_steps must be positive")


@dataclass
class DecodeConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    src: str = "artifacts/runs/tinystories/state.pt"
    vocab_path: str = "artifacts/tinystories_vocab.json"
    merges_path: str = "artifacts/tinystories_merges.json"
    prompt: str = "Once upon a time"
    device_str: str = "cpu"
    max_tokens: int = 200
    sampling: str = "temp"
    temp: float = 0.8
    p: float = 0.9
