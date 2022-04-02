from fairseq.models.transformer import TransformerModel, transformer_iwslt_de_en
from fairseq.models import register_model, register_model_architecture
from . import (
    bidirectional_transformer as _,
    bidirectional_translation_task as _,
)