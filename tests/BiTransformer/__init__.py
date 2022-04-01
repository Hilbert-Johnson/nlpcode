from fairseq.models.transformer import TransformerModel, transformer_iwslt_de_en
from fairseq.models import register_model, register_model_architecture

@register_model('my_transformer')
class MyTransformer(TransformerModel):
    pass

@register_model_architecture('my_transformer', 'iwslt_arch')
def my_transformer_iwslt(args):
    transformer_iwslt_de_en(args)