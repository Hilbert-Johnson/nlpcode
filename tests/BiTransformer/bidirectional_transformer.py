from fairseq.models.transformer import TransformerModel, transformer_iwslt_de_en
from fairseq.models import register_model, register_model_architecture

@register_model('bidirectional_transformer')
class BidirectionalTransformerModel(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.make_shared_component()
    
    def make_shared_component(self):
        for enc_layer, dec_layer in zip(self.encoder.layers, self.decoder.layers):
            dec_layer.self_attn.k_proj = enc_layer.self_attn.k_proj
            dec_layer.self_attn.v_proj = enc_layer.self_attn.v_proj
            dec_layer.self_attn.q_proj = enc_layer.self_attn.q_proj
            dec_layer.self_attn.out_proj = enc_layer.self_attn.out_proj
            dec_layer.fc1 = enc_layer.fc1
            dec_layer.fc2 = enc_layer.fc2

@register_model_architecture('bidirectional_transformer', 'iwslt_arch')
def iwslt_preset_hyperparameters(args):
    transformer_iwslt_de_en(args)