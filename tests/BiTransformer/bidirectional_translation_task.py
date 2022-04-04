import os
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask
from fairseq.data import data_utils, PrependTokenDataset, LanguagePairDataset, ConcatDataset

@register_task('bidirectional_translation_task', dataclass=TranslationConfig)
class BidirectionalTranslationTask(TranslationTask):
    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        shared_dict = self.src_dict
        src, tgt = data_utils.infer_language_pair(self.cfg.data)
        prefix = os.path.join(self.cfg.data, '{}.{}-{}.'.format(split, src, tgt))

        src_raw_dataset = data_utils.load_indexed_dataset(prefix + self.cfg.source_lang, shared_dict)
        tgt_raw_dataset = data_utils.load_indexed_dataset(prefix + self.cfg.target_lang, shared_dict)

        src_prepend_dataset = PrependTokenDataset(src_raw_dataset, shared_dict.index('__2<{}>__'.format(self.cfg.target_lang)))
        tgt_prepend_dataset = PrependTokenDataset(tgt_raw_dataset, shared_dict.index('__2<{}>__'.format(self.cfg.source_lang)))

        src_dataset = src_prepend_dataset if split == 'test' else ConcatDataset([src_prepend_dataset, tgt_prepend_dataset])
        tgt_dataset = tgt_raw_dataset     if split == 'test' else ConcatDataset([tgt_raw_dataset,     src_raw_dataset])

        self.datasets[split] = LanguagePairDataset(
            src_dataset, src_dataset.sizes, shared_dict, tgt_dataset, tgt_dataset.sizes, shared_dict)

    @classmethod
    def setup_task(cls, cfg: TranslationConfig, **kwargs):
        task = super(BidirectionalTranslationTask, cls).setup_task(cfg)
        for lang_token in sorted(['__2<{}>__'.format(cfg.source_lang), '__2<{}>__'.format(cfg.target_lang)]):
            task.src_dict.add_symbol(lang_token)
            task.tgt_dict.add_symbol(lang_token)
        return task 