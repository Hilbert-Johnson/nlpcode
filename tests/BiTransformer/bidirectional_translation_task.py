import os
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq.data import data_utils, PrependTokenDataset, LanguagePairDataset, ConcatDataset

@register_task('bidirectional_translation_task')
class BidirectionalTranslationTask(TranslationTask):
    def load_dataset(self, split, **kwargs):
        shared_dict = self.src_dict
        src, tgt = data_utils.infer_language_pair(self.args.data)
        prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, src, tgt))

        src_raw_dataset = data_utils.load_indexed_dataset(prefix + self.args.source_lang, shared_dict)
        tgt_raw_dataset = data_utils.load_indexed_dataset(prefix + self.args.target_lang, shared_dict)

        src_prepend_dataset = PrependTokenDataset(src_raw_dataset, shared_dict.index('__2<{}>__'.format(self.args.target_lang)))
        tgt_prepend_dataset = PrependTokenDataset(tgt_raw_dataset, shared_dict.index('__2<{}>__'.format(self.args.source_lang)))

        src_dataset = src_prepend_dataset if split == 'test' else ConcatDataset([src_prepend_dataset, tgt_prepend_dataset])
        tgt_dataset = tgt_raw_dataset     if split == 'test' else ConcatDataset([tgt_raw_dataset,     src_raw_dataset])

        self.datasets[split] = LanguagePairDataset(
            src_dataset, src_dataset.sizes, shared_dict, tgt_dataset, tgt_dataset.sizes, shared_dict)

    @classmethod
    def setup_task(cls, args, **kwargs):
        task = super(BidirectionalTranslationTask, cls).setup_task(args)
        for lang_token in sorted(['__2<{}>__'.format(args.source_lang), '__2<{}>__'.format(args.target_lang)]):
            task.src_dict.add_symbol(lang_token)
            task.tgt_dict.add_symbol(lang_token)
        return task