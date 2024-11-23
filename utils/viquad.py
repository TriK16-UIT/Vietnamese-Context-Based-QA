import json
import pandas as pd
import datasets

logger = datasets.logging.get_logger(__name__)


class ViQuADConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(ViQuADConfig, self).__init__(**kwargs)


class ViQuAD(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        ViQuADConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="UIT-ViQuAD2.0",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": "Data/raw/train.parquet"}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": "Data/raw/val.parquet"}),
        ]

    def _generate_examples(self, filepath):
        logger.info("generating examples from = %s", filepath)
        
        df = pd.read_parquet(filepath)

        for key, row in enumerate(df.itertuples()):
            if row.is_impossible is False:
                answers = row.answers
                answer_starts = answers['answer_start'].tolist()
                answers_text = answers['text'].tolist()
            else:
                answer_starts = [0]
                answers_text = ""
                
            example = {
                "id": row.id,
                "title": row.title,
                "context": row.context,
                "question": row.question,
                "answers": {
                    "answer_start": answer_starts,
                    "text": answers_text,
                }
            }
            
            yield key, example