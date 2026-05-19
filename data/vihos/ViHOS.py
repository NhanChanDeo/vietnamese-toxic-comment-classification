"""ViHOS - Vietnamese Hate and Offensive Spans dataset"""

import pandas as pd

import datasets

_DESCRIPTION = """\
This is a dataset of Vietnamese Hate and Offensive Spans dataset from social media texts.
"""

_HOMEPAGE = "https://huggingface.co/datasets/phusroyal/ViHOS"

_LICENSE = "mit"

_URLS = [
    "https://raw.githubusercontent.com/phusroyal/ViHOS/master/data/Span_Extraction_based_version/dev.csv",
    "https://raw.githubusercontent.com/phusroyal/ViHOS/master/data/Span_Extraction_based_version/train.csv",
    "https://raw.githubusercontent.com/phusroyal/ViHOS/master/data/Test_data/test.csv"
]

class ViHOS_config(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(ViHOS_config, self).__init__(**kwargs)

class ViHOS(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        ViHOS_config(name="ViHOS", version=datasets.Version("2.0.0"), description=_DESCRIPTION),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            
            features=datasets.Features(
                {
                    "content": datasets.Value("string"),
                    "span_ids": datasets.Value("string")
                }
            ),
            
            homepage=_HOMEPAGE,
            
            license=_LICENSE
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_URLS)
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir[1],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir[0],
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir[2],
                    "split": "test",
                },
            )
        ]    
    def _generate_examples(self, filepath, split):
        
        data = pd.read_csv(filepath, header=None, sep=",", on_bad_lines='skip', skiprows=[0])
        
        for i in range(len(data)):
            content = str(data.loc[i, 1])
            span_ids = str(data.loc[i, 2])
            if span_ids is None:
                span_ids = ''
    
            yield i, {
                "content": content,
                "span_ids": span_ids,
            }