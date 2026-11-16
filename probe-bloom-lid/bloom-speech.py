# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bloom-speech is a dataset of text aligned speech from bloomlibrary.org. This dataset contains over 50 languages including many low-resource languages. This dataset should be useful for training and/or testing speech-to-text or text-to-speech/ASR models."""


import csv
import json
import os
import random

import datasets
from datasets.tasks import AutomaticSpeechRecognition

_CITATION = """\
@InProceedings{huggingface:bloom-speech,
title = {bloom-speech},
author={Joshua Nemecek, Colin Leong, and Daniel Whitenack
},
year={2022}
}
"""
_DESCRIPTION = """Bloom-speech is a dataset of text aligned speech from bloomlibrary.org. This dataset contains over 50 languages including many low-resource languages. This dataset should be useful for training and/or testing speech-to-text or text-to-speech/ASR models."""
_LANGUAGES = ['aaa', 'ajz', 'bam', 'bi', 'bis', 'bjn', 'bm', 'boz', 'bze', 'bzi', 'cak', 'ceb', 'chd', 'chp', 'clo', 'csw', 'en', 'eng', 'es', 'fli', 'fr', 'fra', 'gu', 
                'guj', 'hbb', 'hi', 'hin', 'id', 'ind', 'ins', 'jmx', 'jra', 'kan', 'kbq', 'kek', 'kjb', 'kmu', 'kn', 'kqr', 'kwu', 'loh', 'mai', 'mal', 'mam', 'mar', 'ml', 'mle', 
                'mr', 'my', 'mya', 'myk', 'nas', 'nsk', 'nsn', 'oj', 'oji', 'omw', 'por', 'pt', 'quc', 'sdk', 'snk', 'spa', 'stk', 'ta', 'taj', 'tam', 'tbj', 'tdc', 'tgl', 'tl', 'tpi', 'tuz', 'tzj']
_LANG_ISO_DICT = {'en': 'eng', 'gu': 'guj', 'ml': 'mal', 'hi': 'hin', 'ta': 'tam', 'kn': 'kan', 'my': 'mya', 'mr': 'mar',
                 'es': 'spa', 'bm': 'bam', 'fr': 'fra', 'tl': 'tgl', 'pt': 'por', 'id': 'ind', 'oj': 'oji', 'bi': 'bis'}
_NC_DICT = {'cc-by':'cc-by-nc', 'cc-by-nd':'cc-by-nc-nd'}                 
_HOMEPAGE = "https://bloomlibrary.org/"

_LICENSE = "Various Creative Commons Licences. Each record indicates the license."

_FILES = {
    "bloom_speech": "bloom_speech.json",
    "book_info": "book_info.json",    
}

class BloomSpeechConfig(datasets.BuilderConfig):
    """BuilderConfig for bloom-speech"""
    def __init__(self, language='', filternumbers=True, quarantine=False, **kwargs):
        super(BloomSpeechConfig, self).__init__(**kwargs)
        self.language = _LANG_ISO_DICT.get(language, language)
        self.quarantine = quarantine
        self.filternumbers = filternumbers

class BloomSpeech(datasets.GeneratorBasedBuilder):
    """Bloom-speech class"""
    BUILDER_CONFIGS = [
        BloomSpeechConfig(name="bloom-speech", description="bloom-speech", language='', filternumbers=True, quarantine=False),
    ] + [BloomSpeechConfig(name=x, description=f'bloom-speech for language code {x}', language=x, filternumbers=True, quarantine=False) for x in _LANGUAGES]

    DEFAULT_CONFIG_NAME = 'bloom-speech'

    BUILDER_CONFIG_CLASS = BloomSpeechConfig

    VERSION = datasets.Version("0.0.1")

    def _info(self):
        features = datasets.Features(
            {
                "file": datasets.Value("string"),
                "audio": datasets.Audio(sampling_rate=16_000),
                "text": datasets.Value("string"),
                "book": datasets.Value("string"),
                "instance": datasets.Value("string"),
                "license": datasets.Value("string"),
                "credits": datasets.Value("string"),
                "original_lang_tag": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,  
            supervised_keys=("file", "text"),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            task_templates=[AutomaticSpeechRecognition(audio_column="audio", transcription_column="text")]
        )

    def _split_generators(self, dl_manager):
        
        if self.config.language == '':
            raise ValueError('Please specify a language.')
        elif self.config.language not in _LANGUAGES:
            raise ValueError(f'{language} does not appear in the list of languages: {_LANGUAGES}')

        urls = _FILES
        data_dir = dl_manager.download(urls)
        with open(data_dir['book_info'], 'r') as f:
            book_info = json.load(f)
        lbook_info = {x:book_info[self.config.language][x] for x in book_info[self.config.language].keys() if self.config.quarantine or not book_info[self.config.language][x]['quarantine']}
        with open(data_dir['bloom_speech'], 'r') as f:
            data = json.load(f)
        ldata = [x for x in data[self.config.language] if x['book'] in lbook_info.keys() and (not self.config.filternumbers or not x['containsNumber'])]

        audio_files = dl_manager.download([x['url'] for x in ldata])
        audio_index = {x['url']:idx for idx,x in enumerate(ldata)}

        if len(ldata) > 50:
            test_size = max(50, int(len(ldata) * 0.1))
        else:
            test_size = len(ldata)
        if (len(ldata) - test_size) > 50:
            train_size = max((len(ldata)-test_size-50), int((len(ldata)-test_size) * 0.8))
        else:
            train_size = len(ldata) - test_size
        dev_size = len(ldata) - train_size - test_size

        random.seed(42)
        random.shuffle(ldata)

        if dev_size > 0:
            dev_data = ldata[-dev_size:]
        else:
            dev_data = []

        # Pass `audio_files` as a tuple to not allow subsets of this list to be passed to different workers when generating the dataset
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "audio_files": tuple(audio_files),
                    "audio_index": audio_index,
                    "data": ldata[:train_size],
                    "book_info": lbook_info,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "audio_files": tuple(audio_files),
                    "audio_index": audio_index,
                    "data": ldata[train_size:train_size+test_size],
                    "book_info": lbook_info,
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "audio_files": tuple(audio_files),
                    "audio_index": audio_index,
                    "data": dev_data,
                    "book_info": lbook_info,
                    "split": "dev",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, audio_files, audio_index, data, book_info, split):
        for key, row in enumerate(data):
            try:
                tfile = audio_files[audio_index[row['url']]]
                if not tfile.endswith('.mp3'):
                    os.rename(tfile, tfile + '.mp3')
                    tfile += '.mp3'
                yield key, {
                    "file": tfile,
                    "audio": tfile,
                    "text": row['text'],
                    "book": row['book'],
                    "instance": book_info[row['book']]['instance'],
                    "license": _NC_DICT.get(book_info[row['book']]['license'],book_info[row['book']]['license']),
                    "credits": book_info[row['book']]['credits'],
                    "original_lang_tag": row['bloomLanguageTag'],
                }
            except:
                print(f'{row["url"]} failed to download. Data may be missing.')
                pass