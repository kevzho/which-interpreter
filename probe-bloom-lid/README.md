---
annotations_creators:
- expert-generated
language_creators:
- expert-generated
language:
- ajz
- bam
- bi
- bis
- bjn
- bm
- boz
- bze
- bzi
- cak
- ceb
- chd
- chp
- clo
- csw
- en
- eng
- es
- fli
- fr
- fra
- gu
- guj
- hbb
- hi
- hin
- id
- ind
- jmx
- jra
- kan
- kbq
- kek
- kjb
- kmu
- kn
- kqr
- kwu
- loh
- mai
- mal
- mam
- mar
- ml
- mle
- mr
- my
- mya
- myk
- nas
- nsk
- nsn
- oj
- oji
- omw
- por
- pt
- quc
- sdk
- snk
- spa
- stk
- ta
- taj
- tam
- tbj
- tdc
- tgl
- tl
- tpi
- tuz
- tzj
license:
- cc-by-nc-4.0
- cc-by-sa-4.0
- cc-by-nc-nd-4.0
- cc-by-nc-sa-4.0
multilinguality:
- multilingual
size_categories:
- 10K<n<100K
source_datasets:
- original
task_categories:
- automatic-speech-recognition
- text-to-speech
paperswithcode_id: null
pretty_name: BloomSpeech
extra_gated_prompt: |-
  One more step before getting this dataset. This dataset is open access and available only for non-commercial use (except for portions of the dataset labeled with a `cc-by-sa` license). A "license" field paired with each of the dataset entries/samples specifies the Creative Commons license for that entry/sample.
  
  These [Creative Commons licenses](https://creativecommons.org/about/cclicenses/) specify that: 
  1. You cannot use the dataset for or directed toward commercial advantage or monetary compensation (except for those portions of the dataset labeled specifically with a `cc-by-sa` license. If you would like to ask about commercial uses of this dataset, please [email us](mailto:sj@derivation.co).
  2. Any public, non-commercial use of the data must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use. 
  3. For those portions of the dataset marked with an ND license, you cannot remix, transform, or build upon the material, and you may not distribute modified material. 
  
  In addition to the above implied by Creative Commons and when clicking "Access Repository" below, you agree: 
  
  1. Not to use the dataset for any use intended to or which has the effect of harming or enabling discrimination against individuals or groups based on legally protected characteristics or categories, including but not limited to discrimination against Indigenous People as outlined in Articles 2; 13-16; and 31 of the United Nations Declaration on the Rights of Indigenous People, 13 September 2007 and as subsequently amended and revised.
  2. That your *contact information* (email address and username) can be shared with the model authors as well.
    
extra_gated_fields:
 I have read the License and agree with its terms: checkbox
---

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Languages](#languages)

- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
<!-- - [Dataset Creation](#dataset-creation) 
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions) -->

## Dataset Description

- **Homepage:** [SIL AI](https://ai.sil.org/)
- **Point of Contact:** [SIL AI email](mailto:idx_aqua@sil.org)
- **Source Data:** [Bloom Library](https://bloomlibrary.org/)

![logo for Bloom Library](https://bloom-vist.s3.amazonaws.com/bloom_logo.png) ![sil-ai logo](https://s3.amazonaws.com/moonup/production/uploads/1661440873726-6108057a823007eaf0c7bd10.png) 

## Dataset Summary

**Bloom** is free, open-source software and an associated website [Bloom Library](https://bloomlibrary.org/), app, and services developed by [SIL International](https://www.sil.org/). Bloom’s primary goal is to equip non-dominant language communities and their members to create the literature they want for their community and children. Bloom also serves organizations that help such communities develop literature and education or other aspects of community development.

This version of the Bloom Library data is developed specifically for the automatic speech recognition and speech-to-text tasks. It includes data from 56 languages across 18 language families. There is a mean of 458 and median of 138 audio records per language. 

**Note**: If you speak one of these languages and can help provide feedback or corrections, please let us know!

**Note**: Although data from [bloom-lm](https://huggingface.co/datasets/sil-ai/bloom-lm) was used in the training of the [BLOOM model](https://huggingface.co/bigscience/bloom), the dataset only represents a small portion of the data used to train that model. Data from "Bloom Library" was combined with a large number of other datasets to train that model. "Bloom Library" is a project that existed prior to the BLOOM model, and is something separate. All that to say... We were using the "Bloom" name before it was cool. 😉  

## Languages

Of the 500+ languages listed at BloomLibrary.org, there are 56 languages available in this dataset. Here are the corresponding ISO 639-3 codes:

ajz, bam, bis, bjn, boz, bze, bzi, cak, ceb, chd, chp, clo, csw, eng, fli, fra, guj, hbb, hin, ind, jmx, jra, kan, kbq, kek, kjb, kmu, kqr, kwu, loh, mai, mal, mam, mar, mle, mya, myk, nas, nsk, nsn, oji, omw, por, quc, sdk, snk, spa, stk, taj, tam, tbj, tdc, tgl, tpi, tuz, tzj

## Dataset Statistics
Some of the languages included in the dataset include few audio cuts. These are not split between training, validation, and test. For those with higher numbers of available stories we include the following numbers of stories in each split:

| ISO 639-3   | Name                          |      Train Cuts |      Validation Cuts |      Test Cuts |
|:------------|:------------------------------|----------------:|---------------------:|---------------:|
| ajz         | Amri Karbi   			|             135 |                   34 |             50 |
| bam         | Bamanankan   			|             203 |                   50 |             50 |
| bis         | Bislama   			|               0 |                    0 |             46 |
| bjn         | Banjar   			|              80 |                   20 |             50 |
| boz         | Bozo, Tieyaxo   		|             427 |                   50 |             52 |
| bze         | Bozo, Jenaama   		|             101 |                   26 |             50 |
| bzi         | Bisu   				|            1363 |                   50 |            157 |
| cak         | Kaqchikel   			|             989 |                   50 |            115 |
| ceb         | Cebuano   			|             553 |                   50 |             67 |
| chd         | Chontal, Highland Oaxaca   	|             205 |                   50 |             50 |
| chp         | Dene   				|               0 |                    0 |             14 |
| clo         | Chontal, Lowland Oaxaca   	|             120 |                   30 |             50 |
| csw         | Cree, Swampy   			|               0 |                    0 |             45 |
| eng         | English   			|            4143 |                   48 |            455 |
| fli         | Fali Muchella   		|              59 |                   15 |             50 |
| fra         | French   			|             261 |                   49 |             50 |
| guj         | Gujarati   			|              27 |                    0 |             48 |
| hbb         | Nya Huba   			|             558 |                   50 |             67 |
| hin         | Hindi   			|              62 |                   15 |             49 |
| ind         | Indonesian   			|               0 |                    0 |             14 |
| jmx         | Mixtec, Western Juxtlahuaca   	|              39 |                    0 |             50 |
| jra         | Jarai   			|             203 |                   50 |             50 |
| kan         | Kannada   			|             281 |                   43 |             50 |
| kbq         | Kamano   			|               0 |                    0 |             27 |
| kek         | Q’eqchi’   			|            1676 |                   49 |            190 |
| kjb         | Q’anjob’al   			|             770 |                   50 |             91 |
| kmu         | Kanite   			|               0 |                    0 |             28 |
| kqr         | Kimaragang   			|               0 |                    0 |             18 |
| kwu         | Kwakum   			|              58 |                   15 |             50 |
| loh         | Narim   			|               0 |                    0 |             15 |
| mai         | Maithili   			|               0 |                    0 |             11 |
| mal         | Malayalam   			|             125 |                   31 |             44 |
| mam         | Mam   				|            1313 |                   50 |            151 |
| mar         | Marathi   			|              25 |                    0 |             49 |
| mle         | Manambu   			|               0 |                    0 |              8 |
| mya         | Burmese   			|             321 |                   50 |             50 |
| myk         | Sénoufo, Mamara   		|             669 |                   50 |             80 |
| nas         | Naasioi   			|              13 |                    0 |             50 |
| nsk         | Naskapi   			|               0 |                    0 |             15 |
| nsn         | Nehan   			|               0 |                    0 |             31 |
| oji         | Ojibwa   			|               0 |                    0 |             25 |
| omw         | Tairora, South   		|               0 |                    0 |             34 |
| por         | Portuguese   			|               0 |                    0 |             34 |
| quc         | K’iche’   			|            1460 |                   50 |            167 |
| sdk         | Sos Kundi   			|             312 |                   50 |             50 |
| snk         | Soninke   			|             546 |                   50 |             66 |
| spa         | Spanish   			|            1816 |                   50 |            207 |
| stk         | Aramba   			|             180 |                   45 |             50 |
| taj         | Tamang, Eastern   		|               0 |                    0 |             24 |
| tam         | Tamil   			|             159 |                   39 |             46 |
| tbj         | Tiang   			|               0 |                    0 |             24 |
| tdc         | Ẽpẽra Pedea   			|               0 |                    0 |             19 |
| tgl         | Tagalog   			|             352 |                   48 |             50 |
| tpi         | Tok Pisin   			|            1061 |                   50 |            123 |
| tuz         | Turka   			|              48 |                   13 |             50 |
| tzj         | Tz’utujil   			|               0 |                    0 |             41 |

## Dataset Structure

### Data Instances

The examples look like this for Hindi:

```
from datasets import load_dataset

# Specify the language code.
dataset = load_dataset('sil-ai/bloom-speech', 'hin', use_auth_token=True) #note you must login to HuggingFace via the huggingface hub or huggingface cli

# A data point consists of transcribed audio in the specified language code.
# To see a transcription:
print(dataset['train']['text'][0])
```

This would produce an output:

```
चित्र: बो और शैम्पू की बोतल
```

Whereas if you wish to gather all the text for a language you may use this:
```
dataset['train']['text']
```

### Data Fields

The metadata fields are below. In terms of licenses, all stories included in the current release are released under a Creative Commons license (even if the individual story metadata fields are missing). 

- **file**: the local path to the audio file
- **audio**: a dictionary with a path, array, and sampling_rate as is standard for Hugging Face audio 
- **text**: the transcribed text 
- **book**: title of the book, e.g. "बो  मेस्सी और शैम्पू".
- **instance**: unique ID for each book/translation assigned by Bloom Library. For example the Hindi version of 'बो  मेस्सी और शैम्पू' is 'eba60f56-eade-4d78-a66f-f52870f6bfdd' 
- **license**: specific license used, e.g. "cc-by-sa" for "Creative Commons, by attribution, share-alike". 
- **credits**: attribution of contributors as described in the book metadata, including authors, editors, etc. if available
- **original_lang_tag**: the language tag originally assigned in Bloom Library. This may include information on script type, etc.

### Data Splits

All languages include a train, validation, and test split. However, for language having a small number of stories, certain of these splits maybe empty. In such cases, we recommend using any data for testing only or for zero-shot experiments. 

## Changelog
- **26 September 2022** Page initiated