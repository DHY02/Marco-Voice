"""
Copyright (C) 2025 AIDC-AI
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import os,sys
import json
import torch
from funasr import AutoModel
# from funasr.utils.postprocess_utils import rich_transcription_postprocess
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def init_models():
    paraformer = AutoModel(
        model="paraformer-zh", 
        vad_model="fsmn-vad", 
        punc_model="ct-punc",
        device=device,  
        disable_update=True,
        batch_size_s=300,  
    )
    
    whisper = pipeline(
        task=Tasks.auto_speech_recognition,
        model='iic/Whisper-large-v3',
        model_revision="v2.0.5",
        device=device,  
    )
    
    return paraformer, whisper

paraformer_model, whisper_model = init_models()

def speech_recognition(language, filepath):
    # try:
    if not os.path.exists(filepath):
        return f"File does not exist: {filepath}"
        
    if language.lower() == "zh":
        with torch.no_grad(): 
            res = paraformer_model.generate(
                input=filepath,
                cache={},
                language="auto",
                use_itn=True,
                merge_vad=True,
                merge_length_s=30,  
            )
        text = res[0]["text"] # rich_transcription_postprocess(res[0]["text"])
    else:
        with torch.no_grad():  
            res = whisper_model(input=filepath, language=None)
        text = res[0]["text"] # rich_transcription_postprocess(res[0]["text"])
    
    return text

    # except Exception as e:
    #     return f"处理失败: {str(e)}"

language_code = {'aa': 'Afar', 'ab': 'Abkhazian', 'ae': 'Avestan', 'af': 'Afrikaans', 'ak': 'Akan', 'am': 'Amharic', 'an': 'Aragonese',
                     'ar': 'Arabic', 'as': 'Assamese', 'av': 'Avaric', 'ay': 'Aymara', 'az': 'Azerbaijani', 'ba': 'Bashkir',
                     'be': 'Belarusian', 'bg': 'Bulgarian', 'bh': 'Bihari languages', 'bi': 'Bislama', 'bm': 'Bambara', 'bn': 'Bengali',
                     'bo': 'Tibetan', 'br': 'Breton', 'bs': 'Bosnian', 'ca': 'Catalan', 'ce': 'Chechen', 'ch': 'Chamorro', 'co': 'Corsican',
                     'cr': 'Cree', 'cs': 'Czech', 'cu': 'Church Slavic', 'cv': 'Chuvash', 'cy': 'Welsh', 'da': 'Danish', 'de': 'German',
                     'dv': 'Divehi', 'dz': 'Dzongkha', 'ee': 'Ewe', 'el': 'Greek', 'en': 'English', 'eo': 'Esperanto', 'es': 'Spanish',
                     'et': 'Estonian', 'eu': 'Basque', 'fa': 'Persian', 'ff': 'Fulah', 'fi': 'Finnish', 'fj': 'Fijian', 'fo': 'Faroese',
                     'fr': 'French', 'fy': 'Western Frisian', 'ga': 'Irish', 'gd': 'Scottish Gaelic', 'gl': 'Galician', 'gn': 'Guaraní',
                     'gu': 'Gujarati', 'gv': 'Manx', 'ha': 'Hausa', 'he': 'Hebrew', 'hi': 'Hindi', 'ho': 'Hiri Motu', 'hr': 'Croatian',
                     'ht': 'Haitian Creole', 'hu': 'Hungarian', 'hy': 'Armenian', 'hz': 'Herero', 'ia': 'Interlingua', 'id': 'Indonesian',
                     'ie': 'Interlingue', 'ig': 'Igbo', 'ii': 'Sichuan Yi', 'ik': 'Inupiaq', 'io': 'Ido', 'is': 'Icelandic', 'it': 'Italian',
                     'iu': 'Inuktitut', 'ja': 'Japanese', 'jv': 'Javanese', 'ka': 'Georgian', 'kg': 'Kongo', 'ki': 'Kikuyu', 'kj': 'Kuanyama',
                     'kk': 'Kazakh', 'kl': 'Kalaallisut', 'km': 'Central Khmer', 'kn': 'Kannada', 'ko': 'Korean', 'kr': 'Kanuri',
                     'ks': 'Kashmiri', 'ku': 'Kurdish', 'kv': 'Komi', 'kw': 'Cornish', 'ky': 'Kirghiz', 'la': 'Latin', 'lb': 'Luxembourgish',
                     'lg': 'Ganda', 'li': 'Limburgish', 'ln': 'Lingala', 'lo': 'Lao', 'lt': 'Lithuanian', 'lu': 'Luba-Katanga',
                     'lv': 'Latvian', 'mg': 'Malagasy', 'mh': 'Marshallese', 'mi': 'Maori', 'mk': 'Macedonian', 'ml': 'Malayalam',
                     'mn': 'Mongolian', 'mr': 'Marathi', 'ms': 'Malay', 'mt': 'Maltese', 'my': 'Burmese', 'na': 'Nauru',
                     'nb': 'Norwegian Bokmål', 'nd': 'North Ndebele', 'ne': 'Nepali', 'ng': 'Ndonga', 'nl': 'Dutch',
                     'nn': 'Norwegian Nynorsk', 'no': 'Norwegian', 'nr': 'South Ndebele', 'nv': 'Navajo', 'ny': 'Chichewa', 'oc': 'Occitan',
                     'oj': 'Ojibwe', 'om': 'Oromo', 'or': 'Oriya', 'os': 'Ossetian', 'pa': 'Panjabi', 'pi': 'Pali', 'pl': 'Polish',
                     'ps': 'Pushto', 'pt': 'Portuguese', 'qu': 'Quechua', 'rm': 'Romansh', 'rn': 'Rundi', 'ro': 'Romanian', 'ru': 'Russian',
                     'rw': 'Kinyarwanda', 'sa': 'Sanskrit', 'sc': 'Sardinian', 'sd': 'Sindhi', 'se': 'Northern Sami', 'sg': 'Sango',
                     'si': 'Sinhalese', 'sk': 'Slovak', 'sl': 'Slovenian', 'sm': 'Samoan', 'sn': 'Shona', 'so': 'Somali', 'sq': 'Albanian',
                     'sr': 'Serbian', 'ss': 'Swati', 'st': 'Sotho, Southern', 'su': 'Sundanese', 'sv': 'Swedish', 'sw': 'Swahili',
                     'ta': 'Tamil', 'te': 'Telugu', 'tg': 'Tajik', 'th': 'Thai', 'ti': 'Tigrinya', 'tk': 'Turkmen', 'tl': 'Tagalog',
                     'tn': 'Tswana', 'to': 'Tonga', 'tr': 'Turkish', 'ts': 'Tsonga', 'tt': 'Tatar', 'tw': 'Twi', 'ty': 'Tahitian',
                     'ug': 'Uighur', 'uk': 'Ukrainian', 'ur': 'Urdu', 'uz': 'Uzbek', 've': 'Venda', 'vi': 'Vietnamese', 'vo': 'Volapük',
                     'wa': 'Walloon', 'wo': 'Wolof', 'xh': 'Xhosa', 'yi': 'Yiddish', 'yo': 'Yoruba', 'za': 'Zhuang', 'zh': 'Chinese',
                     'zu': 'Zulu'}



def process_json(input_json_path, output_json_path, batch_size=4):
    """    
    Args:
        input_json_path (str): Input JSON file path
        output_json_path (str): Output JSON file path
        batch_size (int): Batch size (adjust according to GPU memory)
    """
    language_code = {'aa': 'Afar', 'ab': 'Abkhazian', 'ae': 'Avestan', 'af': 'Afrikaans', 'ak': 'Akan', 'am': 'Amharic', 'an': 'Aragonese',
                     'ar': 'Arabic', 'as': 'Assamese', 'av': 'Avaric', 'ay': 'Aymara', 'az': 'Azerbaijani', 'ba': 'Bashkir',
                     'be': 'Belarusian', 'bg': 'Bulgarian', 'bh': 'Bihari languages', 'bi': 'Bislama', 'bm': 'Bambara', 'bn': 'Bengali',
                     'bo': 'Tibetan', 'br': 'Breton', 'bs': 'Bosnian', 'ca': 'Catalan', 'ce': 'Chechen', 'ch': 'Chamorro', 'co': 'Corsican',
                     'cr': 'Cree', 'cs': 'Czech', 'cu': 'Church Slavic', 'cv': 'Chuvash', 'cy': 'Welsh', 'da': 'Danish', 'de': 'German',
                     'dv': 'Divehi', 'dz': 'Dzongkha', 'ee': 'Ewe', 'el': 'Greek', 'en': 'English', 'eo': 'Esperanto', 'es': 'Spanish',
                     'et': 'Estonian', 'eu': 'Basque', 'fa': 'Persian', 'ff': 'Fulah', 'fi': 'Finnish', 'fj': 'Fijian', 'fo': 'Faroese',
                     'fr': 'French', 'fy': 'Western Frisian', 'ga': 'Irish', 'gd': 'Scottish Gaelic', 'gl': 'Galician', 'gn': 'Guaraní',
                     'gu': 'Gujarati', 'gv': 'Manx', 'ha': 'Hausa', 'he': 'Hebrew', 'hi': 'Hindi', 'ho': 'Hiri Motu', 'hr': 'Croatian',
                     'ht': 'Haitian Creole', 'hu': 'Hungarian', 'hy': 'Armenian', 'hz': 'Herero', 'ia': 'Interlingua', 'id': 'Indonesian',
                     'ie': 'Interlingue', 'ig': 'Igbo', 'ii': 'Sichuan Yi', 'ik': 'Inupiaq', 'io': 'Ido', 'is': 'Icelandic', 'it': 'Italian',
                     'iu': 'Inuktitut', 'ja': 'Japanese', 'jv': 'Javanese', 'ka': 'Georgian', 'kg': 'Kongo', 'ki': 'Kikuyu', 'kj': 'Kuanyama',
                     'kk': 'Kazakh', 'kl': 'Kalaallisut', 'km': 'Central Khmer', 'kn': 'Kannada', 'ko': 'Korean', 'kr': 'Kanuri',
                     'ks': 'Kashmiri', 'ku': 'Kurdish', 'kv': 'Komi', 'kw': 'Cornish', 'ky': 'Kirghiz', 'la': 'Latin', 'lb': 'Luxembourgish',
                     'lg': 'Ganda', 'li': 'Limburgish', 'ln': 'Lingala', 'lo': 'Lao', 'lt': 'Lithuanian', 'lu': 'Luba-Katanga',
                     'lv': 'Latvian', 'mg': 'Malagasy', 'mh': 'Marshallese', 'mi': 'Maori', 'mk': 'Macedonian', 'ml': 'Malayalam',
                     'mn': 'Mongolian', 'mr': 'Marathi', 'ms': 'Malay', 'mt': 'Maltese', 'my': 'Burmese', 'na': 'Nauru',
                     'nb': 'Norwegian Bokmål', 'nd': 'North Ndebele', 'ne': 'Nepali', 'ng': 'Ndonga', 'nl': 'Dutch',
                     'nn': 'Norwegian Nynorsk', 'no': 'Norwegian', 'nr': 'South Ndebele', 'nv': 'Navajo', 'ny': 'Chichewa', 'oc': 'Occitan',
                     'oj': 'Ojibwe', 'om': 'Oromo', 'or': 'Oriya', 'os': 'Ossetian', 'pa': 'Panjabi', 'pi': 'Pali', 'pl': 'Polish',
                     'ps': 'Pushto', 'pt': 'Portuguese', 'qu': 'Quechua', 'rm': 'Romansh', 'rn': 'Rundi', 'ro': 'Romanian', 'ru': 'Russian',
                     'rw': 'Kinyarwanda', 'sa': 'Sanskrit', 'sc': 'Sardinian', 'sd': 'Sindhi', 'se': 'Northern Sami', 'sg': 'Sango',
                     'si': 'Sinhalese', 'sk': 'Slovak', 'sl': 'Slovenian', 'sm': 'Samoan', 'sn': 'Shona', 'so': 'Somali', 'sq': 'Albanian',
                     'sr': 'Serbian', 'ss': 'Swati', 'st': 'Sotho, Southern', 'su': 'Sundanese', 'sv': 'Swedish', 'sw': 'Swahili',
                     'ta': 'Tamil', 'te': 'Telugu', 'tg': 'Tajik', 'th': 'Thai', 'ti': 'Tigrinya', 'tk': 'Turkmen', 'tl': 'Tagalog',
                     'tn': 'Tswana', 'to': 'Tonga', 'tr': 'Turkish', 'ts': 'Tsonga', 'tt': 'Tatar', 'tw': 'Twi', 'ty': 'Tahitian',
                     'ug': 'Uighur', 'uk': 'Ukrainian', 'ur': 'Urdu', 'uz': 'Uzbek', 've': 'Venda', 'vi': 'Vietnamese', 'vo': 'Volapük',
                     'wa': 'Walloon', 'wo': 'Wolof', 'xh': 'Xhosa', 'yi': 'Yiddish', 'yo': 'Yoruba', 'za': 'Zhuang', 'zh': 'Chinese',
                     'zu': 'Zulu'}

    # try:
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data_all = json.load(f)
        print(f"Total data entries: {len(data_all)}")

    # Get all filenames and process in batches
    all_keys = list(data_all.keys())
    for i in range(0, len(all_keys), batch_size):
        batch_keys = all_keys[i:i + batch_size]
        print(f"\nProcessing batch: {i//batch_size + 1} (contains {len(batch_keys)} files)")
        
        for key in batch_keys:
            item = data_all[key]  # Get the data dictionary for the current file
            audio_path = item.get("path", "")
            language = item.get("language", "")
            
            # Get language code (make sure language_code dict is defined)
            language_co = language_code.get(language, "en")  # Default to English
            
            if audio_path:
                print(f"Processing: {key} - Language: {language}({language_co})")
                try:
                    # Call speech recognition function
                    item["hpy_text"] = speech_recognition(language_co, audio_path)
                    print(f"  Recognition result: {item['hpy_text'][:50]}...")  # Show first 50 chars
                except Exception as e:
                    print(f"  Recognition failed: {str(e)}")
                    item["hpy_text"] = "ERROR"
            else:
                print(f"Skip: {key} - Missing audio path")
                item["hpy_text"] = ""
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data_all, f, ensure_ascii=False, indent=4)
    print(f"Batch {i//batch_size + 1}/{(len(items)-1)//batch_size + 1} saved")
    
    print(f"\nProcessing done, results saved to: {output_json_path}")

    # except Exception as e:
    #     print(f"Error occurred: {str(e)}")
    # finally:
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()

if __name__ == "__main__":

    # directory = sys.argv[1] # json file path
    # subdirs = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    # for file in subdirs:
        # metadata_path = os.path.join(directory, file, "metadata_asr.json")
        # if os.path.isfile(metadata_path):   
        #     continue
    # try:
    process_json( # os.path.join(directory, file + "/metadata_asr.json"),
    "/mnt/workspace/fengping/data/multi_lingual_test_set/wavs/cosy_vocie1/ko_404h/metadata.json",
    "/mnt/workspace/fengping/data/multi_lingual_test_set/wavs/cosy_vocie1/ko_404h/metadat_asra.json",
    # input_json_path=os.path.join(directory, file + "/metadata.json"),
    # output_json_path=os.path.join(directory, file + "/metadata_asr.json"),
    batch_size=20)
    # except Exception as e:
    #     print(e)
