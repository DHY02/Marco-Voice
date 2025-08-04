"""
Copyright (C) 2025 AIDC-AI
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import re
# import MeCab 
from transformers import AutoTokenizer
import json

# mecab = MeCab.Tagger("-Owakati")
# mecab = MeCab.Tagger()



PUNCTUATIONS = "，。？！,\.?!＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·｡\":" + "()\[\]{}/;`|=+"


def preprocess(text):
    text = " ".join([t for t in re.split("<\|.*?\|>", text) if t.strip() != ""])
    text = re.sub("<unk>", "", text)
    text = re.sub(r"[%s]+" % PUNCTUATIONS, " ", text)
    return text


def non_latin_preprocess(text):
    text = " ".join([t for t in re.split("<\|.*?\|>", text) if t.strip() != ""])
    text = re.sub("<unk>", "", text)
    text = re.sub(r"[%s]+" % PUNCTUATIONS, "", text)
    return text


def en_zh_text2tokens(text):
    if text == "":
        return []
    text = preprocess(text)
    tokens = []

    pattern = re.compile(r'([\u4e00-\u9fff])')
    parts = pattern.split(text.strip().upper())
    parts = [p for p in parts if len(p.strip()) > 0]
    for part in parts:
        if pattern.fullmatch(part) is not None:
            tokens.append(part)
        else:
            for word in part.strip().split():
                tokens.append(word)
    return tokens


def ja_text2tokens(text):
    if text == "":
        return []
    text = non_latin_preprocess(text)

    parsed = mecab.parse(text).strip() 
    tokens = parsed.split()  
    return tokens


COST_SUB = 3
COST_DEL = 3
COST_INS = 3

ALIGN_CRT = 0
ALIGN_SUB = 1
ALIGN_DEL = 2
ALIGN_INS = 3
ALIGN_END = 4


def en_zh_wer(reference, hypothesis):
    ref_tokens = en_zh_text2tokens(reference)
    hyp_tokens = en_zh_text2tokens(hypothesis)
    return compute_one_wer_info_sub(ref_tokens, hyp_tokens)


def jap_wer(reference, hypothesis):
    ref_tokens = ja_text2tokens(reference)
    hyp_tokens = ja_text2tokens(hypothesis)
    return compute_one_wer_info_sub(ref_tokens, hyp_tokens)


class CommonWer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("/mnt/workspace/nixuanfan/models/nllb-200-3.3B", use_fast=True, token=True,
                                                       trust_remote_code=True)

    def thai_text2tokens(self, text):
        if text == "":
            return []
        text = preprocess(text)
        tokens = self.tokenizer.encode(text)
        return tokens

    def kor_text2tokens(self, text):
        if text == "":
            return []
        text = non_latin_preprocess(text)
        tokens = self.tokenizer.encode(text)
        return tokens

    def thai_wer(self, reference, hypothesis):
        ref_tokens = self.thai_text2tokens(reference)
        hyp_tokens = self.thai_text2tokens(hypothesis)
        # print(ref_tokens, hyp_tokens)
        return compute_one_wer_info_sub(ref_tokens, hyp_tokens)

    def kor_wer(self, reference, hypothesis):
        ref_tokens = self.kor_text2tokens(reference)
        hyp_tokens = self.kor_text2tokens(hypothesis)
        # print(ref_tokens, hyp_tokens)
        return compute_one_wer_info_sub(ref_tokens, hyp_tokens)


def compute_one_wer_info(ref, hyp):
    """Impl minimum edit distance and backtrace.
    Args:
        ref, hyp: # List[str]
    Returns:
        WerInfo
    """

    ref_len = len(ref)
    hyp_len = len(hyp)

    class _DpPoint:
        def __init__(self, cost, align):
            self.cost = cost
            self.align = align

    dp = []
    for i in range(0, ref_len + 1):
        dp.append([])
        for j in range(0, hyp_len + 1):
            dp[-1].append(_DpPoint(i * j, ALIGN_CRT))

    # Initialize
    for i in range(1, hyp_len + 1):
        dp[0][i].cost = dp[0][i - 1].cost + COST_INS;
        dp[0][i].align = ALIGN_INS
    for i in range(1, ref_len + 1):
        dp[i][0].cost = dp[i - 1][0].cost + COST_DEL
        dp[i][0].align = ALIGN_DEL

    # DP
    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            min_cost = 0
            min_align = ALIGN_CRT
            if hyp[j - 1] == ref[i - 1]:
                min_cost = dp[i - 1][j - 1].cost
                min_align = ALIGN_CRT
            else:
                min_cost = dp[i - 1][j - 1].cost + COST_SUB
                min_align = ALIGN_SUB

            del_cost = dp[i - 1][j].cost + COST_DEL
            if del_cost < min_cost:
                min_cost = del_cost
                min_align = ALIGN_DEL

            ins_cost = dp[i][j - 1].cost + COST_INS
            if ins_cost < min_cost:
                min_cost = ins_cost
                min_align = ALIGN_INS

            dp[i][j].cost = min_cost
            dp[i][j].align = min_align

    # Backtrace
    crt = sub = ins = det = 0
    i = ref_len
    j = hyp_len
    align = []
    while i > 0 or j > 0:
        if dp[i][j].align == ALIGN_CRT:
            align.append((i, j, ALIGN_CRT))
            i -= 1
            j -= 1
            crt += 1
        elif dp[i][j].align == ALIGN_SUB:
            align.append((i, j, ALIGN_SUB))
            i -= 1
            j -= 1
            sub += 1
        elif dp[i][j].align == ALIGN_DEL:
            align.append((i, j, ALIGN_DEL))
            i -= 1
            det += 1
        elif dp[i][j].align == ALIGN_INS:
            align.append((i, j, ALIGN_INS))
            j -= 1
            ins += 1

    # err = sub + det + ins
    align.reverse()
    # wer_info = WerInfo(ref_len, err, crt, sub, det, ins, align)
    wer = (sub + det + ins) / ref_len
    return wer

def compute_one_wer_info_sub(ref, hyp):
    """Impl minimum edit distance and backtrace.
    Args:
        ref, hyp: # List[str]
    Returns:
        Tuple: (wer, ins_del_count, sub_count, alignments)
    """
    ref_len = len(ref)
    hyp_len = len(hyp)

    class _DpPoint:
        def __init__(self, cost, align):
            self.cost = cost
            self.align = align

    # Constants for alignment types
    ALIGN_CRT = 0  # Correct
    ALIGN_SUB = 1  # Substitution
    ALIGN_DEL = 2  # Deletion
    ALIGN_INS = 3  # Insertion
    
    # Costs for each operation
    COST_SUB = 1
    COST_DEL = 1
    COST_INS = 1

    dp = []
    for i in range(0, ref_len + 1):
        dp.append([])
        for j in range(0, hyp_len + 1):
            dp[-1].append(_DpPoint(i * j, ALIGN_CRT))

    # Initialize DP table
    for i in range(1, hyp_len + 1):
        dp[0][i].cost = dp[0][i - 1].cost + COST_INS
        dp[0][i].align = ALIGN_INS
    for i in range(1, ref_len + 1):
        dp[i][0].cost = dp[i - 1][0].cost + COST_DEL
        dp[i][0].align = ALIGN_DEL

    # Fill DP table
    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            if hyp[j - 1] == ref[i - 1]:
                min_cost = dp[i - 1][j - 1].cost
                min_align = ALIGN_CRT
            else:
                min_cost = dp[i - 1][j - 1].cost + COST_SUB
                min_align = ALIGN_SUB

            del_cost = dp[i - 1][j].cost + COST_DEL
            if del_cost < min_cost:
                min_cost = del_cost
                min_align = ALIGN_DEL

            ins_cost = dp[i][j - 1].cost + COST_INS
            if ins_cost < min_cost:
                min_cost = ins_cost
                min_align = ALIGN_INS

            dp[i][j].cost = min_cost
            dp[i][j].align = min_align

    # Backtrace to count operations
    crt = sub = ins = det = 0
    i = ref_len
    j = hyp_len
    align = []
    
    while i > 0 or j > 0:
        if dp[i][j].align == ALIGN_CRT:
            align.append((i, j, ALIGN_CRT))
            i -= 1
            j -= 1
            crt += 1
        elif dp[i][j].align == ALIGN_SUB:
            align.append((i, j, ALIGN_SUB))
            i -= 1
            j -= 1
            sub += 1
        elif dp[i][j].align == ALIGN_DEL:
            align.append((i, j, ALIGN_DEL))
            i -= 1
            det += 1
        elif dp[i][j].align == ALIGN_INS:
            align.append((i, j, ALIGN_INS))
            j -= 1
            ins += 1

    align.reverse()
    
    # Calculate WER
    total_errors = sub + det + ins
    wer = total_errors / ref_len if ref_len > 0 else 0.0
    
    # Return WER, insertion/deletion count (combined), substitution count, and alignments
    return wer, (ins + det), sub, align

# class WerInfo:
#     def __init__(self, ref, err, crt, sub, dele, ins, ali):
#         self.r = ref
#         self.e = err
#         self.c = crt
#         self.s = sub
#         self.d = dele
#         self.i = ins
#         self.ali = ali
#         r = max(self.r, 1)
#         self.wer = 100.0 * (self.s + self.d + self.i) / r
#
#     def __repr__(self):
#         s = f"wer {self.wer:.2f} ref {self.r:2d} sub {self.s:2d} del {self.d:2d} ins {self.i:2d}"
#         return s


# class WerStats:
#     def __init__(self):
#         self.infos = []
#
#     def add(self, wer_info):
#         self.infos.append(wer_info)
#
#     def print(self):
#         r = sum(info.r for info in self.infos)
#         if r <= 0:
#             print(f"REF len is {r}, check")
#             r = 1
#         s = sum(info.s for info in self.infos)
#         d = sum(info.d for info in self.infos)
#         i = sum(info.i for info in self.infos)
#         se = 100.0 * s / r
#         de = 100.0 * d / r
#         ie = 100.0 * i / r
#         wer = 100.0 * (s + d + i) / r
#         sen = max(len(self.infos), 1)
#         errsen = sum(info.e > 0 for info in self.infos)
#         ser = 100.0 * errsen / sen
#         print("-" * 80)
#         print(f"ref{r:6d} sub{s:6d} del{d:6d} ins{i:6d}")
#         print(f"WER{wer:6.2f} sub{se:6.2f} del{de:6.2f} ins{ie:6.2f}")
#         print(f"SER{ser:6.2f} = {errsen} / {sen}")
#         print("-" * 80)


# class EnDigStats:
#     def __init__(self):
#         self.n_en_word = 0
#         self.n_en_correct = 0
#         self.n_dig_word = 0
#         self.n_dig_correct = 0
#
#     def add(self, n_en_word, n_en_correct, n_dig_word, n_dig_correct):
#         self.n_en_word += n_en_word
#         self.n_en_correct += n_en_correct
#         self.n_dig_word += n_dig_word
#         self.n_dig_correct += n_dig_correct
#
#     def print(self):
#         print(f"English #word={self.n_en_word}, #correct={self.n_en_correct}\n"
#               f"Digit #word={self.n_dig_word}, #correct={self.n_dig_correct}")
#         print("-" * 80)


# def count_english_ditgit(ref, hyp, wer_info):
#     patt_en = "[a-zA-Z\.\-\']+"
#     patt_dig = "[0-9]+"
#     patt_cjk = re.compile(r'([\u4e00-\u9fff])')
#     n_en_word = 0
#     n_en_correct = 0
#     n_dig_word = 0
#     n_dig_correct = 0
#     ali = wer_info.ali
#     for i, token in enumerate(ref):
#         if re.match(patt_en, token):
#             n_en_word += 1
#             for y in ali:
#                 if y[0] == i + 1 and y[2] == ALIGN_CRT:
#                     j = y[1] - 1
#                     n_en_correct += 1
#                     break
#         if re.match(patt_dig, token):
#             n_dig_word += 1
#             for y in ali:
#                 if y[0] == i + 1 and y[2] == ALIGN_CRT:
#                     j = y[1] - 1
#                     n_dig_correct += 1
#                     break
#         if not re.match(patt_cjk, token) and not re.match(patt_en, token) \
#                 and not re.match(patt_dig, token):
#             print("[WiredChar]:", token)
#     return n_en_word, n_en_correct, n_dig_word, n_dig_correct


def calculate_wer_from_json(input_json_path: str, output_txt_path: str) -> float:
    """
    Calculate WER from JSON file and save the results
    
    Args:
        input_json_path: Input JSON file path
        output_txt_path: Output TXT file path
        
    Returns:
        Average WER value
    """
    # Read JSON file
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data_all = json.load(f)
    
    total_wer = 0.0
    total_ins_del = 0.0
    total_sub = 0.0

    valid_items = 0
    results = []
    
    # Process each entry
    for data in data_all:
        for key, item in data.items():
            ref_text = item.get("text", "")
            hyp_text = item.get("hpy_text", "")

            if not ref_text or not hyp_text:
                print(f"Skipping entry {key}: Missing text or hpy_text field")
                continue
            
            # Choose WER calculation method based on language
            language = item.get("language", "english").lower()
            
            if language == "japanese":
                wer, ins_del, sub, align = jap_wer(ref_text, hyp_text)
            elif language == "chinese":
                wer, ins_del, sub, align = en_zh_wer(ref_text, hyp_text)
            elif language in ["thai", "korean"]:
                wer_calculator = CommonWer()
                if language == "thai":
                    wer, ins_del, sub, align = wer_calculator.thai_wer(ref_text, hyp_text)
                else:
                    wer = wer_calculator.kor_wer(ref_text, hyp_text)
            else:  # Default: English/Chinese processing
                wer, ins_del, sub, align = en_zh_wer(ref_text, hyp_text)
        
            # Record result
            result_line = f"{key}|{ref_text}|{hyp_text}|{wer:.4f}|{ins_del}|{sub}"
            results.append(result_line)
            total_wer += wer
            total_ins_del += ins_del 
            total_sub += sub
            valid_items += 1
    
    # Calculate average WER
    avg_wer = total_wer / valid_items if valid_items > 0 else 0.0
    
    # Write results to file
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        # Write each result line
        f.write("\n".join(results))
        # Write statistics
        f.write(f"\n\nStatistics:\n"
        f"• Average WER: {avg_wer:.2%} \n"
        f"• Total Insertions/Deletions: {total_ins_del} \n"
        f"• Total Substitutions: {total_sub} \n"
        f"• Total Errors: {total_ins_del + total_sub}")
    
    print(f"Processing done, results saved to {output_txt_path}")
    print(f"Average WER: {avg_wer:.4f}")
    print(f"Insertions and deletions: {total_ins_del}")
    print(f"Substitutions: {total_sub}")
    
    return avg_wer, total_ins_del, total_sub

import sys, os

def main():
    # x = "インターネットで敵対的環境コースについて検索すると、おそらく現地企業の住所が出てくるでしょう。"
    # y = "インターネットで 敵対的環境コース について検索すると おそらく現地企業の住所が出てくるでしょう"
    # print(ja_wer(x, y))
    # x = "メインステージの音楽が終わっても、 テスティバルにはヨロコ ウソクマで 演奏を流し続けるセクションがあるかもしれないことを覚えて置いて下さい"
    # y = "メインステージの音楽が終わっても フェスティバルには夜遅くまで演奏を流し続けるセクションがあるかもしれないことを覚えておいてください"
    # print(ja_wer(x, y))
    x = sys.argv[1] # ref
    y = sys.argv[2] # hyp
    print((x, y))
    print(en_zh_wer(x, y))
    # wer_stat = CommonWer()
    # print(wer_stat.kor_wer(x, y))
    # x = "ชาวบารีขึ้นชื่อในเรื่องความเห็นแก่ตัว ยาคายและหญิงยาโส"
    # y = "ชาวปารีสขึ้นชื่อในเรื่องความเห็นแก่ตัว หยาบคาย และหยิ่งยโส"
    # x = "ชาวบารีขึ้นชื่อในเรื่องความเห็นแก่ตัว"
    # y = "ชาวปารีสขึ้นชื่อในเรื่องความเห็นแก่ตัว"

    # print(wer_stat.thai_wer(y, x))
    # #
    # # x = "다리미 수직 간격은 15미터이며 공사는 2011년 8월에 마무리되었으며, 해당 다리의 통행금는 2017년 3월까지이다."
    # # y = "다리 밑 수직 간격은 15미터이며 공사는 2011년 8월에 마무리되었으며 해당 다리의 통행금지는 2017년 3월까지이다"
    # # print(wer_thai.thai_wer(y, x))
    # # print(jap_kor_wer(y, x))
    
    # # x = "インターネットで敵対的環境コースについて検索すると、おそらく現地企業の住所が出てくるでしょう。"
    # # y = "インターネットで 敵対的環境コース について検索すると おそらく現地企業の住所が出てくるでしょう"
    # # print(wer_thai.thai_wer(y, x))
    # # print(jap_kor_wer(y, x))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Speaker verification similarity calculation script")
    parser.add_argument("--input_json", required=True, help="Input JSON file path")
    parser.add_argument("--output_txt", required=True, help="Output TXT file path")
    args = parser.parse_args()
    calculate_wer_from_json(args.input_json, args.output_txt)
