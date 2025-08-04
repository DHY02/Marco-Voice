
# 指标测试

Evaluation/cal_metrix.sh path_to_json_file path_to_output

#### 单文件格式
每个JSON文件应为UTF-8编码，包含字典结构：
```json
{
  "<utt_id>": {
    "text": "文本内容",
    "audio_path": "/path/to/audio/file.wav",
    "style_label": "风格标签",
    "language_id":  'ch',
  },
  // 更多utterance...
}

## 示例文件

### 输入示例
```json
{
  "uttr_001": {
    "text": "今天天气真好",
    "audio_path": "data/audio/happy_001.wav",
    "style_label": "happy",
    "language_id":  'ch',
  },
  "uttr_002": {
    "text": "这让我很失望",
    "audio_path": "data/audio/sad_005.wav",
    "style_label": "sad",
    "language_id":  'ch'
  }
}



