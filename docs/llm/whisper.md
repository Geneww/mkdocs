openai whisper

## Setup

We used Python 3.9.9 and [PyTorch](https://pytorch.org/) 1.10.1 to train and test our models, but the codebase is expected to be compatible with Python 3.8-3.11 and recent PyTorch versions. The codebase also depends on a few Python packages, most notably [OpenAI's tiktoken](https://github.com/openai/tiktoken) for their fast tokenizer implementation. You can download and install (or update to) the latest release of Whisper with the following command:

```
pip install -U openai-whisper
```

Alternatively, the following command will pull and install the latest commit from this repository, along with its Python dependencies:

```
pip install git+https://github.com/openai/whisper.git 
```

To update the package to the latest version of this repository, please run:

```
pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
```

It also requires the command-line tool [`ffmpeg`](https://ffmpeg.org/) to be installed on your system, which is available from most package managers:

```shell
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```



whisper api

```
def transcribe(
    model: "Whisper",
    audio: Union[str, np.ndarray, torch.Tensor],
    *,
    verbose: Optional[bool] = None,
    temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -1.0,
    no_speech_threshold: Optional[float] = 0.6,
    condition_on_previous_text: bool = True,
    initial_prompt: Optional[str] = None,
    word_timestamps: bool = False,
    prepend_punctuations: str = "\"'“¿([{-",
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
    **decode_options,
): 
   """
    Transcribe an audio file using Whisper

    Parameters
    ----------
    model: Whisper
        The Whisper model instance

    audio: Union[str, np.ndarray, torch.Tensor]
        The path to the audio file to open, or the audio waveform

    verbose: bool
        Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays minimal details. If None, does not display anything

    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successively used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed

    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed

    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent

    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

    word_timestamps: bool
        Extract word-level timestamps using the cross-attention pattern and dynamic time warping,
        and include the timestamps for each word in each segment.

    prepend_punctuations: str
        If word_timestamps is True, merge these punctuation symbols with the next word

    append_punctuations: str
        If word_timestamps is True, merge these punctuation symbols with the previous word

    initial_prompt: Optional[str]
        Optional text to provide as a prompt for the first window. This can be used to provide, or
        "prompt-engineer" a context for transcription, e.g. custom vocabularies or proper nouns
        to make it more likely to predict those word correctly.

    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances

    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """

model：Whisper
Whisper模型实例

audio：Union[str，np.ndarray，torch.Tensor]
要打开的音频文件的路径或音频波形

verbose：bool
是否在控制台上显示解码的文本。如果为True，则显示所有细节，
如果为False，则显示最少的细节。如果为None，则不显示任何内容

temperature：Union[float，Tuple[float，...]]
采样的温度。它可以是温度元组，这些元组将根据compression_ratio_threshold或logprob_threshold的失败情况依次使用。

compression_ratio_threshold：float
如果gzip压缩比超过此值，则视为失败

logprob_threshold：float
如果样本标记的平均对数概率低于此值，则视为失败

no_speech_threshold：float
如果无语音概率高于此值，并且样本标记的平均对数概率低于logprob_threshold，则将该段视为无声。

condition_on_previous_text：bool
如果为True，则将模型的先前输出作为下一个窗口的提示；禁用可能会使文本在窗口之间不一致，但模型变得不太容易陷入失败循环，例如重复循环或时间戳失去同步。

word_timestamps：bool
使用交叉注意力模式和动态时间扭曲提取单词级时间戳，并在每个段中包含每个单词的时间戳。

prepend_punctuations：str
如果word_timestamps为True，则将这些标点符号与下一个单词合并

append_punctuations：str
如果word_timestamps为True，则将这些标点符号与前一个单词合并

initial_prompt：Optional[str]
可选的文本，用于为第一个窗口提供提示。这可以用于提供或“提示工程”转录的上下文，例如自定义词汇或专有名词，以使更容易正确预测这些单词。

decode_options：dict
用于构建DecodingOptions实例的关键字参数

返回

包含结果文本（“text”）和段落级别细节（“segments”）以及检测到的口语语言（“language”）的字典，当decode_options [“language”]为None时。
"""
```

