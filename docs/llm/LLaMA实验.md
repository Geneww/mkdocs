# 中文LLaMA&Alpaca大语言模型实验

步骤

llama原始模型转换为Transformers  huggingface格式

1.lora pt

2.lora指令精调

3.模型合并



在**大模型词表扩充必备工具SentencePiece**一文中，我们提到了在目前开源大模型中，LLaMA无疑是最闪亮的星。但是，与 ChatGLM-6B 和 Bloom 原生支持中文不同。 LLaMA 原生仅支持 Latin 或 Cyrillic 语系，对于中文支持不是特别理想。原版LLaMA模型的词表大小是32K，而多语言模型（如：XLM-R、Bloom）的词表大小约为250K。以中文为例，LLaMA词表中的中文token比较少（只有几百个）。这将导致了两个问题：

- LLaMA 原生tokenizer词表中仅包含少量中文字符，在对中文字进行tokenzation时，一个中文汉字往往被切分成多个token（2-3个Token才能组合成一个汉字），显著降低编解码的效率。
- 预训练中没有出现过或者出现得很少的语言学习得不充分。

为了解决这些问题，我们可能就需要进行中文词表扩展。比如：在中文语料库上训练一个中文tokenizer模型，然后将中文 tokenizer 与 LLaMA 原生的 tokenizer 进行合并，通过组合它们的词汇表，最终获得一个合并后的 tokenizer 模型。

而国内**[Chinese-LLaMA-Alpaca](https://link.zhihu.com/?target=https%3A//github.com/ymcui/Chinese-LLaMA-Alpaca)**开源项目详细说明了词表扩展、模型预训练和模型指令精调的整个过程。本文将分享**[Chinese-LLaMA-Alpaca](https://link.zhihu.com/?target=https%3A//github.com/ymcui/Chinese-LLaMA-Alpaca)**是如何从0到1进行中文词表扩充、模型预训练和微调的整个过程。

## **环境搭建**

基础环境配置如下：

- **操作系统:** Ubuntu 18.04
- **CPUs:** 单个节点具有 384GB 内存的 Intel CPU，物理CPU个数为2，每颗CPU核数为20
- **GPUs:** 4 卡 A800 80GB GPUs
- **Python:** 3.10 (需要先升级OpenSSL到1.1.1t版本（**[点击下载OpenSSL](https://link.zhihu.com/?target=https%3A//www.openssl.org/source/openssl-1.1.1t.tar.gz)**），然后再编译安装Python)，**[点击下载Python](https://link.zhihu.com/?target=https%3A//www.python.org/downloads/source/)**
- **NVIDIA驱动程序版本:** 525.105.17，根据不同型号选择不同的驱动程序，**[点击下载](https://link.zhihu.com/?target=https%3A//www.nvidia.com/Download/index.aspx%3Flang%3Den-us)**。
- **CUDA工具包:** 11.6，**[点击下载](https://link.zhihu.com/?target=https%3A//developer.nvidia.com/cuda-toolkit-archive)**
- **cuDNN:** 8.8.1.3_cuda11，**[点击下载](https://link.zhihu.com/?target=https%3A//developer.nvidia.com/cudnn)**

为了便于复现，本文选择使用Doker镜像进行环境搭建。

首先，下载对应版本的Pytorch镜像。

```text
docker pull pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
```

镜像下载完成之后，创建容器。

```text
docker run -dt --name pytorch1131_cu116 --restart=always --gpus all \
-v /home/wuhw/projects/AIGC:/workspace \
-p 12002:22 \
-p 12003:8000 \
-w /workspace \
--shm-size 64g \
pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel \
/bin/bash
```

进入容器。

```text
docker exec -it pytorch1131_cu116_v1 bash
```

### 环境准备

安装vim、git、git-lfs\sshd

```shell
apt-get update
#apt-get upgrade
apt-get install vim
apt-get install git
apt-get install git-lfs
apt-get install openssh-server


```

配置sshd

```shell
# 启动sshd
/etc/init.d/ssh start
# 验证服务是否开启
ps -e | grep sshd
# 开启ssh允许root用户访问登录
# 编辑ssh配置文件/etc/ssh/sshd_config
vim /etc/ssh/sshd_config
# 找到以下行：PermitRootLogin no将其改为：
PermitRootLogin yes
# 重启ssh服务以使更改生效：
service ssh restart
```



安装依赖库。

```shell
pip install transformers==4.28.1 sentencepiece==0.1.97 google protobuf deepspeed -i https://pypi.tuna.tsinghua.edu.cn/simple  --trusted-host pypi.tuna.tsinghua.edu.cn
```

从源码安装 Peft，由于 Peft 库变动频繁，从 `commit id` 为 `13e53fc` 进行安装。

```shell
git clone https://github.com/huggingface/peft.git
git checkout 13e53fc
pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple  --trusted-host pypi.tuna.tsinghua.edu.cn
```

## **远程开发准备**

### pycharm连接docker容器开发

## **代码、模型、数据集准备**

### **代码准备**

下载**[Chinese-LLaMA-Alpaca](https://link.zhihu.com/?target=https%3A//github.com/ymcui/Chinese-LLaMA-Alpaca)**官网代码。

```bash
# 3e2f2529
git clone https://github.com/ymcui/Chinese-LLaMA-Alpaca.git
```

### **模型权重 及 Tokenizer 准备**

将LLaMA原始权重文件转换为Transformers库对应的模型文件格式。具体可参考之前的文章：**从0到1复现斯坦福羊驼（Stanford Alpaca 7B）** 。

```shell
TypeError: Descriptors cannot not be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
If you cannot immediately regenerate your protos, some other possible workarounds are:
 1. Downgrade the protobuf package to 3.20.x or lower.
 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).

```

torchrun命令找不到

```shell
# pytorch1131_cu116镜像内默认环境位于conda base内 /opt/conda/bin
export PATH=${PATH}:/opt/conda/bin
```



pip install datasets -i https://pypi.douban.com/simple --trusted-host pypi.douban.com

pip install scikit-learn -i https://pypi.douban.com/simple --trusted-host pypi.douban.com

pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

精调模型

```shell
run_sft.sh
```





```shell
python merge_llama_with_chinese_lora.py \
    --base_model /workspace/model/llama-7b-hf \
    --tokenizer_path /workspace/Chinese-LLaMA-Alpaca/scripts/merged_tokenizer_hf \
    --lora_model /workspace/outputs/lora \
    --output_type huggingface \
    --output_dir /workspace/outputs/book-merge-hf
```

合并模型时如果报错config.json问题，需要保证base_model包含

adapter_config.json、adapter_model.bin、config.json







合并精调模型

```shell
python merge_llama_with_chinese_lora.py \
     --base_model /workspace/model/llama-7b-hf \
     --tokenizer_path /workspace/outputs/pt_lora_model,/workspace/outputs/instruction/sft_lora_model \
     --lora_model /workspace/outputs/pt_lora_model,/workspace/outputs/instruction/sft_lora_model \
     --output_type huggingface \
     --output_dir /workspace/output/book-alpaca-merge-hf
```



合并plus

```shell
python merge_llama_with_chinese_lora_copy.py \
    --base_model /workspace/model/llama-7b-hf \
    --lora_model /workspace/model/chinese-llama-plus-lora-7b,/workspace/model/chinese-alpaca-plus-lora-7b \
    --output_type huggingface \
    --output_dir /workspace/outputs/merge_lora_hf
```



推理

```shell
python inference_hf.py \
     --base_model /workspace/outputs/merge_lora_hf \
     --with_prompt \
     --interactive
```

inference相关参数

```json
max_length：
	即限定生成的最大长度，这里的长度指的token的长度。并且是最大的长度，在这个长度之内的其他长度的句子也是可以被生成的。
min_length：
	与max_length相反，限定生成的最小长度。在此不做赘述。

early_stopping：
	early_stopping 参数用于控制是否启用 early stopping 策略来提前终止生成序列的过程。在启用 early stopping 策略后，会在生成的序列中检查停止标记（例如，句号、问号、感叹号等），一旦发现该标记，就会将生成的序列剪裁到该标记位置，停止生成。这样，就可以确保生成的序列是自然的、完整的、有意义的，并且符合预期的格式和长度。

max_time：
	运行的最长时间。

do_sample：
	do_sample是一个布尔值，是指是否使用采样（sampling）方法来生成文本。采样是一种生成文本的方法，它从模型输出的概率分布中随机采样一个 token 作为下一个生成的 token，具有一定的随机性和多样性，因此生成的文本可能更加多样化，而不是完全按照概率分布中的概率来选择下一个 token。
	具体来说，如果设置 do_sample=True，那么在生成文本时就会使用采样方法。在采样时可以指定一些参数，例如 temperature、top_p 等，这些参数会影响采样方法的结果，从而影响生成文本的多样性和质量。<br> 如果设置 do_sample=False，那么就会使用贪心算法（greedy decoding）来生成文本，即每次选择模型输出概率最大的 token 作为下一个 token，这种方法生成的文本可能会比较单一和呆板。
	do_sample的设置影响后续一些参数的设置，有些并不能兼容使用。

num_beams ：
	“num_beams"是在进行文本生成时的一个参数。它是指在生成一个序列的时候，预测模型同时生成的候选解的数量。在Beam Search生成文本的方法中，预测模型会对于每一个时间步，生成一定数量的候选解，选取其中最优解进行下一步预测，直到完成整个序列的生成。这个数量即为"num_beams”。<br> "num_beams"设置较高，可以增加生成的候选解的数量，从而得到更多可能性，但是会同时增加计算代价。因此，通常需要根据任务的需求，合理选择"num_beams"的值。

num_beam_groups：
	num_beam_groups是在文本生成中使用的参数，它代表在生成给定序列的下一个单词时要使用的束数组数。一般来说，束搜索是一种生成文本的技术，它通过对生成的文本的分数进行排序来选择最佳的文本生成结果。<br> 在num_beam_groups参数中，如果指定的数字为1，那么将使用单一的束搜索组，如果指定的数字为2或更多，则将使用多个束搜索组。每个束搜索组将独立进行搜索，并通过在多个束搜索组中并行进行搜索来生成更多的文本生成结果。这样可以显著提高生成效率。<br> 总的来说，num_beam_groups参数可以帮助控制生成的文本数量和质量，具体的值可以根据需求进行调整。

penalty_alpha ：
	penalty_alpha是在Transformer模型中生成文本时用于控制生成的文本多样性的参数。它控制生成的文本的多样性和翻译的目标序列的近似程度之间的权衡。当penalty_alpha值越大时，模型生成的文本变得更加多样性，生成的文本与目标序列的近似程度可能会变得更加差。反之，当penalty_alpha值越小时，模型生成的文本变得更加相似于目标序列，多样性可能会变得更加差。因此，根据生成文本的目的和要求，用户可以调整penalty_alpha的值以获得更好的结果。

temperature ：
	temperature 是生成文本模型（如Transformer模型）在生成文本时所使用的参数。<br> 模型通过对每一个单词的概率进行计算，并选择具有最高概率的单词作为下一个单词，以生成文本序列。为了生成多样性更高的文本，通常使用高斯分布重新调整单词的概率值。temperature 参数定义了重新调整的度量。
	如果 temperature 的值为1，则没有任何调整；如果其值比1大，则会生成更加随机的文本；如果其值比1小，则生成的文本更加保守。通常需要多次试验来找到最适合特定数据集的 temperature 值

top_k ：用于top-k筛选的最高概率词汇表标记的数量。

top_p：
	“top_p” 是一个用于限制生成文本中词语选择的概率分布的参数。它代表了仅选择概率分布中前p%大的词语，而舍弃剩余的词语。通俗地说，它用于约束生成文本中词语的数量，防止生成过多的词语，并且只选择最可能的词语。
	具体地，假设我们有一个模型，它预测了下一个单词的概率分布，如果我们将 top_p 设置为0.9，那么我们将只选择概率分布中前 90% 的词语。因此，如果我们生成的词语不在前90%，那么将不会考虑这个词语。这样做可以降低生成的词语数量，同时可以保证生成的词语更可能是正确的。

typical_p：“top_p” 和 “top_k” 是一组限制生成文本中每个词选择的概率的参数。“top_p” 控制生成的文本中，每个词选择的概率必须在最高概率的前 p 的词中，而不是选择来自整个词汇表中的任何词。<br> “typical_p” 是一个新加入的参数，与 “top_p” 类似，但是它限制的是生成的文本中每个词概率的分布，而不是概率的确切值。与 “top_p” 不同的是，“typical_p” 可以产生更长的文本序列，因为它允许更多的词汇选择。<br> 使用这些参数可以限制生成的文本，从而产生更有意义的，更符合目标的生成文本。

epsilon_cutoff：
	epsilon_cutoff 是一个超参数，它在文本生成中用于控制如何选择词语。它是一个数字，代表了从预测分布中删除词语的阈值。<br> 举个例子，如果我们设置 epsilon_cutoff = 0.01 ，那么预测分布中低于0.01的所有词语将不会被选择，只有剩下的更高的分数的词语将被选中。通过调整这个参数，可以控制预测的词语数量，从而控制生成的文本的精细程度。

eta_cutoff：
	“eta_cutoff” 是 GPT-2 文本生成中的一个参数，它控制生成的序列的多样性。它的作用是决定是否要在符合预定要求的前提下生成更加出奇不意的序列。<br> eta_cutoff 表示当前生成的序列与最高概率的第二高概率的比值，如果比值大于 eta_cutoff，则继续生成下一个字符；如果比值小于 eta_cutoff，则停止生成。<br> 通过调整 eta_cutoff 的值，可以控制生成的序列的多样性：如果 eta_cutoff 较低，则生成的序列会更加多样化；如果 eta_cutoff 较高，则生成的序列会更加保守，更接近训练数据。

diversity_penalty：
	diversity_penalty 参数是指对生成的文本多样性的惩罚因子。当 diversity_penalty 设置为正数时，它会惩罚生成的文本里排在前面的单词出现的频率过高的情况，进而提高文本的多样性。较小的 diversity_penalty 值可以使生成的文本更加保守，而较大的值则可以提高文本的创造性。需要注意的是，对文本多样性的惩罚因子过大也可能导致生成的文本质量下降，因此选择正确的 diversity_penalty 值需要根据具体应用场景和需求进行调整。

repetition_penalty ：
	“repetition_penalty” 是用于文本生成中的参数，用于惩罚重复的单词和句子。默认值为1.0，其中较高的值意味着更强的惩罚，生成的文本中将出现更少的重复。如果取值为0，则没有惩罚，生成的文本可能包含大量重复的内容。<br> 在实际使用中，通常需要对该参数进行调整，以根据具体需求在生成的文本中达到适当的平衡，例如在一些任务中，过多的重复可能不是理想的，因此需要将repetition_penalty设置为较高的值；而在其他任务中，更多的重复可能是有意义的，因此需要将repetition_penalty设置为较低的值。

encoder_repetition_penalty ：
	encoder_repetition_penalty是Transformers库中用于文本生成任务的一个参数。它决定了对模型对源文本进行重复生成文本的惩罚程度。当设置了较高的值时，模型会对生成出来的文本与源文本重复的部分进行惩罚，从而有可能生成出不同于源文本的新文本。设置的值越大，模型对重复的文本的惩罚程度就越高。如果不设置此参数，则默认为1.0。

length_penalty：
	length_penalty是一个因子，用于调整生成的序列长度的影响。在生成的文本中，长度是相对的。例如，一篇长文章与另一篇短文章相比可能更有价值。为了考虑这种情况，可以在生成文本时考虑长度，通过设置长度惩罚因子。<br> 具体来说，length_penalty是一个根据生成的序列长度动态调整生成序列的分数的因子。如果length_penalty的值较小，则长序列的分数会比较低，因此生成的序列可能较短。如果length_penalty的值较大，则长序列的分数可能更高，因此生成的序列可能更长。在适当的情况下，可以通过调整length_penalty的值来影响生成文本的长度。

no_repeat_ngram_size：
	no_repeat_ngram_size 参数指定了生成文本时需要避免的重复文本的大小。该参数指示在生成的文本中，最多允许多少个词与原始文本中的词进行重复。如果该值设置为 3，则生成的文本中的任何一段长度为三个词的子序列都不能与原始文本的任何子序列重复。设置该值可以防止生成的文本中出现重复词汇，因此可以提高生成文本的多样性。

bad_words_ids：
	bad_words_ids 是 Text Generation 中的一个参数，用来表示需要被禁止出现在生成的文本中的单词（通常是一些不当语言、种族歧视等）的 ID。这些单词在生成文本时将会被忽略，以确保生成的文本在内容和语言方面符合某些道德和社会规范。<br> 例如，假设你使用的是 BERT 模型，并且在你的词汇表中有一些不恰当的词。如果你不希望这些词在生成的文本中出现，你可以将它们的 ID 添加到 bad_words_ids 列表中。生成的文本中将不会再出现这些词的 ID。

force_words_ids：
	force_words_ids参数可以用于强制模型在生成的文本中包含特定的单词或子序列。具体来说，它是一个包含单词或子序列id的列表，表示模型必须生成包含这些单词或子序列的文本。<br> 如果给定的单词或子序列在生成的过程中无法满足约束条件，则会尝试进行多次重新采样，直到满足条件或达到最大重试次数。如果最终无法生成满足条件的文本，则会引发一个异常。
	在一些应用场景中，force_words_ids可以用于约束生成的文本必须满足某些关键词或要求，例如在生成广告文案时强制包含某些销售亮点词汇。

renormalize_logits：
	在 Text Generation 中，renormalize_logits 是一个控制标准化逻辑概率的布尔值。当设置为 True 时，它会将模型中的 logits 规范化为概率分布，以便于计算损失和生成样本。在某些情况下，如果 logits 不规范化，可能会出现训练困难或生成不稳定的问题，因此可以将 renormalize_logits 设置为 True 来解决这些问题。<br> 需要注意的是，将 renormalize_logits 设置为 True 会对生成的样本产生一定的影响。具体来说，它会使生成的样本更加平滑，这意味着它们可能会失去一些原始模型的细节。因此，在某些情况下，可以将 renormalize_logits 设置为 False，以便生成更加准确和详细的样本。但是，这也可能导致训练问题和不稳定的生成结果，需要在使用时谨慎权衡。

constraints ：
	constraints参数用于对生成文本的内容进行限制，以满足特定的条件或要求。该参数可以设置为一个约束函数的列表，每个函数都返回一个布尔值，表示输入序列是否符合特定的限制条件。<br> 具体来说，约束函数可以检查生成的序列是否包含特定的单词或短语，或者是否满足特定的语法规则。例如，可以使用一个约束函数来检查生成的序列是否符合特定的主题，或者是否包含不允许出现的单词。
forced_bos_token_id：时，forced_bos_token_id是一个可选参数，用于指定要强制用作开始标记的token ID，即在生成文本时将其添加到生成序列的开头。
	默认情况下，Transformers使用模型的开始标记作为生成序列的第一个token，但是在某些情况下，可能需要强制使用其他token作为开始标记。例如，在对话生成任务中，我们可能希望生成序列的开头始终是某个固定的问句。

remove_invalid_values ：在文本生成任务中，模型可能会生成一些无效的值，如NaN或Inf。如果这些无效值被允许出现，可能会导致模型出现错误或不可预测的结果。因此，如果在模型输出中检测到此类无效值，可以选择将其移除。在Transformers库中，可以使用remove_invalid_values参数来控制是否移除无效值。当remove_invalid_values设置为True时，任何无效值都会被移除。当设置为False时，这些无效值将保留在输出中。

exponential_decay_length_penalty：
	exponential_decay_length_penalty参数指定在生成文本时如何惩罚长度。该参数的默认值为0.0，表示不应用长度惩罚。当exponential_decay_length_penalty参数为非零值时，将按指数衰减的方式对生成文本的长度进行惩罚。<br> 衰减的速度由decay_rate参数决定。例如，如果decay_rate设置为0.5，对于长度为N的文本，生成器会给该文本的得分乘以0.5^N。这样做可以有效防止生成器生成过长的文本。<br> exponential_decay_length_penalty参数适用于需要生成特定长度文本的任务，比如对话系统中对话长度的控制。通常情况下，生成较短的文本可以更加流畅自然，而过长的文本可能会出现语法错误或者逻辑混乱等问题。
suppress_tokens：suppress_tokens参数可以用来抑制模型生成指定的标记(token)。具体来说，当模型生成一个标记时，如果该标记在suppress_tokens中指定了，那么模型将会跳过该标记，继续生成下一个标记。这个参数通常用于特殊情况下的文本生成任务，比如过滤掉一些不合适的词汇或者指定一些强制要求的单词。

num_return_sequences：
	num_return_sequences 参数用于控制模型生成的文本序列的数量。默认情况下，模型将生成一个完整的文本序列，但是你可以设置 num_return_sequences 参数为一个大于 1 的整数，以生成多个文本序列。这对于生成多个相关但不完全相同的文本序列很有用，例如为一组给定的提示或问题生成多个可能的答案。

output_attentions：
	output_attentions 是一个参数，用于指定是否返回注意力矩阵。注意力机制是深度学习中常用的一种机制，用于在给定输入时，自动分配不同的权重给不同的输入项，并计算其与输出的关联度。<br> 在文本生成任务中，注意力机制可以用来让模型了解输入文本的各部分之间的关联，从而更好地生成上下文相关的输出。如果 output_attentions 被设置为 True，那么在调用 generate() 方法时，模型将返回一个元组 (output, attentions)。其中，output 是生成的文本序列，attentions 是一个元组，包含每个层次的注意力矩阵。注意力矩阵是一个三维张量，其维度为 [batch_size, num_heads, sequence_length, sequence_length]，其中，batch_size 表示输入的样本数，num_heads 表示注意力头数，sequence_length 表示输入序列的长度。
```



inference

```shell
docker run --rm -it -p 8010:8000 -v /path/to/models:/models -e MODEL=/models/ggml-model-name.bin ghcr.io/abetlen/llama-cpp-python:latest
```

