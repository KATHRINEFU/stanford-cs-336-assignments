# CS336 学习笔记

## Assignment 1：BPE Tokenizer（2026-11）

---

## 一、AI / LLM 知识

### 1. 为什么用 byte-level BPE

- Unicode 有约 15 万个字符，直接当词表太大、太稀疏。
- 先用 **UTF-8** 把文本编码成字节：初始词表只有 **256 个字节**，任何文本都能表示，**不会有 OOV（未登录词）**。
- 但纯字节序列太长，计算量大。**BPE** 是一个折中：反复把"最常一起出现的两个 token"合并成一个新 token，用更大的词表换更短的序列。

### 2. BPE 训练三步

1. **词表初始化**：256 个字节 + special tokens（比如 `<|endoftext|>`）。
2. **Pre-tokenization**：用 GPT-2 的正则 `PAT` 把文本切成"类似单词"的块，统计每个块出现的次数，存成 `{(b'l', b'o', b'w'): 5, ...}`。
   - 合并**不会跨越块的边界**，所以 `dog!` 和 `dog.` 能共享 `dog`。
   - 不用每次都扫全文，只要按块的次数加权统计。
3. **Merge 循环**：统计所有相邻 pair 的次数（按块的次数加权），选次数最多的，把它合成一个新 token，加进词表，重复直到达到 `vocab_size`。

### 3. 容易忘的细节

- **打平时选字典序更大的 pair**：`('s','t') > ('e','s')`，所以在 PDF 例子里第一轮合并的是 `st`，**不是** `es`。（我一开始注释里写错了。）
- PDF 例子的合并顺序：`s t → e st → o w → l ow → w est → n e → ...`，可以用来验证实现。
- **Special tokens**：pre-tokenize 之前要先用 `re.split` 按 special token **切开**（记得 `re.escape`），保证合并不会跨越文档边界。
- **多进程分块**：chunk 的边界要落在 special token 上（`find_chunk_boundaries`）。
- **合并时从左到右、不重叠**：命中后跳 2 格。比如合并 `(a, a)`，`a a a` 的结果是 `(aa, a)`。
- **被合并的 pair，合并后计数一定变成 0**：因为从左到右扫描时，所有相邻的 `x y` 都会被合并，不可能剩下。

### 4. 性能优化（`test_train_bpe_speed` 失败的原因）

**问题**：原来的 `apply_merge` 每一轮都**遍历所有词**、重建整个 `pretoken_counter`，但真正包含被合并 pair 的词只有几个。

**优化思路：**

| 数据结构 | 作用 |
|---|---|
| `words`：list | 第 i 个位置存第 i 个词当前的切分。**用位置 i 当这个词不变的"身份"** |
| `word_counts`：list | 和 `words` 平行，存次数 |
| `pair_counter`：dict | pair → 总次数 |
| `pair_to_words`：dict[pair, set[int]] | **反向索引**：pair → 包含它的词的位置 |
| `max_heap` | 找最大的 pair |

- 合并时只从 `pair_to_words` 拿受影响的词，**原地**修改 `words[i]`。
- 更新计数用"**先把旧词的 pair 全减，再把新词的 pair 全加**"：简单、不会错，重复 pair 和重叠情况都自动处理。`pair_to_words` 也用同样的方式同步更新。
- 为什么一定要保存每个词的切分？因为 `pair_counter` 只有数字，**丢失了"谁挨着谁"的信息**，算不出合并后新产生的 pair（比如 `(e, st)`）。
- 结果：0.92 秒 → 约 0.25 秒，输出完全一致。

### 5. Lazy deletion（堆的延迟删除）

`heapq` 删不掉中间的条目，所以：

- **计数一变，就 push 一条新条目**，旧的不删。
- pop 出来后检查：`pair_counter.get(pair) != -count` → 说明是**过期条目**，跳过。
- 为什么不会选错：每个还存在的 pair 在堆里**一定有一条**最新条目；排在它前面的要么是过期的（被跳过），要么是真正更大的。
- 同一个 pair 有两条"都对得上"的条目也没关系：第一条被合并后，这个 pair 计数变成 0、被删掉，第二条 `.get()` 返回 `None`，被跳过。
- 打平时 `ReversePair` 只看 pair 本身，和条目新旧无关，所以 tie-break 也是对的。

---

## 二、Python 基础

### 1. Counter / dict

- `Counter` 访问不存在的 key 返回 **0**，所以 `counter[k] += n` 不会报错。
- `dict.keys()` / `.values()` / `.items()` 不要混：
  - ❌ 我写过 `list(counter.items())` 当作次数列表 → 得到的是 `(key, value)` 元组，后面 `int + tuple` 报 `TypeError`。
  - ✅ 次数应该用 `.values()`。
- `merged[new_key] += count` 在这里和 `=` 结果一样，因为不同词合并后不会变成同一个 tuple（字节串不变、合并顺序相同）。`+=` 是防御性写法。
- `dict.pop(key)`：**删掉 key，并返回它的 value**。适合"取出来处理掉"。
- ⚠️ `defaultdict` 用 `d[key]` 访问**不存在的 key 会自动新建**。key 被 pop 掉之后，再 `d[key].discard(...)` 会悄悄把它加回来（变成空 set）。

### 2. set 和 list 的方法不一样

| 操作 | list | set |
|---|---|---|
| 添加 | `.append(x)` | `.add(x)` |
| 删除指定元素 | `.remove(x)` / `.pop(i)` 按位置 | `.remove(x)`（不存在会报错）/ `.discard(x)`（不存在不报错） |
| `.pop()` | 可以传位置 | **不接受参数**，随机弹出一个 |

- ❌ 我把 `defaultdict(list)` 改成 `defaultdict(set)` 之后，还在用 `.append` / `.pop(index)`。
- 反向索引用 **set** 的原因：`(a, a, a)` 里 `(a, a)` 出现两次，用 list 会把同一个词记两次、处理两次。
- 删除时用 `.discard`：同一个词里重复的 pair 会被删两次，第二次元素已经不在了。

### 3. 边遍历边修改会报错

```
RuntimeError: Set changed size during iteration
```

- ❌ `idxs = d[pair]` **没有复制**，`idxs` 和 `d[pair]` 是**同一个对象**。一边 `for` 遍历它，一边在循环里 `discard`，就会报错。
- ✅ 方法一：遍历副本 `d[pair].copy()`。
- ✅ 方法二：`d.pop(pair)` 把它从 dict 里拿出来，dict 不再引用它，就没人会改它了。
- 教训：**Python 的赋值只是起了个新名字，不会复制对象。**

### 4. 循环变量和索引

- ❌ `index = 0` 写在外层循环**里面** → 每轮都被重置。
- ❌ `index += 1` 写在**内层**循环 → 按 pair 递增，而不是按词。
- ✅ 用 `enumerate`：`for index, (word, count) in enumerate(zip(words, counts)):`
- ❌ `for i in range(...)` 结束后，`i` 会**保留最后一个值**，下面的 `while i < ...` 没重新 `i = 0` 就会从中间开始。
- ❌ `if` 命中后 `i += 2`，但忘了 `continue` / `else`，代码会继续往下执行。

### 5. tuple / list / hash

- tuple 能 hash，可以当 dict 的 key 和放进 set；**list 不能**。所以词要先转成 `tuple(...)` 再存。
- tuple 切片 `t[i:i+2]` 得到的还是 tuple，可以直接当 pair 的 key。
- 合并后的新 symbol 应该是 **bytes** `a + b`，❌ 不是 pair 本身 `(a, b)`。

### 6. 多进程注意

- macOS 上 `ProcessPoolExecutor` 会重新 import 主文件 → 入口一定要放在 `if __name__ == "__main__":` 下面，不然会无限启动进程。
- 进程启动本身有开销（几百毫秒），小文件不一定划算。

### 7. 用 cProfile 找性能瓶颈

```bash
python -m cProfile -s tottime profile_bpe.py | head -25   # 直接看
python -m cProfile -o out.prof profile_bpe.py             # 保存结果
```

```python
import pstats
p = pstats.Stats("out.prof")
p.sort_stats("cumtime").print_stats(15)
p.sort_stats("tottime").print_stats("train_bpe")   # 只看自己代码里的函数
```

| 列 | 意思 |
|---|---|
| `ncalls` | 调用次数 |
| `tottime` | 函数**自己**花的时间（不含子函数）→ 找真正干活的热点 |
| `cumtime` | 累计时间（含子函数）→ 看哪个大步骤慢 |

- 套路：先看 tottime 最大的函数，再看 **ncalls 异常大**的调用（比如 `len` 被调用了 1200 万次），回代码里找是哪个循环导致的。
- ⚠️ cProfile **只统计主进程**。子进程的时间在主进程里只显示为 `poll` / `acquire` / `wait`（"在等"）。想分析 pretokenize 本身，就在主进程里直接调用它。

---

## 三、学习方法

- 先写正确的慢版本，**保存输出**，优化后**对比结果是否完全一致**，而不是只看测试有没有过。
- 先用 PDF 的小例子（`low/lower/widest/newest`）手动走一遍，再跑大数据。
- 先 profile 再优化，不要凭感觉猜瓶颈。
