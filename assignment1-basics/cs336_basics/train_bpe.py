
import os
import regex as re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import heapq
from itertools import repeat
from collections import defaultdict

from .pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Number of worker processes (how many chunks are processed at the same time).
    num_of_process = kwargs.get("num_processes", min(4, os.cpu_count() or 1))
    # Target size of each chunk. Each worker holds only ONE chunk in memory at a time,
    # so smaller chunks -> lower peak memory. Decoupled from num_of_process on purpose.
    target_chunk_bytes = kwargs.get("chunk_bytes", 50 * 1024 * 1024)  # 50MB
    
    pretoken_counter = Counter() # dict[tuple[bytes], int]
    vocab: dict[int, bytes] = {token_id: bytes([token_id]) for token_id in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    merges: list[tuple[bytes, bytes]] = []

    with open(input_path, "rb") as f:
        file_size = os.path.getsize(input_path)
        # e.g. 2.2GB / 50MB -> ~45 chunks, still only num_of_process workers running at once
        num_chunks = max(num_of_process, file_size // target_chunk_bytes)
        boundaries = find_chunk_boundaries(f, num_chunks, [token.encode('utf-8') for token in special_tokens])
        ranges = list(zip(boundaries[:-1], boundaries[1:]))
        
        with ProcessPoolExecutor(max_workers=num_of_process) as executor:
            # chunk_counters = executor.map( pretokenize_chunk, [str(input_path)] * len(ranges),[start for start, _ in ranges],[end for _, end in ranges])
            chunk_counters = executor.map(pretokenize_chunk, repeat(str(input_path)), (start for start, _ in ranges),(end for _, end in ranges), repeat(tuple(special_tokens)) )
            pretoken_counter = Counter()
            for chunk_counter in chunk_counters:
                pretoken_counter.update(chunk_counter)

        # run BPE
        # example after pre tokenizing:  [('l', 'o', 'w): 5, ('l', 'o', 'w', 'e', 'r'): 2, ('n', 'e', 'w', 'e', 's', 't'): 3]
        # use heap, and mark pairs that needs updating

        # optimization #1
        # use a list to maintain a pretokens, so that list[i] represent 'l', 'o', 'w' and even lo merged, list [i] still represents 'lo', w'
        # change of list[i] (pretoken) can be inplace
        pretoken_words = list(pretoken_counter.keys())
        pretoken_words_counts = list(pretoken_counter.values())
        pair_to_words = defaultdict(set)

        pair_counter = Counter()
        for index, (pretoken, count) in enumerate(zip(pretoken_words, pretoken_words_counts)):
            for i in range(len(pretoken)-1):
                pair_counter[pretoken[i: i+2]] += count
                pair_to_words[pretoken[i: i+2]].add(index)

        max_heap = [
            (-count, ReversePair(pair), pair)
            for pair, count in pair_counter.items()
        ]
        # The final pair is used to retrieve the original pair without reversing anything.
        # max_heap = [(-count, pair) for pair, count in pair_counter.items()]
        heapq.heapify(max_heap)
        # pair_counter: {lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}

        # merge rounds, until 256 + special tokens + new merges reach target vocabulary size
        while len(vocab) < vocab_size and max_heap:
            count_in_negative, _, most_common_pair = heapq.heappop(max_heap) # get -9, ('e', 's')

            # pop out stale pair-count
            # eg. (-9, ("s", "t"))  ← 旧条目，之前没被选中，所以还在 (-3, ("s", "t"))  ← 刚推入的新条目
            # 现在得到的最大是9， ("s", "t")， 进行if条件里，发现pair_counter.get(most_common_pair)  已经被update到新的3了，不等于旧条目显示的9，所以continue。
            # 这个时候又正好把(-9, ("s", "t"))  pop出去了，   
            if pair_counter.get(most_common_pair) != -count_in_negative:
                continue
            
            # do merge
            merges.append(most_common_pair)
            vocab[len(vocab)] = most_common_pair[0] + most_common_pair[1]
            # pair_counter need update pairs that needs re-calculation, eg. previously widest 3, newest 6 -> es 9, st 9, after merging es, st should update to 0
            # pair counter updated -> max_heap need update too
            pretoken_words, pretoken_words_counts = apply_merge(pretoken_words, pretoken_words_counts, pair_to_words, most_common_pair, pair_counter, max_heap)

    return (vocab, merges)


def pretokenize_chunk(
    input_path: str,
    start: int,
    end: int,
    special_tokens: tuple[str, ...],
) -> Counter[tuple[bytes, ...]]:
    """Each worker opens and processes one independent file range."""
    counter = Counter()

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")

    # Longest first avoids ambiguity if one special token is a prefix of another.
    special_pattern = "|".join(
        re.escape(token)
        for token in sorted(special_tokens, key=len, reverse=True)
    )

    normal_text_parts = (
        re.split(special_pattern, chunk)
        if special_pattern
        else [chunk]
    )

    for text in normal_text_parts:
        for match in re.finditer(PAT, text):
            pretoken_bytes = match.group().encode("utf-8")
            symbols = tuple(bytes([byte]) for byte in pretoken_bytes)
            counter[symbols] += 1

    return counter

def apply_merge(
        pretoken_words: list[tuple[bytes, ...]],
        pretoken_words_counts: list[int],
        pair_to_words: dict[tuple, list[int]],
        most_common_pair: tuple[bytes, bytes], 
        pair_counter: dict[tuple[bytes, ...], int], 
        max_heap:  list[tuple[int, tuple[bytes]]]) -> dict[tuple[bytes], int]:
    # after merge: pretoken_counter: [('l', 'o', 'w'): 5, ('l', 'o', 'w', 'e', 'r'): 2, ('n', 'e', 'w', 'es', 't'): 3, ('n', 'e', 'w', 'es', 't'): 6]
    # pair_counter: {lo: 7, ow: 7, we: 8-6, er: 2, wi: 3, id: 3, de: 3, es: 9-9, st: 9-9, ne: 6, ew: 6, wes: 9, est: 9}
    # before merge: pretoken_words: [('l', 'o', 'w'), ('l', 'o', 'w', 'e', 'r'), ('n', 'e', 'w', 'e', 's', 't')]
    # before merge: pretoken_words_counts: [5, 2, 3]
    # before merge: pair_to_words: {lo: [1, 2], ow: [1,2], we: [2], er: [2]}
    
    updated_pairs = set()

    # optimization #1 no traverse to find which need update, use pair_to_words directly
    # and update pretoken inplace in pretoken_words, no merged_counter needed
    # how to handle pair_counter update? - same?
    indexes_need_update = pair_to_words.pop(most_common_pair)
    for index in indexes_need_update:
        word_need_update = pretoken_words[index]
        new_tuple_list = []

        # removing old pairs before merge
        for i in range(len(word_need_update)-1):
            old_pair = word_need_update[i: i+2]
            pair_to_words[old_pair].discard(index)
            pair_counter[old_pair] -= pretoken_words_counts[index] # we: 8-6, no need to update heap, stale check
            updated_pairs.add(old_pair)

        i = 0
        while i< len(word_need_update):
            if (i+1 < len(word_need_update) and word_need_update[i] == most_common_pair[0] and word_need_update[i+1] == most_common_pair[1]):
                new_tuple_list.append(most_common_pair[0] + most_common_pair[1])
                i+=2
                continue
            new_tuple_list.append(word_need_update[i])
            i+=1

        new_pretoken = tuple(new_tuple_list)
        pretoken_words[index] = new_pretoken

        i = 0
        # adding new pairs after merge
        for i in range(len(new_pretoken)-1):
            new_pair = new_pretoken[i: i+2]
            pair_to_words[new_pair].add(index)
            pair_counter[new_pair] += pretoken_words_counts[index]
            updated_pairs.add(new_pair)

    for pair in updated_pairs:
        count = pair_counter.get(pair, 0)
        if count <= 0: 
            pair_counter.pop(pair, None)
            continue
        heapq.heappush( max_heap,(-count, ReversePair(pair), pair))

    return (pretoken_words, pretoken_words_counts)

def reverse_bytes(b: bytes) -> bytes:
    return bytes(255 - x for x in b)

from functools import total_ordering


@total_ordering
class ReversePair:
    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other):
        return self.pair > other.pair

    def __eq__(self, other):
        return self.pair == other.pair