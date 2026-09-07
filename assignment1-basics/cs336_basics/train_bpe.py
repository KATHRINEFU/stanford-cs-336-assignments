
import os
import regex as re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import heapq
from itertools import repeat

from .pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    num_of_process = 4
    
    pretoken_counter = Counter() # dict[tuple[bytes], int]
    vocab: dict[int, bytes] = {token_id: bytes([token_id]) for token_id in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    merges: list[tuple[bytes, bytes]] = []

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_of_process, [token.encode('utf-8') for token in special_tokens])
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
        pair_counter = Counter()
        for pretoken, count in pretoken_counter.items():
            for i in range(len(pretoken)-1):
                pair_counter[pretoken[i: i+2]] += count

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
            if len(merges) == 8:
                print("actual e-r:", pair_counter.get((b"e", b"r")))

                for item in max_heap:
                    if item[-1] == (b"e", b"r"):
                        print("heap e-r:", item)
            # pair_counter need update pairs that needs re-calculation, eg. previously widest 3, newest 6 -> es 9, st 9, after merging es, st should update to 0
            # pair counter updated -> max_heap need update too
            pretoken_counter = apply_merge(pretoken_counter, most_common_pair, pair_counter, max_heap)

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
        pretoken_counter: dict[tuple[bytes, ...], int], 
        most_common_pair: tuple[bytes, bytes], 
        pair_counter: dict[tuple[bytes, ...], int], 
        max_heap:  list[tuple[int, tuple[bytes]]]) -> dict[tuple[bytes], int]:
    # after merge: pretoken_counter: [('l', 'o', 'w'): 5, ('l', 'o', 'w', 'e', 'r'): 2, ('n', 'e', 'w', 'es', 't'): 3, ('n', 'e', 'w', 'es', 't'): 6]
    # pair_counter: {lo: 7, ow: 7, we: 8-6, er: 2, wi: 3, id: 3, de: 3, es: 9-9, st: 9-9, ne: 6, ew: 6, wes: 9, est: 9}
    merged_counter = Counter()
    updated_pairs = set()
    new_symbol = most_common_pair[0] + most_common_pair[1]

    for pretoken, pretoken_count in pretoken_counter.items():
        new_symbols = []
        i = 0
        merged_positions = []

        while i<len(pretoken):

            # merge first, then recalculate, otherwise XYXY will be XYX, Y
            if (i+1 < len(pretoken) and pretoken[i] == most_common_pair[0] and pretoken[i+1] == most_common_pair[1]):
                new_symbols.append(pretoken[i] + pretoken[i+1])
                merged_positions.append(i)
                i += 2
            else:
                new_symbols.append(pretoken[i])
                i+=1

        new_pretoken = tuple(new_symbols)
        merged_counter[new_pretoken] += pretoken_count

        if not merged_positions:
            continue

        # update affected: left and right adjacent
        for old_index in merged_positions:

            # left
            if old_index > 0:
                old_pair = (pretoken[old_index-1], pretoken[old_index])
                new_pair = (pretoken[old_index - 1], new_symbol)

                pair_counter[old_pair] -= pretoken_count # we: 8-6, no need to update heap, stale check
                pair_counter[new_pair] += pretoken_count

                updated_pairs.add(old_pair)
                updated_pairs.add(new_pair)

            # right
            if old_index<len(pretoken)-2:
                old_pair = (pretoken[old_index + 1], pretoken[old_index + 2])
                new_pair = (new_symbol, pretoken[old_index+2])

                pair_counter[old_pair] -= pretoken_count # st: 9-9, no need to update heap, stale check
                pair_counter[new_pair] += pretoken_count
                updated_pairs.add(old_pair)
                updated_pairs.add(new_pair)

            # remove merged pair
            pair_counter[most_common_pair] -= pretoken_count # es: 9-9
            updated_pairs.add(most_common_pair)

    for pair in updated_pairs:
        count = pair_counter.get(pair, 0)
        if count <= 0: 
            pair_counter.pop(pair, None)
            continue
        heapq.heappush( max_heap,(-count, ReversePair(pair), pair))

    
    return merged_counter

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