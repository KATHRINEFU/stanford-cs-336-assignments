1. It took around 10min, 430mb, longest token is " accomplishment"
2. pretonization tooks most of the time (in the _valid dataset), saying "{method 'acquire' of '_thread.lock' objects}" 
3. skip the openWeb text for now because my laptop memory & cpu is rather small
'''
(.venv) (base) katefu@Kates-MacBook-Pro-4 assignment1-basics % /usr/bin/time -l python -m cs336_basics.train_bpe_tinystories
Saved vocab (10000) and merges (9743) to data/tinystories_bpe/
Longest token: b' accomplishment' (15 bytes)
Decoded: ' accomplishment'
      653.07 real      1600.96 user        44.60 sys
           430440448  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
             1343554  page reclaims
                1136  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
               11235  voluntary context switches
             4209635  involuntary context switches
         27233984611  instructions retired
         10868955625  cycles elapsed
           141378944  peak memory footprint
'''