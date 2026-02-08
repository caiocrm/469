https://www.tibia.com/forum/?action=thread&threadid=4992945

1 - No intricate math. I don't think the idea in 2002 was to make the players use something like a Hill Cipher.
2 - I proved (in my view) proofs that, those are the tools needed to solve language 469:
2.1 - Paradox Tower books where deeply connected with the lang 469.
2.2 - The matrix being the only thing that changed and the whole Jekhr language being implemented 1 year later, proved that we need the matrix
2.3 - Not all sequences are words, some are opcodes like START, and boundary separators.
2.4 - That the Mathemagic rules we need to use are:
```markdown
1 + 1 = 1
1 + 1 = 13 
1 + 1 = 49
1 + 1 = 94

invert(13) = 31
49 = change_of_base(31, from=16, to=10)

so, considering 13 in base 10, 49 in base 10 and 94 in base 10:
1 + 1 = 13 change_of_base(invert(13), from=16, to=10)
1 + 1 = 49 ascii of 1
1 + 1 = 94 invert(94) = 49
```
2.5 - it's not clear where to use, but currently i'm using when a token starts with 11 and mod 4 = 1

Now to what i think what 469 generation decoding process looks like:
```markdown
Dtjfhg Jhfvzk (Book 1)
    dtjfhg
    jhfvzk
    bbliiug
    bkjjjjjjj
    xhvuo
    fffff
    zkkbk h
    lbhiovz
    klhi igbb

Ljkhbl Nilse (Book 2)
    ljkhbl nilse jfpce ojvco ld
    slcld ylddiv dnolsd dd sd
    sdcp cppcs cccpc cpsc
    awdp cpcw cfw ce
    cpvc ev vcemmev vrvf
    cp fd vmfpm xcv
```

- Book 1, 2 and the matrix must be S-tables (simple dictionaries lookups), because as i said, it is the more lore friendly approach, no intricate math.
- The modulo operation is needed to walk on those dictionaries
How?
```markdown
It's such an odd coincidence that:
book 1 have 9 rows (to select we must perform some number MOD 9)
book 2 have 6 rows (to select we must perform some number mod 6)
the matrix have dimensions 4x4 (to select we must perform some number MOD 4)
Thus, Language 469.
Which is the dumbest approach to this? Yes, get a know sequence and split in chunks of 4.
Now you have 4 numbers that could be used to retrieve data from the books

Book 2 is the final step of the pipeline, since it have 26 blocks and the Jekhr thing basically delivered to us the fact that the language 469 use 26 letters.
I coded some rules to test, and one combination yields EXACTLY "BEHOLDER" (using the old matrix). Run decode.py to see for yourself.
```


consider this book:
180036468895219911800651288952364672119118003576513534783046467972783967340579282758576512527570584521765219727830464876515956461141451988997

According to my theory, 003 is a control token that represents a boundary. Let's isolate the sequence beteween the two 003 in the book.
We now have:
6468895219911800651288952364672119118
Run in the decoding process:
```markdown
Output of decode.py:

Token: 8895 -> a=8 b=8 c=9 d=5 - 1 - 0 - ['ljkhbl', 'nilse', 'jfpce', 'ojvco', 'ld'] - 9 - 5
Token: 2199 -> a=2 b=1 c=9 d=9 - 1 - 0 - ['ljkhbl', 'nilse', 'jfpce', 'ojvco', 'ld'] - 9 - 9
Token: 1180 -> a=1 b=1 c=8 d=0 - 2 - 1 - ['slcld', 'ylddiv', 'dnolsd', 'dd', 'sd'] - 8 - 0
Token: 0651 -> a=0 b=6 c=5 d=1 - 4 - 3 - ['awdp', 'cpcw', 'cfw', 'ce'] - 5 - 1
Token: 2889 -> a=2 b=8 c=8 d=9 - 3 - 2 - ['sdcp', 'cppcs', 'cccpc', 'cpsc'] - 8 - 9
Token: 5236 -> a=5 b=2 c=3 d=6 - 1 - 0 - ['ljkhbl', 'nilse', 'jfpce', 'ojvco', 'ld'] - 3 - 6
Token: 4672 -> a=4 b=6 c=7 d=2 - 4 - 3 - ['awdp', 'cpcw', 'cfw', 'ce'] - 7 - 2

=== src0 ===
6468895219911800651288952364672119118
BEHOLDER
```
- I also found UND DANKH (almost UND DANKE), so searching for german words while bruteforcing parameters are required. And the book funny letters suggest the usage of german umlaut vowels (or just mabye a hint that we should use ascii).
- The decoding process is incomplete. Maybe it is one offset per eye. Maybe we are missing rules or the rules are dynamic.
- if you're gonna test your own sequences, ask yourself if you know the start of the sequence.
