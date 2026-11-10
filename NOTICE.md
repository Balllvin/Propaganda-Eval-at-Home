# Credit and third party notices

The Censorship and Propaganda Eval (prompts, required facts, judge prompt, reference grader and scorer) is by
Nous Research and is published at https://huggingface.co/datasets/NousResearch/Propaganda_Eval under the MIT licence.
This repository does not copy that data. It downloads it at run time at a pinned revision and checks its sha256.

The prompt fill, verdict check, retry rule and scoring rule in `propaganda_eval/judge.py` and `propaganda_eval/score.py`
follow Nous's `judge/grade.py` and `judge/score.py`, used under this licence:

```
MIT License

Copyright (c) 2026 Nous Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Each item's prompt and facts derive from a cited source under that source's own terms; the source URL is in the data.
