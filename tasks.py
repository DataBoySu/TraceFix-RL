"""
tasks.py — Static Task Registry
================================
This is a "dumb" registry. Tasks are hardcoded dicts representing
curated buggy programs generated offline via MutationEngine.

Exported symbols:
  TASKS_BY_DIFFICULTY   Dict[str, List[Dict]] — tasks grouped by tier
  ALL_TASKS             List[Dict]             — flat list for random sampling

Run mutation_engine.py + dataset_generator.py locally (offline) to
generate new candidates, curate the best ones, and add them here.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List


# ---------------------------------------------------------------------------
# Test helpers (module-level; accept namespace dict, raise AssertionError)
# ---------------------------------------------------------------------------

# ── sum_even_numbers ────────────────────────────────────────────────────────

def _tse_1(ns):
    res = ns["sum_even_numbers"]([1, 2, 3, 4])
    assert res == 6, f"Test failed: input=[1, 2, 3, 4], expected=6, got={res}"
def _tse_2(ns):
    res = ns["sum_even_numbers"]([])
    assert res == 0, f"Test failed: input=[], expected=0, got={res}"
def _tse_3(ns):
    res = ns["sum_even_numbers"]([1, 3, 5])
    assert res == 0, f"Test failed: input=[1, 3, 5], expected=0, got={res}"
def _tse_4(ns):
    res = ns["sum_even_numbers"]([2, 2, 2])
    assert res == 6, f"Test failed: input=[2, 2, 2], expected=6, got={res}"

# ── reverse_string ──────────────────────────────────────────────────────────

def _trs_1(ns):
    res = ns["reverse_string"]("abc")
    assert res == "cba", f"Test failed: input='abc', expected='cba', got={res!r}"
def _trs_2(ns):
    res = ns["reverse_string"]("")
    assert res == "", f"Test failed: input='', expected='', got={res!r}"
def _trs_3(ns):
    res = ns["reverse_string"]("a")
    assert res == "a", f"Test failed: input='a', expected='a', got={res!r}"
def _trs_4(ns):
    res = ns["reverse_string"]("abcd")
    assert res == "dcba", f"Test failed: input='abcd', expected='dcba', got={res!r}"

# ── binary_search ───────────────────────────────────────────────────────────

def _tbs_1(ns):
    res = ns["binary_search"]([1, 2, 3, 4, 5], 3)
    assert res == 2, f"Test failed: input=([1, 2, 3, 4, 5], 3), expected=2, got={res}"
def _tbs_2(ns):
    res = ns["binary_search"]([1, 2, 3, 4, 5], 6)
    assert res == -1, f"Test failed: input=([1, 2, 3, 4, 5], 6), expected=-1, got={res}"
def _tbs_3(ns):
    res = ns["binary_search"]([], 1)
    assert res == -1, f"Test failed: input=([], 1), expected=-1, got={res}"
def _tbs_4(ns):
    res = ns["binary_search"]([7], 7)
    assert res == 0, f"Test failed: input=([7], 7), expected=0, got={res}"

# ── flatten ─────────────────────────────────────────────────────────────────

def _tfl_1(ns):
    res = ns["flatten"]([1, [2, 3]])
    assert res == [1, 2, 3], f"Test failed: input=[1, [2, 3]], expected=[1, 2, 3], got={res}"
def _tfl_2(ns):
    res = ns["flatten"]([])
    assert res == [], f"Test failed: input=[], expected=[], got={res}"
def _tfl_3(ns):
    res = ns["flatten"]([1, [2, [3]]])
    assert res == [1, 2, 3], f"Test failed: input=[1, [2, [3]]], expected=[1, 2, 3], got={res}"
def _tfl_4(ns):
    res = ns["flatten"]([[1], [2, 3], [4]])
    assert res == [1, 2, 3, 4], f"Test failed: input=[[1], [2, 3], [4]], expected=[1, 2, 3, 4], got={res}"

# ── word_count ──────────────────────────────────────────────────────────────

def _twc_1(ns):
    res = ns["word_count"]("hello world hello")
    assert res == {"hello": 2, "world": 1}, f"Test failed: input='hello world hello', expected={{'hello': 2, 'world': 1}}, got={res}"
def _twc_2(ns):
    res = ns["word_count"]("Hi, hi!")
    assert res == {"hi": 2}, f"Test failed: input='Hi, hi!', expected={{'hi': 2}}, got={res}"
def _twc_3(ns):
    res = ns["word_count"]("")
    assert res == {}, f"Test failed: input='', expected={{}}, got={res}"
def _twc_4(ns):
    res = ns["word_count"]("Hello HELLO hello")
    assert res == {"hello": 3}, f"Test failed: input='Hello HELLO hello', expected={{'hello': 3}}, got={res}"

# ── lru_cache ───────────────────────────────────────────────────────────────

def _tlru_1(ns):
    C = ns["LRUCache"]
    c = C(2); c.put(1, 1); c.put(2, 2)
    res = c.get(1)
    assert res == 1, f"Test failed: Capacity 2. Added (1,1), then (2,2). Expected get(1) to be 1, got {res}"

def _tlru_2(ns):
    C = ns["LRUCache"]
    c = C(1); c.put(1, 1); c.put(2, 2)
    res = c.get(1)
    assert res == -1, f"Test failed: Capacity 1. Added (1,1), then (2,2). Expected key 1 to be evicted (return -1), got {res}"

def _tlru_3(ns):
    C = ns["LRUCache"]
    c = C(2); c.put(1, 1); c.put(2, 2); c.get(1); c.put(3, 3)
    res = c.get(2)
    assert res == -1, f"Test failed: Capacity 2. Added (1,1), then (2,2), got(1), added (3,3). Expected key 2 to be evicted (return -1) since 1 was promoted during get(1), got {res}. Did you promote key 1 during get()?"

# ── valid_parentheses ────────────────────────────────────────────────────────

def _tvp_1(ns):
    res = ns["is_valid"]("()")
    assert res == True, f"Test failed: input='()', expected=True, got={res}"
def _tvp_2(ns):
    res = ns["is_valid"]("(]")
    assert res == False, f"Test failed: input='(]', expected=False, got={res}"
def _tvp_3(ns):
    res = ns["is_valid"]("([{}])")
    assert res == True, f"Test failed: input='([{{}}])', expected=True, got={res}"
def _tvp_4(ns):
    res = ns["is_valid"]("")
    assert res == True, f"Test failed: input='', expected=True, got={res}"

# ── merge_intervals ──────────────────────────────────────────────────────────

def _tmi_1(ns):
    res = ns["merge_intervals"]([[1, 3], [2, 6]])
    assert res == [[1, 6]], f"Test failed: input=[[1, 3], [2, 6]], expected=[[1, 6]], got={res}"
def _tmi_2(ns):
    res = ns["merge_intervals"]([[1, 4], [4, 5]])
    assert res == [[1, 5]], f"Test failed: input=[[1, 4], [4, 5]], expected=[[1, 5]], got={res}"
def _tmi_3(ns):
    res = ns["merge_intervals"]([[1, 2], [3, 4]])
    assert res == [[1, 2], [3, 4]], f"Test failed: input=[[1, 2], [3, 4]], expected=[[1, 2], [3, 4]], got={res}"


# ---------------------------------------------------------------------------
# Static task registry
# ---------------------------------------------------------------------------

def _t(name, description, code, solution, tests, difficulty, bug_type):
    return dict(
        name=name, description=description,
        code=code, solution=solution,
        tests=tests, difficulty=difficulty, bug_type=bug_type,
    )


# ── EASY ──────────────────────────────────────────────────────────────────

TASK_SUM_EVEN_WRONG_OP = _t(
    name="sum_even_wrong_condition",
    description="Debug the sum_even_numbers function so it passes all tests.",
    difficulty="easy",
    bug_type="wrong_operator",
    code=[
        "def sum_even_numbers(nums):",
        "    total = 0",
        "    for n in nums:",
        "        if n % 2 != 0:",
        "            total += n",
        "    return total",
    ],
    solution=[
        "def sum_even_numbers(nums):",
        "    total = 0",
        "    for n in nums:",
        "        if n % 2 == 0:",
        "            total += n",
        "    return total",
    ],
    tests=[_tse_1, _tse_2, _tse_3, _tse_4],
)

TASK_SUM_EVEN_MISSING_INIT = _t(
    name="sum_even_missing_accumulator",
    description="Debug the sum_even_numbers function so it passes all tests.",
    difficulty="easy",
    bug_type="wrong_operator",
    code=[
        "def sum_even_numbers(nums):",
        "    total = 0",
        "    for n in nums:",
        "        if n % 2 == 0:",
        "            total -= n",
        "    return total",
    ],
    solution=[
        "def sum_even_numbers(nums):",
        "    total = 0",
        "    for n in nums:",
        "        if n % 2 == 0:",
        "            total += n",
        "    return total",
    ],
    tests=[_tse_1, _tse_2, _tse_3, _tse_4],
)

TASK_REVERSE_WRONG_STEP = _t(
    name="reverse_string_wrong_step",
    description="Debug the reverse_string function so it passes all tests.",
    difficulty="easy",
    bug_type="off_by_one",
    code=[
        "def reverse_string(s):",
        "    return s[::-2]",
    ],
    solution=[
        "def reverse_string(s):",
        "    return s[::-1]",
    ],
    tests=[_trs_1, _trs_2, _trs_3, _trs_4],
)

TASK_REVERSE_NO_REVERSE = _t(
    name="reverse_string_returns_original",
    description="Debug the reverse_string function so it passes all tests.",
    difficulty="easy",
    bug_type="wrong_operator",
    code=[
        "def reverse_string(s):",
        "    return s[::1]",
    ],
    solution=[
        "def reverse_string(s):",
        "    return s[::-1]",
    ],
    tests=[_trs_1, _trs_2, _trs_3, _trs_4],
)


# ── MEDIUM ────────────────────────────────────────────────────────────────

TASK_BS_OFF_BY_ONE = _t(
    name="binary_search_off_by_one",
    description="Debug the binary_search function so it passes all tests.",
    difficulty="medium",
    bug_type="off_by_one",
    code=[
        "def binary_search(arr, target):",
        "    left, right = 0, len(arr)",
        "    while left <= right:",
        "        mid = (left + right) // 2",
        "        if arr[mid] == target:",
        "            return mid",
        "        elif arr[mid] < target:",
        "            left = mid + 1",
        "        else:",
        "            right = mid - 1",
        "    return -1",
    ],
    solution=[
        "def binary_search(arr, target):",
        "    left, right = 0, len(arr) - 1",
        "    while left <= right:",
        "        mid = (left + right) // 2",
        "        if arr[mid] == target:",
        "            return mid",
        "        elif arr[mid] < target:",
        "            left = mid + 1",
        "        else:",
        "            right = mid - 1",
        "    return -1",
    ],
    tests=[_tbs_1, _tbs_2, _tbs_3, _tbs_4],
)

TASK_BS_WRONG_MID = _t(
    name="binary_search_wrong_mid",
    description="Debug the binary_search function so it passes all tests.",
    difficulty="medium",
    bug_type="wrong_operator",
    code=[
        "def binary_search(arr, target):",
        "    left, right = 0, len(arr) - 1",
        "    while left <= right:",
        "        mid = left + right",
        "        if mid >= len(arr):",
        "            return -1",
        "        if arr[mid] == target:",
        "            return mid",
        "        elif arr[mid] < target:",
        "            left = mid + 1",
        "        else:",
        "            right = mid - 1",
        "    return -1",
    ],
    solution=[
        "def binary_search(arr, target):",
        "    left, right = 0, len(arr) - 1",
        "    while left <= right:",
        "        mid = (left + right) // 2",
        "        if arr[mid] == target:",
        "            return mid",
        "        elif arr[mid] < target:",
        "            left = mid + 1",
        "        else:",
        "            right = mid - 1",
        "    return -1",
    ],
    tests=[_tbs_1, _tbs_2, _tbs_3, _tbs_4],
)

TASK_FLATTEN_APPEND = _t(
    name="flatten_missing_recursion",
    description="Debug the flatten function so it passes all tests.",
    difficulty="medium",
    bug_type="wrong_function_call",
    code=[
        "def flatten(lst):",
        "    result = []",
        "    for item in lst:",
        "        if isinstance(item, list):",
        "            result.append(item)",
        "        else:",
        "            result.append(item)",
        "    return result",
    ],
    solution=[
        "def flatten(lst):",
        "    result = []",
        "    for item in lst:",
        "        if isinstance(item, list):",
        "            result.extend(flatten(item))",
        "        else:",
        "            result.append(item)",
        "    return result",
    ],
    tests=[_tfl_1, _tfl_2, _tfl_3, _tfl_4],
)

TASK_FLATTEN_LOGIC_INVERT = _t(
    name="flatten_inverted_branch",
    description="Debug the flatten function so it passes all tests.",
    difficulty="medium",
    bug_type="logic_inversion",
    code=[
        "def flatten(lst):",
        "    result = []",
        "    for item in lst:",
        "        if not isinstance(item, list):",
        "            result.extend(flatten(item))",
        "        else:",
        "            result.append(item)",
        "    return result",
    ],
    solution=[
        "def flatten(lst):",
        "    result = []",
        "    for item in lst:",
        "        if isinstance(item, list):",
        "            result.extend(flatten(item))",
        "        else:",
        "            result.append(item)",
        "    return result",
    ],
    tests=[_tfl_1, _tfl_2, _tfl_3, _tfl_4],
)

TASK_WC_NO_LOWER = _t(
    name="word_count_no_lower",
    description="Debug the word_count function so it passes all tests.",
    difficulty="medium",
    bug_type="missing_return",
    code=[
        "import string",
        "def word_count(text):",
        "    for p in string.punctuation:",
        "        text = text.replace(p, '')",
        "    words = text.split()",
        "    counts = {}",
        "    for w in words:",
        "        counts[w] = counts.get(w, 0) + 1",
        "    return counts",
    ],
    solution=[
        "import string",
        "def word_count(text):",
        "    text = text.lower()",
        "    for p in string.punctuation:",
        "        text = text.replace(p, '')",
        "    words = text.split()",
        "    counts = {}",
        "    for w in words:",
        "        counts[w] = counts.get(w, 0) + 1",
        "    return counts",
    ],
    tests=[_twc_1, _twc_2, _twc_3, _twc_4],
)

TASK_WC_NO_PUNCT = _t(
    name="word_count_no_punct_strip",
    description="Debug the word_count function so it passes all tests.",
    difficulty="medium",
    bug_type="missing_return",
    code=[
        "def word_count(text):",
        "    text = text.lower()",
        "    words = text.split()",
        "    counts = {}",
        "    for w in words:",
        "        counts[w] = counts.get(w, 0) + 1",
        "    return counts",
    ],
    solution=[
        "import string",
        "def word_count(text):",
        "    text = text.lower()",
        "    for p in string.punctuation:",
        "        text = text.replace(p, '')",
        "    words = text.split()",
        "    counts = {}",
        "    for w in words:",
        "        counts[w] = counts.get(w, 0) + 1",
        "    return counts",
    ],
    tests=[_twc_1, _twc_2, _twc_3, _twc_4],
)


# ── HARD ──────────────────────────────────────────────────────────────────

TASK_LRU_WRONG_EVICT = _t(
    name="lru_cache_wrong_eviction",
    description="Debug the LRUCache function so it passes all tests.",
    difficulty="hard",
    bug_type="off_by_one",
    code=[
        "class LRUCache:",
        "    def __init__(self, capacity):",
        "        self.capacity = capacity",
        "        self.cache = []",
        "    def get(self, key):",
        "        for i, (k, v) in enumerate(self.cache):",
        "            if k == key:",
        "                self.cache.append(self.cache.pop(i))",
        "                return v",
        "        return -1",
        "    def put(self, key, value):",
        "        for i, (k, _) in enumerate(self.cache):",
        "            if k == key:",
        "                self.cache.pop(i)",
        "                break",
        "        if len(self.cache) >= self.capacity:",
        "            self.cache.pop(0)",
        "        self.cache.append((key, value))",
    ],
    solution=[
        "class LRUCache:",
        "    def __init__(self, capacity):",
        "        self.capacity = capacity",
        "        self.cache = []",
        "    def get(self, key):",
        "        for i, (k, v) in enumerate(self.cache):",
        "            if k == key:",
        "                self.cache.append(self.cache.pop(i))",
        "                return v",
        "        return -1",
        "    def put(self, key, value):",
        "        for i, (k, _) in enumerate(self.cache):",
        "            if k == key:",
        "                self.cache.pop(i)",
        "                break",
        "        if len(self.cache) >= self.capacity:",
        "            self.cache.pop(0)",
        "        self.cache.append((key, value))",
    ],
    tests=[_tlru_1, _tlru_2, _tlru_3],
)

TASK_LRU_NO_PROMOTE = _t(
    name="lru_cache_no_promotion",
    description="Debug the LRUCache function so it passes all tests.",
    difficulty="hard",
    bug_type="missing_return",
    code=[
        "class LRUCache:",
        "    def __init__(self, capacity):",
        "        self.capacity = capacity",
        "        self.cache = []",
        "    def get(self, key):",
        "        for i, (k, v) in enumerate(self.cache):",
        "            if k == key:",
        "                return v",
        "        return -1",
        "    def put(self, key, value):",
        "        for i, (k, _) in enumerate(self.cache):",
        "            if k == key:",
        "                self.cache.pop(i)",
        "                break",
        "        if len(self.cache) >= self.capacity:",
        "            self.cache.pop(0)",
        "        self.cache.append((key, value))",
    ],
    solution=[
        "class LRUCache:",
        "    def __init__(self, capacity):",
        "        self.capacity = capacity",
        "        self.cache = []",
        "    def get(self, key):",
        "        for i, (k, v) in enumerate(self.cache):",
        "            if k == key:",
        "                self.cache.append(self.cache.pop(i))",
        "                return v",
        "        return -1",
        "    def put(self, key, value):",
        "        for i, (k, _) in enumerate(self.cache):",
        "            if k == key:",
        "                self.cache.pop(i)",
        "                break",
        "        if len(self.cache) >= self.capacity:",
        "            self.cache.pop(0)",
        "        self.cache.append((key, value))",
    ],
    tests=[_tlru_1, _tlru_2, _tlru_3],
)

TASK_VP_WRONG_MAPPING = _t(
    name="valid_parentheses_wrong_mapping",
    description="Debug the is_valid function so it passes all tests.",
    difficulty="hard",
    bug_type="wrong_operator",
    code=[
        "def is_valid(s):",
        "    stack = []",
        "    mapping = {')': '[', ']': '{', '}': '('}",
        "    for c in s:",
        "        if c in mapping.values():",
        "            stack.append(c)",
        "        elif c in mapping:",
        "            if not stack or stack.pop() != mapping[c]:",
        "                return False",
        "    return len(stack) == 0",
    ],
    solution=[
        "def is_valid(s):",
        "    stack = []",
        "    mapping = {')': '(', ']': '[', '}': '{'}",
        "    for c in s:",
        "        if c in mapping.values():",
        "            stack.append(c)",
        "        elif c in mapping:",
        "            if not stack or stack.pop() != mapping[c]:",
        "                return False",
        "    return len(stack) == 0",
    ],
    tests=[_tvp_1, _tvp_2, _tvp_3, _tvp_4],
)

TASK_VP_MISSING_EMPTY_CHECK = _t(
    name="valid_parentheses_no_empty_check",
    description="Debug the is_valid function so it passes all tests.",
    difficulty="hard",
    bug_type="logic_inversion",
    code=[
        "def is_valid(s):",
        "    stack = []",
        "    mapping = {')': '(', ']': '[', '}': '{'}",
        "    for c in s:",
        "        if c in mapping.values():",
        "            stack.append(c)",
        "        elif c in mapping:",
        "            if stack.pop() != mapping[c]:",
        "                return False",
        "    return len(stack) == 0",
    ],
    solution=[
        "def is_valid(s):",
        "    stack = []",
        "    mapping = {')': '(', ']': '[', '}': '{'}",
        "    for c in s:",
        "        if c in mapping.values():",
        "            stack.append(c)",
        "        elif c in mapping:",
        "            if not stack or stack.pop() != mapping[c]:",
        "                return False",
        "    return len(stack) == 0",
    ],
    tests=[_tvp_1, _tvp_2, _tvp_3, _tvp_4],
)

TASK_MI_STRICT_OVERLAP = _t(
    name="merge_intervals_strict_overlap",
    description="Debug the merge_intervals function so it passes all tests.",
    difficulty="hard",
    bug_type="wrong_operator",
    code=[
        "def merge_intervals(intervals):",
        "    intervals.sort()",
        "    merged = []",
        "    for interval in intervals:",
        "        if not merged or merged[-1][1] < interval[0]:",
        "            merged.append(list(interval))",
        "        else:",
        "            merged[-1][1] = max(merged[-1][1], interval[1])",
        "    return merged",
    ],
    solution=[
        "def merge_intervals(intervals):",
        "    intervals.sort()",
        "    merged = []",
        "    for interval in intervals:",
        "        if not merged or merged[-1][1] < interval[0]:",
        "            merged.append(list(interval))",
        "        else:",
        "            merged[-1][1] = max(merged[-1][1], interval[1])",
        "    return merged",
    ],
    tests=[_tmi_1, _tmi_2, _tmi_3],
)

TASK_MI_NO_SORT = _t(
    name="merge_intervals_missing_sort",
    description="Debug the merge_intervals function so it passes all tests.",
    difficulty="hard",
    bug_type="missing_return",
    code=[
        "def merge_intervals(intervals):",
        "    merged = []",
        "    for interval in intervals:",
        "        if not merged or merged[-1][1] < interval[0]:",
        "            merged.append(list(interval))",
        "        else:",
        "            merged[-1][1] = max(merged[-1][1], interval[1])",
        "    return merged",
    ],
    solution=[
        "def merge_intervals(intervals):",
        "    intervals.sort()",
        "    merged = []",
        "    for interval in intervals:",
        "        if not merged or merged[-1][1] < interval[0]:",
        "            merged.append(list(interval))",
        "        else:",
        "            merged[-1][1] = max(merged[-1][1], interval[1])",
        "    return merged",
    ],
    tests=[_tmi_1, _tmi_2, _tmi_3],
)


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

TASKS_BY_DIFFICULTY: Dict[str, List[Dict]] = {
    "easy": [
        TASK_SUM_EVEN_WRONG_OP,
        TASK_SUM_EVEN_MISSING_INIT,
        TASK_REVERSE_WRONG_STEP,
        TASK_REVERSE_NO_REVERSE,
    ],
    "medium": [
        TASK_BS_OFF_BY_ONE,
        TASK_BS_WRONG_MID,
        TASK_FLATTEN_APPEND,
        TASK_FLATTEN_LOGIC_INVERT,
        TASK_WC_NO_LOWER,
        TASK_WC_NO_PUNCT,
    ],
    "hard": [
        TASK_LRU_WRONG_EVICT,
        TASK_LRU_NO_PROMOTE,
        TASK_VP_WRONG_MAPPING,
        TASK_VP_MISSING_EMPTY_CHECK,
        TASK_MI_STRICT_OVERLAP,
        TASK_MI_NO_SORT,
    ],
}

# Flat list — used for random sampling when training_step is not set
ALL_TASKS: List[Dict] = [
    t for bucket in TASKS_BY_DIFFICULTY.values() for t in bucket
]
