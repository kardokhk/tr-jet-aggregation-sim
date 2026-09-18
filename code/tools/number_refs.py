"""Convert [@key] citations in a markdown draft into numbered Vancouver citations.

Numbers follow order of first appearance outside HTML comments; ranges of three or more are
compressed (e.g. [3-5]). Reference text comes from verified-reference JSON files only.
Usage: python3 code/tools/number_refs.py draft.md out.md refs1.json [refs2.json ...]
A JSON file is either {key: entry} or a list of {"key":..., "vancouver":...}. Python 3.6 compatible.
"""
import json
import re
import sys

ALIASES = {"vukadinovic2025echoprime": "vukadinovic2026echoprime"}


def load_refs(paths):
    refs = {}
    for p in paths:
        d = json.load(open(p))
        if isinstance(d, list):
            d = {r["key"]: r["vancouver"] for r in d}
        for k, v in d.items():
            refs[ALIASES.get(k, k)] = v
    return refs


def main(src, dst, ref_paths):
    text = open(src).read()
    body = text.split("\n## References")[0]
    visible = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    order = []
    for grp in re.findall(r"\[(@[^\]]+)\]", visible):
        for k in re.findall(r"@([\w]+)", grp):
            if k not in order:
                order.append(k)
    refs = load_refs(ref_paths)
    missing = [k for k in order if k not in refs]
    if missing:
        sys.exit("unverified or missing keys: %s" % missing)
    num = {k: i + 1 for i, k in enumerate(order)}

    def conv(m):
        keys = re.findall(r"@([\w]+)", m.group(1))
        ns = sorted(set(num[k] for k in keys))
        out, i = [], 0
        while i < len(ns):
            j = i
            while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
                j += 1
            out.append("%d-%d" % (ns[i], ns[j]) if j - i >= 2 else ",".join(str(x) for x in ns[i:j + 1]))
            i = j + 1
        return "[" + ",".join(out) + "]"

    new_body = re.sub(r"\[(@[^\]]+)\]", conv, body)
    reflist = "\n\n".join("%d. %s" % (num[k], refs[k]) for k in order)
    open(dst, "w").write(new_body + "\n## References\n\n" + reflist + "\n")
    print("%d references numbered -> %s" % (len(order), dst))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
