"""Itemized word count of a manuscript draft against the EHJ-CVI Original Article limits.

Usage (from the project root; Python 3.6 or later, standard library only):

    python3 code/tools/wordcount_ehjci.py <draft.md> <tables.md> <legends.md> <refs.json> [<refs.json> ...]

Counted towards the 5,000-word limit ("including references, figure legends and tables"):
  main text    '## Introduction' up to the first level-2 heading after '## Conclusions' (headings included)
  references   every reference cited anywhere in the draft, the main tables or the legends, one entry each,
               counted as the words of the verified entry plus one for its number
  legends      the main figure legends ('**Figure N. ...' blocks), without alt text and without the
               graphical abstract legend
  tables       the whole '## Main table(s)' section of the tables file (titles, cells and footnotes)
The abstract ('## Abstract' up to the next level-2 heading, Keywords line excluded) has its own 250-word limit.
Title page, keywords, declarations, alt text and the graphical abstract legend are reported but not counted.

The same functions are imported by submission/source/build_submission_md.py, so the count printed here is the
count the build enforces. Exit status: 0 within both limits, 1 above a limit, 2 on a structural error.
A reference file that does not exist is skipped with a note; a cited key without a verified entry is listed
and counted as zero words (the build refuses such a draft).
"""
from __future__ import print_function

import io
import json
import os
import re
import sys

LIMIT_TOTAL = 5000
LIMIT_ABSTRACT = 250
ALIASES = {"vukadinovic2025echoprime": "vukadinovic2026echoprime"}   # as in code/tools/number_refs.py


class StructureError(Exception):
    pass


def read(path):
    with io.open(path, encoding="utf-8") as fh:
        return fh.read()


def strip_comments(s):
    return re.sub(r"<!--.*?-->", "", s, flags=re.S)


def words(s):
    """Word counter of the submission build (unchanged since manuscript v04): citation keys, emphasis marks,
    heading marks, table rules and free-standing punctuation are not words."""
    s = re.sub(r"\[@[^\]]+\]", "", s)
    s = re.sub(r"[*#]", "", s)
    s = re.sub(r"-{3,}:?", " ", s.replace("|", " "))
    s = re.sub(r"(?<!\S)[():;,.]+(?!\S)", " ", s)
    return len(s.split())


def cited_keys(text):
    """Citation keys in order of first appearance."""
    order = []
    for grp in re.findall(r"\[(@[^\]]+)\]", text):
        for k in re.findall(r"@([\w]+)", grp):
            if k not in order:
                order.append(k)
    return order


def load_refs(paths):
    """Verified reference entries {key: vancouver}; missing files are returned separately."""
    refs, missing = {}, []
    for p in paths:
        if not os.path.exists(p):
            missing.append(p)
            continue
        d = json.loads(read(p))
        if isinstance(d, list):
            d = dict((r["key"], r["vancouver"]) for r in d)
        for k, v in d.items():
            refs[ALIASES.get(k, k)] = v
    return refs, missing


def split_manuscript(raw):
    """Cut a manuscript draft into its parts by level-2 headings.

    Returns a dict with: title, front (title-page block between the title and the abstract, Keywords and Word
    count lines removed), keywords, abstract (without heading), extra (sections between the abstract and the
    Introduction, normally empty), body (Introduction to Conclusions), decl (declarations up to References).
    """
    m = strip_comments(raw)
    lines = m.split("\n")
    title_idx = [i for i, l in enumerate(lines) if l.startswith("# ")]
    if not title_idx:
        raise StructureError("no level-1 title line ('# Title') in the draft")
    title = lines[title_idx[0]][2:].strip()
    heads = [(i, l[3:].strip()) for i, l in enumerate(lines) if l.startswith("## ")]
    names = [h[1].lower() for h in heads]

    def find(*cands):
        for c in cands:
            if c in names:
                return names.index(c)
        raise StructureError("required heading '## %s' not found; level-2 headings are: %s"
                             % (cands[0].capitalize(), [h[1] for h in heads]))

    i_abs, i_int = find("abstract"), find("introduction")
    i_con, i_ref = find("conclusions", "conclusion"), find("references")
    if not (i_abs < i_int < i_con < i_ref):
        raise StructureError("headings out of order: Abstract, Introduction, Conclusions, References expected")
    if i_ref != len(heads) - 1:
        raise StructureError("'## References' must be the last level-2 heading (found '%s' after it)"
                             % heads[i_ref + 1][1])

    def seg(a, b):
        return "\n".join(lines[heads[a][0]:heads[b][0]])

    front = "\n".join(lines[title_idx[0] + 1:heads[i_abs][0]])
    abstract = "\n".join(lines[heads[i_abs][0] + 1:heads[i_abs + 1][0]])
    kw_re = re.compile(r"^\*\*Keywords:?\*\*:?[ \t]*(.+)$", flags=re.M)
    keywords = None
    for block in (front, abstract):
        hit = kw_re.search(block)
        if hit and keywords is None:
            keywords = hit.group(1).strip()
    front = kw_re.sub("", front)
    abstract = kw_re.sub("", abstract).strip()
    front = re.sub(r"^\*\*Word count:?\*\*.*$", "", front, flags=re.M)
    front = re.sub(r"\n{3,}", "\n\n", front).strip()
    return {
        "title": title,
        "front": front,
        "keywords": keywords,
        "abstract": abstract,
        "extra": seg(i_abs + 1, i_int).strip() if i_int > i_abs + 1 else "",
        "body": seg(i_int, i_con + 1).strip(),
        "decl": seg(i_con + 1, i_ref).strip(),
        "body_sections": [(heads[k][1], words(seg(k, k + 1))) for k in range(i_int, i_con + 1)],
        "decl_sections": [heads[k][1] for k in range(i_con + 1, i_ref)],
    }


def main_tables(raw):
    """The '## Main table(s)' section of the tables file (without its heading)."""
    t = strip_comments(raw)
    hit = re.search(r"^## Main tables?[ \t]*$", t, flags=re.M)
    if not hit:
        raise StructureError("no '## Main table' heading in the tables file")
    rest = t[hit.end():]
    nxt = re.search(r"^## ", rest, flags=re.M)
    return (rest[:nxt.start()] if nxt else rest).strip()


def parse_legends(raw):
    """Main figure legends and the graphical abstract legend.

    Returns (figures, ga): figures is a list of (number, legend, alt) in file order, where alt is the text of
    an 'Alt text:' paragraph inside the block or None; ga is the graphical abstract legend ('' if absent).
    """
    t = strip_comments(raw)
    hit = re.search(r"^## Graphical abstract.*$", t, flags=re.M)
    main, ga = (t[:hit.start()], t[hit.end():]) if hit else (t, "")
    nxt = re.search(r"^#{1,2} ", ga, flags=re.M)
    if nxt:
        ga = ga[:nxt.start()]
    ga = re.sub(r"^\*\*Legend[^*]*\*\*[ \t]*", "", ga.strip())
    ga = re.sub(r"^Legend \([^)]*\):?[ \t]*", "", ga).strip()
    figures = []
    for block in re.split(r"\n\s*\n", main):
        block = block.strip()
        if not block or block.startswith("#"):
            continue
        new = re.match(r"\*\*Figure (\d+)[.:]", block)
        if new:
            figures.append([new.group(1), block, None])
        elif figures and re.match(r"(\*\*)?Alt text[.:]", block):
            figures[-1][2] = re.sub(r"^(\*\*)?Alt text[.:]?(\*\*)?[.:]?[ \t]*", "", block)
        elif figures:
            figures[-1][1] += "\n\n" + block
        else:
            raise StructureError("text before the first '**Figure N.' legend: %r" % block[:60])
    if not figures:
        raise StructureError("no '**Figure N.' legend found in the legends file")
    nums = [f[0] for f in figures]
    if nums != [str(i + 1) for i in range(len(nums))]:
        raise StructureError("main figure legends are not numbered 1..n in order: %s" % nums)
    return [tuple(f) for f in figures], ga


def count(draft_raw, tables_raw, legends_raw, ref_paths, alt_texts=None):
    """Itemized count; alt_texts optionally maps figure number to alt text (only to collect citations)."""
    ms = split_manuscript(draft_raw)
    tab = main_tables(tables_raw)
    figures, ga = parse_legends(legends_raw)
    legends = "\n\n".join(f[1] for f in figures)
    alts = [f[2] or (alt_texts or {}).get(f[0], "") for f in figures]
    refs, missing_files = load_refs(ref_paths)
    order = cited_keys("\n".join([ms["front"], ms["abstract"], ms["extra"], ms["body"], ms["decl"], tab, legends]
                                 + alts + [ga]))
    unverified = [k for k in order if k not in refs]
    c = {
        "main": words(ms["body"]),
        "references": sum(len(refs[k].split()) + 1 for k in order if k in refs),
        "legends": words(legends),
        "tables": words(tab),
        "abstract": words(ms["abstract"]),
        "n_references": len(order),
        "n_figures": len(figures),
        "n_main_tables": len(re.findall(r"^\*\*Table \d+", tab, flags=re.M)),
        "sections": ms["body_sections"],
        "not_counted": [("title", words(ms["title"])), ("title page", words(ms["front"])),
                        ("keywords", words(ms["keywords"] or "")), ("declarations", words(ms["decl"])),
                        ("alt text", sum(words(a) for a in alts)), ("graphical abstract legend", words(ga))],
        "extra": words(ms["extra"]),
        "unverified_keys": unverified,
        "missing_ref_files": missing_files,
        "abstract_citations": cited_keys(ms["abstract"]),
    }
    c["total"] = c["main"] + c["references"] + c["legends"] + c["tables"]
    c["over_total"] = c["total"] > LIMIT_TOTAL
    c["over_abstract"] = c["abstract"] > LIMIT_ABSTRACT
    return c


def one_line(c):
    """The line printed on the title page."""
    return ("{:,} (main text {:,}, references {:,}, figure legends {:,}, tables {:,}; abstract {:,} words, "
            "not included)").format(c["total"], c["main"], c["references"], c["legends"], c["tables"],
                                    c["abstract"])


def report(c):
    out = ["EHJ-CVI word count (limit {:,}; abstract limit {:,})".format(LIMIT_TOTAL, LIMIT_ABSTRACT)]
    out.append("  main text (Introduction to Conclusions)   {:>6,}".format(c["main"]))
    for name, n in c["sections"]:
        out.append("      {:<38} {:>6,}".format(name, n))
    out.append("  references ({} entries)                   {:>6,}".format(c["n_references"], c["references"]))
    out.append("  figure legends ({} figures, no alt text)   {:>6,}".format(c["n_figures"], c["legends"]))
    out.append("  main tables ({} table{})                    {:>6,}".format(
        c["n_main_tables"], "" if c["n_main_tables"] == 1 else "s", c["tables"]))
    out.append("  TOTAL                                     {:>6,}   {}".format(
        c["total"], "OVER the limit by {:,}".format(c["total"] - LIMIT_TOTAL) if c["over_total"]
        else "{:,} below the limit".format(LIMIT_TOTAL - c["total"])))
    out.append("  abstract                                  {:>6,}   {}".format(
        c["abstract"], "OVER the limit by {:,}".format(c["abstract"] - LIMIT_ABSTRACT) if c["over_abstract"]
        else "{:,} below the limit".format(LIMIT_ABSTRACT - c["abstract"])))
    out.append("  not counted: " + "; ".join("%s %d" % (k, v) for k, v in c["not_counted"]))
    if c["extra"]:
        out.append("  NOTE: {:,} words sit between the abstract and the Introduction and are not counted"
                   .format(c["extra"]))
    if c["abstract_citations"]:
        out.append("  NOTE: the abstract cites %s; abstracts normally carry no citations" % c["abstract_citations"])
    for p in c["missing_ref_files"]:
        out.append("  NOTE: reference file not found, skipped: %s" % p)
    if c["unverified_keys"]:
        out.append("  WARNING: cited keys without a verified entry (counted as 0 words; the build will refuse): %s"
                   % c["unverified_keys"])
    return "\n".join(out)


def main(argv):
    if len(argv) < 4:
        sys.stderr.write(__doc__)
        return 2
    try:
        c = count(read(argv[0]), read(argv[1]), read(argv[2]), argv[3:])
    except StructureError as e:
        sys.stderr.write("wordcount_ehjci: %s\n" % e)
        return 2
    print(report(c))
    return 1 if (c["over_total"] or c["over_abstract"]) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
