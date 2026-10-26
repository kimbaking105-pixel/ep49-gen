#!/usr/bin/env python3
"""
Объединяет шарды генератора в один файл и удаляет дубликаты позиций между шардами и процессами.

  python merge.py --src data --dst train.tsv

Формат выхода совпадает со строками шарда:
  fen \t score_stm_cp \t result_stm \t ply \t depth \t game_id
Шарды без .idx (не дописанные) пропускаются.
"""
import argparse, glob, os, sys
from collections import Counter

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()

    bases = sorted(os.path.splitext(p)[0] for p in glob.glob(os.path.join(a.src, "shard_*.tsv")))
    use, hs = [], []
    for b in bases:
        if not os.path.exists(b + ".idx"):
            print("пропуск (нет .idx):", os.path.basename(b))
            continue
        h = np.fromfile(b + ".idx", dtype="<u8")
        use.append(b)
        hs.append(h)
    if not use:
        sys.exit("нет готовых шардов")
    allh = np.concatenate(hs)
    _, first = np.unique(allh, return_index=True)
    keep = np.zeros(len(allh), dtype=bool)
    keep[first] = True

    res, depth_c = Counter(), Counter()
    n_out, n_lines, abs_sum, ply_sum = 0, 0, 0.0, 0
    off = 0
    with open(a.dst, "w", encoding="utf-8", newline="\n") as out:
        for b, h in zip(use, hs):
            cnt = len(h)
            with open(b + ".tsv", encoding="utf-8") as f:
                i = 0
                for line in f:
                    if i >= cnt:
                        sys.exit(f"{b}: строк больше, чем в .idx; шард повреждён")
                    if keep[off + i]:
                        out.write(line)
                        p = line.rstrip("\n").split("\t")
                        abs_sum += abs(int(p[1]))
                        res[float(p[2])] += 1
                        ply_sum += int(p[3])
                        depth_c[int(p[4])] += 1
                        n_out += 1
                    i += 1
                n_lines += i
                if i != cnt:
                    sys.exit(f"{b}: строк {i}, в .idx {cnt}; шард повреждён")
            off += cnt

    total = len(allh)
    print(f"шардов: {len(use)}")
    print(f"строк всего: {total}, уникальных по хешу: {n_out}, дубликатов удалено: {total - n_out} "
          f"({(total - n_out) / max(1, total) * 100:.1f}%)")
    if n_out:
        print(f"исход со стороны хода: выигрыш {res[1.0]/n_out*100:.1f}%, ничья {res[0.5]/n_out*100:.1f}%, "
              f"проигрыш {res[0.0]/n_out*100:.1f}%")
        print(f"средний |score| {abs_sum/n_out:.0f} сП, средний полуход {ply_sum/n_out:.0f}")
        print("глубина поиска (топ-5):", depth_c.most_common(5))
    print("записано:", a.dst)


if __name__ == "__main__":
    main()
