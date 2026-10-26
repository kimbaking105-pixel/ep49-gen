#!/usr/bin/env python3
"""
Генератор позиций для дообучения ep49: самоигра движка с лимитом узлов на ход.

ЗАПУСК (4 процесса генерации, 2 ядра оставляем системе), каждый процесс в своём окне:
  python gen.py --engine eng_b15.exe --net net_comp_ft_ep49.nnue --out data --workers 4 --worker 0
  python gen.py ... --worker 1
  python gen.py ... --worker 2
  python gen.py ... --worker 3

ПЕРЕД ЗАПУСКОМ:
  1. Собрать движок без трассировки UCI: -DUCI_TRACE_ENABLED=0 (иначе пишется uci_log.txt).
  2. pip install numpy chess
  3. Остановка — Ctrl+C. Повторный запуск той же командой продолжает с места остановки:
     готовые шарды пропускаются, незаконченный шард генерируется заново.

ФОРМАТ (каждый шард — два файла):
  shard_XXXXX.tsv  строки: fen \t score_stm_cp \t result_stm \t ply \t depth \t game_id
  shard_XXXXX.idx  uint64 little-endian: хеш Zobrist позиции, по одному на строку TSV
    score_stm_cp — оценка последней итерации поиска со стороны, которая ходит, в сантипешках.
                   Мат-оценки (|score| >= 10000) не сохраняются.
    result_stm   — исход партии со стороны, которая ходит: 1.0 / 0.5 / 0.0
    ply          — номер полухода от начала партии (включая случайное начало)
    depth        — глубина последней завершённой итерации поиска
    game_id      — <шард>-<номер партии в шарде>

РАЗНООБРАЗИЕ И ДУБЛИКАТЫ:
  - случайное начало: 4–12 случайных легальных ходов (--rand-min / --rand-max);
  - шум: после --noise-from полухода с вероятностью --noise вместо хода движка играется
    случайный легальный ход. Метка этой позиции остаётся от поиска;
  - дубликаты: по хешу позиции (без счётчиков ходов). Проверяются внутри шарда, против всех
    уже записанных шардов (включая шарды других процессов — они подгружаются перед каждым
    шардом) и повторы внутри партии (например, движение короля в ничьей). Окончательная
    проверка на всём наборе — merge.py.
  - позиции с шахом и мат-оценки не сохраняются.

Скорость: почти всё время уходит на поиск движка. Лишние запросы убраны: нет isready перед
каждым ходом, нет ucinewgame между партиями (TT сохраняется).
"""
import argparse, json, os, random, re, subprocess, sys, threading, queue, time
from collections import Counter

import numpy as np
import chess
import chess.polyglot

MATE_T = 10000
INFO_RE = re.compile(r"info depth (\d+) score (cp|mate) (-?\d+)")


def ts():
    return time.strftime("%H:%M:%S")


class Engine:
    def __init__(self, exe, net, hash_mb):
        exe = os.path.abspath(exe)
        self.p = subprocess.Popen([exe], cwd=os.path.dirname(exe), stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        self.q = queue.Queue()
        threading.Thread(target=self._rd, daemon=True).start()
        self.send("uci"); self.wait("uciok")
        self.send(f"setoption name Hash value {hash_mb}")
        self.send(f"setoption name EvalFile value {os.path.abspath(net)}")
        self.send("isready"); self.wait("readyok")

    def _rd(self):
        for ln in self.p.stdout:
            self.q.put(ln.rstrip("\r\n"))
        self.q.put(None)

    def send(self, s):
        self.p.stdin.write(s + "\n"); self.p.stdin.flush()

    def wait(self, token, timeout=170):
        end = time.time() + timeout
        while True:
            ln = self.q.get(timeout=max(0.1, end - time.time()))
            if ln is None:
                raise RuntimeError("engine eof")
            if ln.startswith(token):
                return ln

    def search(self, start_fen, moves, nodes):
        """Возвращает (score_stm_cp | None, depth, bestmove). Без isready: команды идут по порядку."""
        cmd = f"position fen {start_fen}"
        if moves:
            cmd += " moves " + " ".join(moves)
        self.send(cmd)
        self.send(f"go nodes {nodes}")
        score, depth = None, 0
        while True:
            ln = self.q.get(timeout=170)
            if ln is None:
                raise RuntimeError("engine eof during search")
            if ln.startswith("info depth"):
                m = INFO_RE.match(ln)
                if m:
                    depth = int(m.group(1))
                    v = int(m.group(3))
                    if m.group(2) == "mate":
                        v = MATE_T if v > 0 else -MATE_T
                    score = v
            elif ln.startswith("bestmove"):
                return score, depth, ln.split()[1]

    def close(self):
        try:
            self.send("quit")
        except Exception:
            pass
        try:
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


class EngineHolder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.eng = Engine(cfg.engine, cfg.net, cfg.hash)

    def restart(self):
        try:
            self.eng.close()
        except Exception:
            pass
        time.sleep(1)
        self.eng = Engine(self.cfg.engine, self.cfg.net, self.cfg.hash)


class Known:
    """Отсортированный массив хешей всех записанных позиций (свои и чужие шарды)."""
    def __init__(self, out):
        self.out = out
        self.loaded = set()
        self.arr = np.empty(0, dtype=np.uint64)

    def refresh(self):
        new = []
        for fn in sorted(os.listdir(self.out)):
            if fn.endswith(".idx") and fn not in self.loaded:
                new.append(np.fromfile(os.path.join(self.out, fn), dtype="<u8"))
                self.loaded.add(fn)
        if new:
            self.arr = np.sort(np.concatenate([self.arr] + new))

    def mark(self, idx_name, hs):
        self.loaded.add(idx_name)
        self.arr = np.sort(np.concatenate([self.arr, hs]))


def dedupe_mask(rows, known):
    """True для строк, которые надо оставить: первая встреча хеша и хеша нет в known."""
    n = len(rows)
    if n == 0:
        return np.zeros(0, dtype=bool)
    hs = np.fromiter((r[1] for r in rows), dtype=np.uint64, count=n)
    _, first = np.unique(hs, return_index=True)
    keep = np.zeros(n, dtype=bool)
    keep[first] = True
    if known.arr.size:
        pos = np.minimum(np.searchsorted(known.arr, hs), known.arr.size - 1)
        keep &= known.arr[pos] != hs
    return keep


def play_game(eng, rng, cfg):
    board = chess.Board()
    for _ in range(rng.randint(cfg.rand_min, cfg.rand_max)):
        legal = list(board.legal_moves)
        if not legal or board.is_game_over(claim_draw=True):
            break
        board.push(rng.choice(legal))
    start_fen = board.fen()
    start_ply = len(board.move_stack)

    played, pending = [], []
    n_noise = 0
    skipped_check = 0
    win_streak = {chess.WHITE: 0, chess.BLACK: 0}
    draw_streak = 0
    result, term = "1/2-1/2", "UNKNOWN"
    while True:
        if board.is_game_over(claim_draw=True):
            out = board.outcome(claim_draw=True)
            if out:
                result, term = out.result(), out.termination.name
            break
        ply = start_ply + len(played)
        if ply >= cfg.max_plies:
            result, term = "1/2-1/2", "MAXPLIES"
            break
        in_check = board.is_check()
        # Позиция с шахом не сохраняется как метка: ход нужен, но полный поиск не нужен.
        nodes = cfg.check_nodes if in_check else cfg.nodes
        score, depth, bm = eng.search(start_fen, played, nodes)
        if bm in ("0000", "(none)"):
            result, term = "1/2-1/2", "NOMOVE"
            break
        if in_check:
            skipped_check += 1
        else:
            pending.append((board.fen(), chess.polyglot.zobrist_hash(board), score, depth, ply,
                            board.turn))
            # Адъюдикация: досрочный исход, чтобы не тратить время на заведомо решённые партии.
            if score is not None and cfg.resign_cp > 0:
                win_streak[board.turn] = win_streak[board.turn] + 1 if score >= cfg.resign_cp else 0
                if win_streak[board.turn] >= cfg.resign_moves:
                    result = "1-0" if board.turn == chess.WHITE else "0-1"
                    term = "ADJ_RESIGN"
                    break
                draw_streak = draw_streak + 1 if abs(score) <= cfg.draw_cp else 0
                if draw_streak >= cfg.draw_moves:
                    result, term = "1/2-1/2", "ADJ_DRAW"
                    break
        if ply >= cfg.noise_from and rng.random() < cfg.noise:
            mv = rng.choice(list(board.legal_moves))
            n_noise += 1
        else:
            mv = chess.Move.from_uci(bm)
            if mv not in board.legal_moves:
                raise RuntimeError(f"illegal move {bm} in {board.fen()}")
        played.append(mv.uci())
        board.push(mv)

    rows, skipped_mate = [], 0
    for fen, h, score, depth, ply, white_to_move in pending:
        if score is None or abs(score) >= MATE_T:
            skipped_mate += 1
            continue
        if result == "1-0":
            r = 1.0 if white_to_move else 0.0
        elif result == "0-1":
            r = 0.0 if white_to_move else 1.0
        else:
            r = 0.5
        rows.append((fen, h, score, r, ply, depth))
    info = dict(result=result, term=term, plies=start_ply + len(played),
                noise=n_noise, skipped_check=skipped_check, skipped_mate=skipped_mate)
    return rows, info


def new_stats():
    return dict(shards=0, pos=0, dup=0, games=0, plies=0, noise=0, restarts=0,
                skipped_check=0, skipped_mate=0, results=Counter(), terms=Counter(),
                abs_score_sum=0.0, score_rows=0, started=time.time())


def build_shard(holder, k, cfg, known, S):
    rng = random.Random(cfg.seed * 1000003 + k)
    rows, games, restarts_here = [], 0, 0
    t0 = time.time(); last_hb = t0
    while True:
        keep = dedupe_mask(rows, known)
        if int(keep.sum()) >= cfg.shard_size:
            break
        try:
            grows, info = play_game(holder.eng, rng, cfg)
        except Exception as e:
            restarts_here += 1
            S["restarts"] += 1
            print(f"[{ts()}] w{cfg.worker} шард {k}: ошибка движка ({e}), перезапуск", flush=True)
            if restarts_here > 20:
                raise RuntimeError("слишком много ошибок движка в одном шарде")
            holder.restart()
            continue
        tag = f"{k}-{games}"
        games += 1
        S["games"] += 1
        S["plies"] += info["plies"]
        S["noise"] += info["noise"]
        S["skipped_check"] += info["skipped_check"]
        S["skipped_mate"] += info["skipped_mate"]
        S["results"][info["result"]] += 1
        S["terms"][info["term"]] += 1
        for fen, h, score, r, ply, depth in grows:
            rows.append((fen, h, score, r, ply, depth, tag))
            S["abs_score_sum"] += abs(score)
            S["score_rows"] += 1
        now = time.time()
        if now - last_hb >= cfg.status_every:
            last_hb = now
            uniq = int(dedupe_mask(rows, known).sum())
            rate = uniq / max(1e-9, now - t0)   # уникальные позиции текущего шарда в секунду
            print(f"[{ts()}] w{cfg.worker} шард {k} в работе: партий {games}, строк {len(rows)}, "
                  f"уникальных {uniq}/{cfg.shard_size}, {rate:.1f} поз/с", flush=True)
    keep = dedupe_mask(rows, known)
    dup = len(rows) - int(keep.sum())
    kept = [r for r, kp in zip(rows, keep) if kp][:cfg.shard_size]
    return kept, games, dup, time.time() - t0


def write_shard(out, k, kept):
    base = os.path.join(out, f"shard_{k:05d}")
    tsv_tmp, idx_tmp = base + ".tsv.tmp", base + ".idx.tmp"
    with open(tsv_tmp, "w", encoding="utf-8", newline="\n") as f:
        for fen, h, score, r, ply, depth, tag in kept:
            f.write(f"{fen}\t{score}\t{r:.1f}\t{ply}\t{depth}\t{tag}\n")
    hs = np.fromiter((r[1] for r in kept), dtype=np.uint64, count=len(kept))
    with open(idx_tmp, "wb") as f:
        hs.astype("<u8").tofile(f)
    os.replace(tsv_tmp, base + ".tsv")     # сначала TSV, потом IDX: IDX означает «шард готов»
    os.replace(idx_tmp, base + ".idx")
    return hs


def write_readme(out):
    path = os.path.join(out, "README.txt")
    if os.path.exists(path):
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "Шарды генератора ep49.\n"
            "shard_XXXXX.tsv: fen | score_stm_cp | result_stm | ply | depth | game_id\n"
            "shard_XXXXX.idx: uint64 хеш позиции на строку TSV (для дедупликации)\n"
            "score_stm_cp: оценка со стороны, которая ходит; мат-оценки не сохраняются\n"
            "result_stm: 1.0 / 0.5 / 0.0 со стороны, которая ходит\n"
            "Объединение и окончательная дедупликация: python merge.py --src <папка> --dst train.tsv\n")


def main():
    ap = argparse.ArgumentParser(description="Генератор позиций ep49 (см. шапку файла)")
    ap.add_argument("--engine", required=True, help="движок без трассировки UCI")
    ap.add_argument("--net", required=True, help="файл сети ep49")
    ap.add_argument("--out", required=True, help="папка для шардов")
    ap.add_argument("--shards", type=int, default=1000, help="конец диапазона шардов (не включая)")
    ap.add_argument("--start", type=int, default=0, help="первый шард диапазона")
    ap.add_argument("--shard-size", type=int, default=10000, help="уникальных позиций в шарде")
    ap.add_argument("--nodes", type=int, default=5000, help="узлов на ход (метка)")
    ap.add_argument("--check-nodes", type=int, default=1000, help="узлов на ход в позиции с шахом (без метки)")
    ap.add_argument("--hash", type=int, default=64, help="Hash движка, МБ, на один процесс")
    ap.add_argument("--rand-min", type=int, default=4)
    ap.add_argument("--rand-max", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.01, help="вероятность случайного хода на полуход")
    ap.add_argument("--noise-from", type=int, default=12, help="с какого полухода применять шум")
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--resign-cp", type=int, default=1000, help="адъюдикация: порог оценки, 0 = выключено")
    ap.add_argument("--resign-moves", type=int, default=8, help="сколько подряд ходов стороны выше порога")
    ap.add_argument("--draw-cp", type=int, default=10, help="адъюдикация ничьей: |оценка| не больше")
    ap.add_argument("--draw-moves", type=int, default=40, help="сколько подряд позиций равны для ничьи")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1, help="сколько процессов генерации всего")
    ap.add_argument("--worker", type=int, default=0, help="номер этого процесса: 0..workers-1")
    ap.add_argument("--status-every", type=int, default=60, help="период вывода прогресса внутри шарда, с")
    cfg = ap.parse_args()
    if not (0 <= cfg.worker < cfg.workers):
        sys.exit("--worker должен быть в диапазоне 0..workers-1")

    os.makedirs(cfg.out, exist_ok=True)
    if cfg.worker == 0:
        write_readme(cfg.out)
    known = Known(cfg.out)
    known.refresh()
    holder = EngineHolder(cfg)
    S = new_stats()
    cpus = os.cpu_count()
    print(f"[{ts()}] w{cfg.worker}/{cfg.workers} старт: шардов {cfg.shards} по {cfg.shard_size}, "
          f"узлов {cfg.nodes}, Hash {cfg.hash} МБ, шум {cfg.noise}, логич. ядер в системе {cpus}, "
          f"уже известно позиций {known.arr.size}", flush=True)
    try:
        for k in range(cfg.start, cfg.shards):
            if k % cfg.workers != cfg.worker:
                continue
            base = os.path.join(cfg.out, f"shard_{k:05d}")
            if os.path.exists(base + ".tsv") and os.path.exists(base + ".idx"):
                continue
            known.refresh()
            kept, games, dup, dt = build_shard(holder, k, cfg, known, S)
            hs = write_shard(cfg.out, k, kept)
            known.mark(f"shard_{k:05d}.idx", hs)
            S["shards"] += 1
            S["pos"] += len(kept)
            S["dup"] += dup
            el = time.time() - S["started"]
            rate_own = S["pos"] / max(1e-9, el)
            done_all = sum(1 for fn in os.listdir(cfg.out) if fn.endswith(".idx"))
            remaining = max(0, cfg.shards - done_all)
            eta = remaining * cfg.shard_size / max(1e-9, rate_own * cfg.workers) / 3600
            n_res = sum(S["results"].values()) or 1
            w = S["results"].get("1-0", 0) / n_res * 100
            d = S["results"].get("1/2-1/2", 0) / n_res * 100
            l = S["results"].get("0-1", 0) / n_res * 100
            avg_plies = S["plies"] / max(1, S["games"])
            mean_abs = S["abs_score_sum"] / max(1, S["score_rows"])
            print(f"[{ts()}] w{cfg.worker} шард {k} готов: +{len(kept)} поз, дублей отброшено {dup}, "
                  f"партий {games}, {dt:.0f} с | мой темп {rate_own:.1f} поз/с, всего ~{rate_own*cfg.workers:.0f} поз/с | "
                  f"шардов в папке {done_all}/{cfg.shards}, оценка осталось ~{eta:.1f} ч | "
                  f"W/D/L белые {w:.0f}/{d:.0f}/{l:.0f}% | ср. полуходов {avg_plies:.0f} | "
                  f"ср. |оценка| {mean_abs:.0f} сП | шум {S['noise']} | перезапуски {S['restarts']}",
                  flush=True)
            stats = {k_: (dict(v) if isinstance(v, Counter) else v) for k_, v in S.items()}
            stats["started"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(S["started"]))
            stats["elapsed_s"] = round(el, 1)
            with open(os.path.join(cfg.out, f"stats_w{cfg.worker}.json"), "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=1)
    except KeyboardInterrupt:
        print(f"\n[{ts()}] w{cfg.worker} остановлен пользователем. Незаконченный шард будет пересоздан при запуске.",
              flush=True)
    finally:
        holder.eng.close()
    print(f"[{ts()}] w{cfg.worker} завершён: шардов {S['shards']}, позиций {S['pos']}.", flush=True)


if __name__ == "__main__":
    main()
