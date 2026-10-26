// ============================================================================
// MyChessEngine, BUILD 1.5 (Visual Studio: C++17, x64, Release)
//
// КРАТКО: в build 1.5 переписан весь поисковый слой (negamax, qsearch, порядок
// ходов, итеративное углубление, PV, вывод info). Доска, генератор ходов, SEE,
// оценка (ручная и NNUE), UCI и тесты сохранены из build 1.4.
//
// ЧТО ИМЕННО ПЕРЕПИСАНО И ПОЧЕМУ — см. комментарии вида «build 1.5» в тексте и файлы:
//   CHANGES_build1.5.txt  — подробный список изменений, с причинами и результатами
//   BUILD_VS_build1.5.txt — как собрать в Visual Studio и проверить
//
// ОСНОВНЫЕ ИЗМЕНЕНИЯ (кратко):
//   1. Исправлен вывод мата: теперь "score mate N" (в 1.4 всегда писалось "score cp").
//   2. Продление за шах ограничено: не более 4 раз на путь от корня (в 1.4 без лимита,
//      это давало взрыв узлов на позициях с длинными шахами).
//   3. Добавлен mate-distance pruning (отсечение по дистанции до мата).
//   4. Удалён reverse futility pruning (RFP): он отсекал узел, где мат в 2 хода
//      виден только при полном поиске (проверено на позиции 15 мат-в-2).
//   5. LMR + PVS с полным ре-серчем, LMP на малых глубинах.
//   6. Порядок ходов: TT, взятия по SEE, promo, killer 1/2, countermove, история
//      с gravity (история больше не растёт без границ), плохие взятия в конце.
//   7. Тихие шахи в qsearch на первом уровне за горизонтом (закрывают мат на горизонте).
//   8. Исправлено завершение: поток поиска присоединяется перед выходом
//      (в 1.4 возможен "terminate called" при выходе).
//
// РЕЗУЛЬТАТЫ (кратко, подробнее в CHANGES_build1.5.txt):
//   мат-в-2 16/16 (было 14/16), мат-в-3 10/10, узлы bench d10 −45%,
//   6 позиций d11 узлов −70%, скорость (nps) в bench ниже из-за тихих шахов в qsearch.
//
// Файл сети по умолчанию: net_comp_ft_ep49.nnue (рядом с exe или путь EvalFile). Хеш 0x6F45FA7F.
// ============================================================================

// ============================================================================
// build 1.4 (версия Chess768) — переписана под одну сеть.
// Оценка: NNUE nnue-pytorch, признаки Chess768 (768, без королевских бакетов),
// L1=160, L2=L3=32, один стек слоёв (без выбора по фазам материала).
// Файл сети: net_comp_ft_ep49.nnue (основная) или net_c768.nnue (запасная); serialize.py --features=Chess768 --l1=160.
// Поиск, генерация ходов и ручная оценка (evaluate_bb) взяты из build 1.3 без изменений.
// Без файла сети поведение совпадает с build 1.3.
// UCI: setoption name EvalFile value <путь|off>; nneval <FEN>; nnuecheck <depth> <FEN>; nnueinfo.
// Исправления (fix1): сброс кэша аккумуляторов при смене сети; пути UTF-8 (кириллица в Windows);
// option EvalFile в uci; относительный путь ищется и рядом с exe; предупреждение о хеше сети.
// Прежняя сеть net_v0 (HalfKAv2_hm^, 32 бакета) этой сборкой не поддерживается.
// ============================================================================

#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <cstdio>
#include <cstring>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cerrno>
#include <climits>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <string>
#include <thread>
#include <atomic>
#include <vector>
#include <utility>   // build 1.5: std::swap в сортировке ходов
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <wchar.h>
#endif

// ----------------------------------------------------------------------------
// Portable bit intrinsics (GCC/Clang builtins or MSVC intrinsics).
// Argument must be non-zero for ctz/clz, as with the builtins.
// ----------------------------------------------------------------------------
#if defined(_MSC_VER)
#include <intrin.h>
static inline int bb_ctzll(uint64_t x) { unsigned long i; _BitScanForward64(&i, x); return (int)i; }
static inline int bb_clzll(uint64_t x) { unsigned long i; _BitScanReverse64(&i, x); return 63 - (int)i; }
static inline int bb_popcountll(uint64_t x) { return (int)__popcnt64(x); }
#else
static inline int bb_ctzll(uint64_t x) { return __builtin_ctzll(x); }
static inline int bb_clzll(uint64_t x) { return __builtin_clzll(x); }
static inline int bb_popcountll(uint64_t x) { return __builtin_popcountll(x); }
#endif
// ------------------------------- POSITION -----------------------------------
enum Piece : int { EMPTY = 0, PAWN = 1, KNIGHT = 2, BISHOP = 3, ROOK = 4, QUEEN = 5, KING = 6 };
enum Color : int { WHITE = 0, BLACK = 1, COLOR_NB = 2 };
constexpr int WK = 1, WQ = 2, BK = 4, BQ = 8;
constexpr int NO_SQ = 127;
constexpr int file_of(int sq) { return sq & 7; }
constexpr int rank_of(int sq) { return sq >> 4; }
constexpr int sq88(int f, int r) { return r * 16 + f; }
constexpr bool on_board(int sq) { return (sq & 0x88) == 0; }
constexpr bool valid_file(int f) { return (unsigned)f < 8; }
constexpr bool valid_rank(int r) { return (unsigned)r < 8; }
inline int piece_of(int pc) { return pc & 7; }
inline int color_of(int pc) { return pc >> 3; }
inline int make_pc(int p, int c) { return p | (c << 3); }

// -----------------------------------------------------------------------------
// build1.1-bitboard stages 1-3
//
// A 64-bit square is mapped as rank * 8 + file. Bitboards are authoritative;
// the compact 64-square piece lookup is maintained only for piece-at-square
// access and for the preserved 0x88 reference algorithms.
// -----------------------------------------------------------------------------
using Bitboard = uint64_t;
constexpr int bb_sq_from_sq88(int sq) {
    return rank_of(sq) * 8 + file_of(sq);
}
constexpr int sq88_from_bb_sq(int sq) {
    return (sq >> 3) * 16 + (sq & 7);
}
constexpr Bitboard bb_square(int sq64) {
    return Bitboard(1) << sq64;
}

struct Position {
    // Bitboards are the authoritative occupancy/state representation. This
    // compact 64-square lookup is only a piece-at-square accelerator; there is
    // no 0x88 board storage in the active position.
    uint8_t piece[64];
    Bitboard bbPieces[COLOR_NB][7];
    Bitboard bbOccupied[COLOR_NB];
    Bitboard bbOccupiedAll;
    int sideToMove;
    int castle;
    int ep;
    int halfmove;
    int fullmove;
    int ksq[COLOR_NB]; // build1.1 candidate: cached king squares, derived from board
    uint64_t key;
};

static uint8_t bb_invalid_piece = EMPTY;
static inline uint8_t& bb_piece_ref(Position& p, int sq88) {
    return on_board(sq88) ? p.piece[bb_sq_from_sq88(sq88)] : bb_invalid_piece;
}
static inline const uint8_t& bb_piece_ref(const Position& p, int sq88) {
    return on_board(sq88) ? p.piece[bb_sq_from_sq88(sq88)] : bb_invalid_piece;
}
int findKing(const Position& p, int color);
static void clear_pos(Position& p) {
    std::memset(p.piece, 0, sizeof p.piece);
    p.sideToMove = WHITE;
    p.castle = 0;
    p.ep = NO_SQ;
    p.halfmove = 0;
    p.fullmove = 1;
    p.ksq[WHITE] = p.ksq[BLACK] = NO_SQ;
    std::memset(p.bbPieces, 0, sizeof p.bbPieces);
    std::memset(p.bbOccupied, 0, sizeof p.bbOccupied);
    p.bbOccupiedAll = 0;
    p.key = 0;
}

static void bb_clear(Position& p) {
    std::memset(p.bbPieces, 0, sizeof p.bbPieces);
    std::memset(p.bbOccupied, 0, sizeof p.bbOccupied);
    p.bbOccupiedAll = 0;
}

static void bb_add_piece(Position& p, int pc, int sq88) {
    if (!pc || !on_board(sq88)) return;
    const int c = color_of(pc), pt = piece_of(pc);
    const Bitboard b = bb_square(bb_sq_from_sq88(sq88));
    p.bbPieces[c][pt] |= b;
    p.bbOccupied[c] |= b;
    p.bbOccupiedAll |= b;
}

static void bb_remove_piece(Position& p, int pc, int sq88) {
    if (!pc || !on_board(sq88)) return;
    const int c = color_of(pc), pt = piece_of(pc);
    const Bitboard b = bb_square(bb_sq_from_sq88(sq88));
    p.bbPieces[c][pt] &= ~b;
    p.bbOccupied[c] &= ~b;
    p.bbOccupiedAll &= ~b;
}

static void bb_move_piece(Position& p, int pc, int from, int to) {
    bb_remove_piece(p, pc, from);
    bb_add_piece(p, pc, to);
}

static void bb_rebuild(Position& p) {
    bb_clear(p);
    for (int sq = 0; sq < 128; ++sq) {
        if (on_board(sq) && bb_piece_ref(p, sq))
            bb_add_piece(p, bb_piece_ref(p, sq), sq);
    }
}

static bool bb_piece_lookup_matches(const Position& p);

static bool bb_state_matches(const Position& p) {
    Position ref = p;
    bb_rebuild(ref);
    return std::memcmp(p.bbPieces, ref.bbPieces, sizeof p.bbPieces) == 0 &&
        std::memcmp(p.bbOccupied, ref.bbOccupied, sizeof p.bbOccupied) == 0 &&
        p.bbOccupiedAll == ref.bbOccupiedAll &&
        bb_piece_lookup_matches(p);
}

static bool bb_piece_lookup_matches(const Position& p) {
    for (int sq64 = 0; sq64 < 64; ++sq64) {
        const Bitboard bit = bb_square(sq64);
        int expected = EMPTY;
        for (int c = WHITE; c <= BLACK; ++c)
            for (int pt = PAWN; pt <= KING; ++pt)
                if (p.bbPieces[c][pt] & bit)
                    expected = make_pc(pt, c);
        if (p.piece[sq64] != expected) return false;
    }
    return true;
}

static uint64_t zPiece[16][128], zCastle[16], zEp[128], zSide;
static uint64_t splitmix64(uint64_t& x) {
    x += 0x9e3779b97f4a7c15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
static void initZobrist() {
    static bool done = false;
    if (done) return;
    done = true;
    uint64_t s = 0x123456789abcdef0ULL;
    for (int pc = 0; pc < 16; ++pc)
        for (int sq = 0; sq < 128; ++sq)
            zPiece[pc][sq] = splitmix64(s);
    zSide = splitmix64(s);
    for (int i = 0; i < 16; ++i) zCastle[i] = splitmix64(s);
    for (int i = 0; i < 128; ++i) zEp[i] = splitmix64(s);
}
// =============================================================================
// build0.3: canonical en-passant Zobrist normalization — Level A.
//
// Previously computeHash()/makeMove()/the manual null-move code all folded the
// RAW p.ep square into the hash unconditionally. That is wrong in two ways:
//   1. Two positions that differ only by a "dead" ep square (no pawn is
//      actually able to capture en passant) hashed to different keys even
//      though they are the same chess position for all practical purposes
//      (TT probes, repetition, etc.). This inflates TT/repetition misses and
//      can even make a benign FEN look "different" from the canonical one
//      reached by playing the double push over the board.
//   2. A malformed/hand-written FEN could set an ep square with no capturing
//      pawn geometry at all (e.g., no enemy pawn to actually capture), and the
//      raw square still leaked into the hash.
//
// hash_ep_square() below is the single source of truth for "does the ep
// square actually matter for hashing purposes". It performs a cheap, purely
// local board check (no move generation, no recursion) and is used from
// computeHash(), makeMove()'s incremental update, and the manual null-move
// hash patch in search_negamax(), so all three stay consistent with each
// other by construction.
//
// This is intentionally "Level A": it only checks that a pawn of the side to
// move is geometrically adjacent to the capture square and that the captured
// pawn actually exists. It does NOT verify that the capture would be legal
// (e.g. it may still be pinned). Full legality (Level B) is out of scope for
// this pass — gen_pawn()/generateLegalMoves() already reject illegal en
// passant captures during move generation, this function only controls
// Zobrist hashing.
// =============================================================================
static int hash_ep_square(const Position& p) {
    if (p.ep == NO_SQ) return NO_SQ;
    const int us = p.sideToMove;
    const int r = rank_of(p.ep);
    // The ep square must sit on the rank a pawn lands on after a double push:
    // rank 5 (0-indexed) for White to move, rank 2 for Black to move.
    if (us == WHITE) {
        if (r != 5) return NO_SQ;
    }
    else {
        if (r != 2) return NO_SQ;
    }
    const int dir = (us == WHITE) ? 16 : -16;
    const int captured = p.ep - dir;
    if (!on_board(captured)) return NO_SQ;
    if (bb_piece_ref(p, captured) != make_pc(PAWN, us ^ 1)) return NO_SQ;
    const int left = captured - 1;
    const int right = captured + 1;
    bool candidate = false;
    if (on_board(left) && bb_piece_ref(p, left) == make_pc(PAWN, us)) candidate = true;
    if (on_board(right) && bb_piece_ref(p, right) == make_pc(PAWN, us)) candidate = true;
    return candidate ? p.ep : NO_SQ;
}
static uint64_t computeHash(const Position& p) {
    initZobrist();
    uint64_t k = 0;
    for (int sq = 0; sq < 128; ++sq)
        if (bb_piece_ref(p, sq)) k ^= zPiece[bb_piece_ref(p, sq)][sq];
    if (p.sideToMove == BLACK) k ^= zSide;
    k ^= zCastle[p.castle];
    k ^= zEp[hash_ep_square(p)];
    return k;
}
static int char_to_pc(char c) {
    int col = isupper((unsigned char)c) ? WHITE : BLACK;
    switch (tolower((unsigned char)c)) {
    case 'p': return make_pc(PAWN, col);
    case 'n': return make_pc(KNIGHT, col);
    case 'b': return make_pc(BISHOP, col);
    case 'r': return make_pc(ROOK, col);
    case 'q': return make_pc(QUEEN, col);
    case 'k': return make_pc(KING, col);
    default: return EMPTY;
    }
}
static bool parse_fen(Position& p, const char* fen) {
    clear_pos(p);
    int f = 0, r = 7;
    const char* s = fen;
    while (*s && *s != ' ') {
        if (*s == '/') {
            if (f != 8 || r == 0) return false;
            f = 0; --r;
        }
        else if (*s >= '1' && *s <= '8') {
            f += *s - '0';
            if (f > 8) return false;
        }
        else {
            int pc = char_to_pc(*s);
            if (!pc || !valid_file(f) || !valid_rank(r)) return false;
            bb_piece_ref(p, sq88(f, r)) = pc;
            ++f;
        }
        ++s;
    }
    if (f != 8 || r != 0 || *s != ' ') return false;
    ++s;
    if (*s == 'w') p.sideToMove = WHITE;
    else if (*s == 'b') p.sideToMove = BLACK;
    else return false;
    ++s;
    if (*s != ' ') return false;
    ++s;
    if (*s == '-') ++s;
    else {
        while (*s && *s != ' ') {
            if (*s == 'K') p.castle |= WK;
            else if (*s == 'Q') p.castle |= WQ;
            else if (*s == 'k') p.castle |= BK;
            else if (*s == 'q') p.castle |= BQ;
            else return false;
            ++s;
        }
    }
    if (*s != ' ') return false;
    ++s;
    if (*s == '-') { p.ep = NO_SQ; ++s; }
    else {
        if (*s < 'a' || *s > 'h') return false;
        int ef = *s++ - 'a';
        if (*s < '1' || *s > '8') return false;
        int er = *s++ - '1';
        p.ep = sq88(ef, er);
        if (!on_board(p.ep)) return false;
    }
    if (*s == ' ') {
        ++s;
        p.halfmove = 0;
        while (*s >= '0' && *s <= '9') p.halfmove = p.halfmove * 10 + (*s++ - '0');
        if (*s == ' ') {
            ++s;
            p.fullmove = 0;
            while (*s >= '0' && *s <= '9') p.fullmove = p.fullmove * 10 + (*s++ - '0');
            if (p.fullmove < 1) p.fullmove = 1;
        }
    }
    p.ksq[WHITE] = findKing(p, WHITE);
    p.ksq[BLACK] = findKing(p, BLACK);
    bb_rebuild(p);
    p.key = computeHash(p);
    return true;
}
static void set_start(Position& p) {
    parse_fen(p, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}
// ----------------------------- MOVE GENERATOR --------------------------------
constexpr int MAX_MOVES = 256;
enum MoveFlag : int { F_CAPTURE = 1, F_PROMO = 2, F_CASTLE = 4, F_EP = 8, F_DOUBLE = 16 };
struct Move {
    uint8_t from;
    uint8_t to;
    uint8_t promo;
    uint8_t flags;
};
static bool move_equal(const Move& a, const Move& b) {
    return a.from == b.from && a.to == b.to && a.promo == b.promo && a.flags == b.flags;
}
struct MoveList {
    // build1.1 candidate: active entries are written by add(); only count is initialized.
    Move m[MAX_MOVES];
    int count = 0;
    void clear() { count = 0; }
    void add(int from, int to, int promo, int flags) {
        if (count >= MAX_MOVES) return;
        Move& x = m[count++];
        x.from = (uint8_t)from; x.to = (uint8_t)to;
        x.promo = (uint8_t)promo; x.flags = (uint8_t)flags;
    }
};
static const int KNIGHT_DIRS[8] = { 33, 31, 18, 14, -14, -18, -31, -33 };
static const int BISHOP_DIRS[4] = { 17, 15, -15, -17 };
static const int ROOK_DIRS[4] = { 16, 1, -1, -16 };
static const int KING_DIRS[8] = { 17, 16, 15, 1, -1, -15, -16, -17 };

// -----------------------------------------------------------------------------
// build1.1-bitboard stage 4: non-sliding attack tables.
// These tables are deliberately independent of occupancy. Sliding attacks are
// introduced in stage 5. The current 0x88 attack code remains the reference.
// -----------------------------------------------------------------------------
static Bitboard bb_pawn_attacks[COLOR_NB][64];
static Bitboard bb_knight_attacks[64];
static Bitboard bb_king_attacks[64];
static bool bb_attack_tables_ready = false;

static Bitboard bb_reference_step_attacks(int sq64, const int* dirs, int count) {
    Bitboard out = 0;
    const int from = sq88_from_bb_sq(sq64);
    for (int i = 0; i < count; ++i) {
        const int to = from + dirs[i];
        if (on_board(to)) out |= bb_square(bb_sq_from_sq88(to));
    }
    return out;
}

static Bitboard bb_reference_pawn_attacks(int color, int sq64) {
    const int from = sq88_from_bb_sq(sq64);
    const int dir = color == WHITE ? 16 : -16;
    Bitboard out = 0;
    for (int delta : { dir - 1, dir + 1 }) {
        const int to = from + delta;
        if (on_board(to)) out |= bb_square(bb_sq_from_sq88(to));
    }
    return out;
}

static void bb_init_attack_tables() {
    if (bb_attack_tables_ready) return;
    bb_attack_tables_ready = true;
    for (int sq64 = 0; sq64 < 64; ++sq64) {
        bb_pawn_attacks[WHITE][sq64] = bb_reference_pawn_attacks(WHITE, sq64);
        bb_pawn_attacks[BLACK][sq64] = bb_reference_pawn_attacks(BLACK, sq64);
        bb_knight_attacks[sq64] = bb_reference_step_attacks(
            sq64, KNIGHT_DIRS, 8);
        bb_king_attacks[sq64] = bb_reference_step_attacks(
            sq64, KING_DIRS, 8);
    }
}

static Bitboard bb_pawn_attack_mask(int color, int sq64) {
    bb_init_attack_tables();
    return bb_pawn_attacks[color][sq64];
}

static Bitboard bb_knight_attack_mask(int sq64) {
    bb_init_attack_tables();
    return bb_knight_attacks[sq64];
}

static Bitboard bb_king_attack_mask(int sq64) {
    bb_init_attack_tables();
    return bb_king_attacks[sq64];
}

// -----------------------------------------------------------------------------
// build1.1-bitboard stage 5: correct occupancy-aware sliding attacks.
// This remains the simple ray reference implementation used to build and
// exhaustively verify the fancy magic tables below against the existing 0x88
// isSquareAttacked() implementation.
// -----------------------------------------------------------------------------
static Bitboard bb_sliding_attacks_reference(int sq64, Bitboard occupied, bool bishop) {
    static const int bishop_df[4] = { 1, 1, -1, -1 };
    static const int bishop_dr[4] = { 1, -1, 1, -1 };
    static const int rook_df[4] = { 1, -1, 0, 0 };
    static const int rook_dr[4] = { 0, 0, 1, -1 };
    const int* dfs = bishop ? bishop_df : rook_df;
    const int* drs = bishop ? bishop_dr : rook_dr;
    const int count = 4;
    const int file = sq64 & 7;
    const int rank = sq64 >> 3;
    Bitboard attacks = 0;
    for (int i = 0; i < count; ++i) {
        int f = file + dfs[i];
        int r = rank + drs[i];
        while ((unsigned)f < 8 && (unsigned)r < 8) {
            const int to = r * 8 + f;
            const Bitboard bit = bb_square(to);
            attacks |= bit;
            if (occupied & bit) break;
            f += dfs[i];
            r += drs[i];
        }
    }
    return attacks;
}

// Fancy magic tables: the relevant occupancy bits for each square are hashed
// by a 64-bit multiply and a right shift into a compact per-square table.
// The reference ray implementation above remains available for initialization
// and exhaustive differential tests.
struct BBMagicEntry {
    Bitboard mask;
    Bitboard magic;
    uint32_t offset;
    uint8_t shift;
};

// generated magic constants; square = rank*8+file, a1=0
static const uint64_t bb_magic_bishop[64] = {
0x10102002004a1420ULL ,0x8020040400584008ULL ,
0x10510800811201c8ULL ,0x5204042080000088ULL ,
0x2204106880000002ULL ,0x1401042004000000ULL ,
0x0400880410042004ULL ,0x0028208200a02020ULL ,
0x1500241990010e00ULL ,0x8001200182020a40ULL ,
0x40004101030b0000ULL ,0x8002041042000100ULL ,
0x4010011041020038ULL ,0x0000010421044000ULL ,
0x1500210808020a00ULL ,0x8000088400880520ULL ,
0x0405004010040100ULL ,0x1005823210040108ULL ,
0x2708008102040011ULL ,0x4048200404009100ULL ,
0x0018104101400024ULL ,0x0003000601190101ULL ,
0x8004803108491000ULL ,0x8014241200820800ULL ,
0x0006e080100c3040ULL ,0x0501044a11041800ULL ,
0x9020300008004045ULL ,0x0894080000220040ULL ,
0x1001010083104000ULL ,0x5004030040900080ULL ,
0x000400422c012400ULL ,0x0002128698404812ULL ,
0x1010108404900440ULL ,0x0928021182084100ULL ,
0x2006080409020024ULL ,0x1010202020180080ULL ,
0xa010008200202200ULL ,0x2098015100019004ULL ,
0x0002041440810811ULL ,0x802a02020000b098ULL ,
0x0009015090004060ULL ,0x4000821082081001ULL ,
0x0100210040420800ULL ,0x0800004010488a00ULL ,
0x2000081104004040ULL ,0x4c8e029015000082ULL ,
0x0420340322224842ULL ,0x1298260043400210ULL ,
0x0000822802400008ULL ,0x00008a0101600000ULL ,
0x3040003412080021ULL ,0x3040290220884800ULL ,
0x4a1500401041004aULL ,0x8010200282020781ULL ,
0x0020203142209091ULL ,0x0070300600902110ULL ,
0x0040808800b62048ULL ,0x0000810400c44420ULL ,
0x00080400440c0441ULL ,0x8340080020840411ULL ,
0x0000000104208200ULL ,0x0000800810d00080ULL ,
0x0400530411080200ULL ,0x4040702400932244ULL
};
static const uint64_t bb_magic_rook[64] = {
0x1080004008801020ULL ,0x0840092002c03000ULL ,
0x1900200010400900ULL ,0x0880100008000480ULL ,
0x4200100420080200ULL ,0x8100020100080400ULL ,
0x0200040110886200ULL ,0x0200008040220411ULL ,
0x0404800084400220ULL ,0x0000401000402000ULL ,
0x0086001081220440ULL ,0x0408800800100280ULL ,
0x000a001201040820ULL ,0x8848800200840080ULL ,
0x4001000100040200ULL ,0x0442000102105084ULL ,
0x9080010020804100ULL ,0x0040404000201009ULL ,
0x0000808010002009ULL ,0x2200090021d00100ULL ,
0x0008008008040080ULL ,0x0004004002010040ULL ,
0x0011040008015042ULL ,0x00000a0001768104ULL ,
0x0000800080204009ULL ,0x2010004140002001ULL ,
0x9800200280100080ULL ,0x1000100080080080ULL ,
0x0442000a00049020ULL ,0x2100040080020080ULL ,
0x0800120400900148ULL ,0x0010040a00128541ULL ,
0x2800804000800030ULL ,0x1010002000400041ULL ,
0x4000200011004100ULL ,0x0610008410800800ULL ,
0x0400802402800800ULL ,0xc100020080800400ULL ,
0x0002000802000401ULL ,0x0182085882000401ULL ,
0x0220204000808000ULL ,0x2860100040024022ULL ,
0x0001002004110040ULL ,0x99101042000a0020ULL ,
0x0004080004008080ULL ,0x0010040002008080ULL ,
0x2012004881020004ULL ,0x8300842444820011ULL ,
0x0088403882010200ULL ,0x0820400080210100ULL ,
0x0110910040a00300ULL ,0x0801100280080480ULL ,
0x0242009008200600ULL ,0x1002000489500200ULL ,
0x0040800200010080ULL ,0x0091800041000080ULL ,
0x0000209300488001ULL ,0x04c1002414824001ULL ,
0x020020000b001041ULL ,0x7000100004200901ULL ,
0x8002002004100802ULL ,0x30010002084c0007ULL ,
0x0888221800813004ULL ,0x4000002840840112ULL
};

static BBMagicEntry bb_bishop_magic_info[64];
static BBMagicEntry bb_rook_magic_info[64];
static std::vector<Bitboard> bb_magic_attack_table;
static bool bb_magic_tables_ready = false;
static bool bb_magic_tables_valid = false;

static Bitboard bb_magic_relevant_mask(int sq64, bool bishop) {
    static const int bishop_df[4] = { 1, 1, -1, -1 };
    static const int bishop_dr[4] = { 1, -1, 1, -1 };
    static const int rook_df[4] = { 1, -1, 0, 0 };
    static const int rook_dr[4] = { 0, 0, 1, -1 };
    const int* dfs = bishop ? bishop_df : rook_df;
    const int* drs = bishop ? bishop_dr : rook_dr;
    const int file = sq64 & 7;
    const int rank = sq64 >> 3;
    Bitboard mask = 0;
    for (int i = 0; i < 4; ++i) {
        int f = file + dfs[i];
        int r = rank + drs[i];
        while ((unsigned)f < 8 && (unsigned)r < 8) {
            const int next_f = f + dfs[i];
            const int next_r = r + drs[i];
            // The outermost square cannot change the attack set: it is
            // attacked whether occupied or empty, so it is excluded from the
            // hash key. All inner ray squares remain relevant blockers.
            if ((unsigned)next_f >= 8 || (unsigned)next_r >= 8) break;
            mask |= bb_square(r * 8 + f);
            f = next_f;
            r = next_r;
        }
    }
    return mask;
}

static Bitboard bb_magic_subset_from_index(uint32_t index, Bitboard mask) {
    Bitboard subset = 0;
    while (mask) {
        const int sq64 = bb_ctzll(mask);
        mask &= mask - 1;
        if (index & 1u) subset |= bb_square(sq64);
        index >>= 1;
    }
    return subset;
}

static bool bb_init_magic_tables() {
    if (bb_magic_tables_ready) return bb_magic_tables_valid;
    bb_magic_tables_ready = true;
    bb_magic_tables_valid = true;
    bb_magic_attack_table.clear();
    bb_magic_attack_table.reserve(108000);

    auto init_piece = [&](bool bishop, BBMagicEntry* info,
        const uint64_t* magics) {
            for (int sq64 = 0; sq64 < 64; ++sq64) {
                BBMagicEntry& entry = info[sq64];
                entry.mask = bb_magic_relevant_mask(sq64, bishop);
                const int bits = bb_popcountll(entry.mask);
                entry.shift = static_cast<uint8_t>(64 - bits);
                entry.magic = magics[sq64];
                entry.offset = static_cast<uint32_t>(bb_magic_attack_table.size());
                const uint32_t variations = 1u << bits;
                bb_magic_attack_table.resize(
                    bb_magic_attack_table.size() + variations, 0);
                std::vector<uint8_t> assigned(variations, 0);
                for (uint32_t i = 0; i < variations; ++i) {
                    const Bitboard occupied =
                        bb_magic_subset_from_index(i, entry.mask);
                    const uint32_t magic_index = static_cast<uint32_t>(
                        ((occupied * entry.magic) >> entry.shift));
                    const Bitboard attacks =
                        bb_sliding_attacks_reference(sq64, occupied, bishop);
                    Bitboard& slot =
                        bb_magic_attack_table[entry.offset + magic_index];
                    if (assigned[magic_index]) {
                        if (slot != attacks) {
                            bb_magic_tables_valid = false;
                            return;
                        }
                    }
                    else {
                        slot = attacks;
                        assigned[magic_index] = 1;
                    }
                }
            }
        };

    init_piece(true, bb_bishop_magic_info, bb_magic_bishop);
    if (bb_magic_tables_valid)
        init_piece(false, bb_rook_magic_info, bb_magic_rook);
    if (!bb_magic_tables_valid) {
        bb_magic_attack_table.clear();
        std::printf("info string magic tables invalid, using reference rays\n");
    }
    return bb_magic_tables_valid;
}

static inline Bitboard bb_magic_lookup(const BBMagicEntry& entry,
    Bitboard occupied) {
    const uint32_t index = static_cast<uint32_t>(
        (((occupied & entry.mask) * entry.magic) >> entry.shift));
    return bb_magic_attack_table[entry.offset + index];
}

static Bitboard bb_bishop_attack_mask(int sq64, Bitboard occupied) {
    if (!bb_init_magic_tables())
        return bb_sliding_attacks_reference(sq64, occupied, true);
    return bb_magic_lookup(bb_bishop_magic_info[sq64], occupied);
}

static Bitboard bb_rook_attack_mask(int sq64, Bitboard occupied) {
    if (!bb_init_magic_tables())
        return bb_sliding_attacks_reference(sq64, occupied, false);
    return bb_magic_lookup(bb_rook_magic_info[sq64], occupied);
}

static Bitboard bb_queen_attack_mask(int sq64, Bitboard occupied) {
    return bb_bishop_attack_mask(sq64, occupied) |
        bb_rook_attack_mask(sq64, occupied);
}

static void add_promos(MoveList& ml, int from, int to, int flags) {
    static const int pr[4] = { QUEEN, ROOK, BISHOP, KNIGHT };
    for (int i = 0; i < 4; ++i) ml.add(from, to, pr[i], flags | F_PROMO);
}
static void gen_steps(const Position& p, int sq, int us, const int* d, int n, MoveList& ml) {
    for (int i = 0; i < n; ++i) {
        int to = sq + d[i];
        if (!on_board(to)) continue;
        int pc = bb_piece_ref(p, to);
        if (!pc) ml.add(sq, to, 0, 0);
        else if (color_of(pc) != us) ml.add(sq, to, 0, F_CAPTURE);
    }
}
static void gen_slides(const Position& p, int sq, int us, const int* d, int n, MoveList& ml) {
    for (int i = 0; i < n; ++i) {
        for (int to = sq + d[i]; on_board(to); to += d[i]) {
            int pc = bb_piece_ref(p, to);
            if (!pc) { ml.add(sq, to, 0, 0); continue; }
            if (color_of(pc) != us) ml.add(sq, to, 0, F_CAPTURE);
            break;
        }
    }
}
static void gen_pawn(const Position& p, int sq, int us, MoveList& ml) {
    const int dir = (us == WHITE) ? 16 : -16;
    const int startRank = (us == WHITE) ? 1 : 6;
    const int promoRank = (us == WHITE) ? 7 : 0;
    int one = sq + dir;
    if (on_board(one) && bb_piece_ref(p, one) == EMPTY) {
        if (rank_of(one) == promoRank) add_promos(ml, sq, one, 0);
        else {
            ml.add(sq, one, 0, 0);
            int two = one + dir;
            if (rank_of(sq) == startRank && on_board(two) && bb_piece_ref(p, two) == EMPTY)
                ml.add(sq, two, 0, F_DOUBLE);
        }
    }
    const int caps[2] = { dir - 1, dir + 1 };
    for (int i = 0; i < 2; ++i) {
        int to = sq + caps[i];
        if (!on_board(to)) continue;
        int pc = bb_piece_ref(p, to);
        if (pc) {
            if (color_of(pc) == us) continue;
            if (rank_of(to) == promoRank) add_promos(ml, sq, to, F_CAPTURE);
            else ml.add(sq, to, 0, F_CAPTURE);
        }
        else if (p.ep != NO_SQ && to == p.ep) {
            // build0.3: an ep square from a malformed/hand-written FEN (or any
            // stale raw p.ep) must not produce a phantom en-passant capture
            // when there is no actual pawn to remove. We require:
            //   - the ep square sits on the correct rank for `us` to capture
            //     onto (rank 5 for White, rank 2 for Black);
            //   - the square behind it (relative to `us`) actually holds an
            //     enemy pawn that would be captured.
            const bool rankOk = (us == WHITE) ? (rank_of(p.ep) == 5) : (rank_of(p.ep) == 2);
            if (rankOk) {
                int cs = to + (us == WHITE ? -16 : 16);
                if (on_board(cs) && bb_piece_ref(p, cs) == make_pc(PAWN, us ^ 1)) {
                    ml.add(sq, to, 0, F_CAPTURE | F_EP);
                }
            }
        }
    }
}
static void gen_castle(const Position& p, int us, MoveList& ml) {
    if (us == WHITE) {
        constexpr int e1 = sq88(4, 0);
        if (bb_piece_ref(p, e1) != make_pc(KING, WHITE)) return;
        if ((p.castle & WK) && bb_piece_ref(p, sq88(7, 0)) == make_pc(ROOK, WHITE)
            && !bb_piece_ref(p, sq88(5, 0)) && !bb_piece_ref(p, sq88(6, 0)))
            ml.add(e1, sq88(6, 0), 0, F_CASTLE);
        if ((p.castle & WQ) && bb_piece_ref(p, sq88(0, 0)) == make_pc(ROOK, WHITE)
            && !bb_piece_ref(p, sq88(1, 0)) && !bb_piece_ref(p, sq88(2, 0)) && !bb_piece_ref(p, sq88(3, 0)))
            ml.add(e1, sq88(2, 0), 0, F_CASTLE);
    }
    else {
        constexpr int e8 = sq88(4, 7);
        if (bb_piece_ref(p, e8) != make_pc(KING, BLACK)) return;
        if ((p.castle & BK) && bb_piece_ref(p, sq88(7, 7)) == make_pc(ROOK, BLACK)
            && !bb_piece_ref(p, sq88(5, 7)) && !bb_piece_ref(p, sq88(6, 7)))
            ml.add(e8, sq88(6, 7), 0, F_CASTLE);
        if ((p.castle & BQ) && bb_piece_ref(p, sq88(0, 7)) == make_pc(ROOK, BLACK)
            && !bb_piece_ref(p, sq88(1, 7)) && !bb_piece_ref(p, sq88(2, 7)) && !bb_piece_ref(p, sq88(3, 7)))
            ml.add(e8, sq88(2, 7), 0, F_CASTLE);
    }
}
static void generateMoves_0x88(const Position& p, MoveList& ml) {
    ml.clear();
    const int us = p.sideToMove;
    for (int r = 0; r < 8; ++r) for (int f = 0; f < 8; ++f) {
        int sq = sq88(f, r);
        int pc = bb_piece_ref(p, sq);
        if (!pc || color_of(pc) != us) continue;
        switch (piece_of(pc)) {
        case PAWN:   gen_pawn(p, sq, us, ml); break;
        case KNIGHT: gen_steps(p, sq, us, KNIGHT_DIRS, 8, ml); break;
        case BISHOP: gen_slides(p, sq, us, BISHOP_DIRS, 4, ml); break;
        case ROOK:   gen_slides(p, sq, us, ROOK_DIRS, 4, ml); break;
        case QUEEN:  gen_slides(p, sq, us, BISHOP_DIRS, 4, ml);
            gen_slides(p, sq, us, ROOK_DIRS, 4, ml); break;
        case KING:   gen_steps(p, sq, us, KING_DIRS, 8, ml); break;
        default: break;
        }
    }
    gen_castle(p, us, ml);
}
static int generateMoves_0x88(const Position& p, Move* out) {
    MoveList ml; generateMoves_0x88(p, ml);
    for (int i = 0; i < ml.count; ++i) out[i] = ml.m[i];
    return ml.count;
}
// -----------------------------------------------------------------------------
// build1.1-bitboard stage 7: pseudo-legal move generation.
// The old 0x88 generator remains available as generateMoves_0x88() for
// differential testing. The active generateMoves() path uses the bitboard
// mirror for source iteration, occupancy, targets, and castling rights.
// -----------------------------------------------------------------------------
static int bb_pop_lsb(Bitboard& bits) {
    const int sq = bb_ctzll(bits);
    bits &= bits - 1;
    return sq;
}

static void bb_add_step_moves(const Position& p, int sq64, int piece,
    MoveList& ml) {
    const int us = p.sideToMove;
    const Bitboard own = p.bbOccupied[us];
    const Bitboard enemy = p.bbOccupied[us ^ 1];
    Bitboard targets = (piece == KNIGHT)
        ? bb_knight_attack_mask(sq64)
        : bb_king_attack_mask(sq64);
    targets &= ~own;
    while (targets) {
        const int to64 = bb_pop_lsb(targets);
        const int flags = (enemy & bb_square(to64)) ? F_CAPTURE : 0;
        ml.add(sq88_from_bb_sq(sq64), sq88_from_bb_sq(to64), 0, flags);
    }
}

static void bb_add_slider_moves(const Position& p, int sq64, int piece,
    MoveList& ml) {
    const int us = p.sideToMove;
    const Bitboard own = p.bbOccupied[us];
    const Bitboard enemy = p.bbOccupied[us ^ 1];
    const Bitboard occupied = p.bbOccupiedAll;
    Bitboard targets = 0;
    if (piece == BISHOP) targets = bb_bishop_attack_mask(sq64, occupied);
    else if (piece == ROOK) targets = bb_rook_attack_mask(sq64, occupied);
    else {
        // Preserve the old generator's queen ordering: diagonals first,
        // orthogonals second.
        targets = bb_bishop_attack_mask(sq64, occupied);
        Bitboard rook_targets = bb_rook_attack_mask(sq64, occupied);
        targets &= ~own;
        while (targets) {
            const int to64 = bb_pop_lsb(targets);
            const int flags = (enemy & bb_square(to64)) ? F_CAPTURE : 0;
            ml.add(sq88_from_bb_sq(sq64), sq88_from_bb_sq(to64), 0, flags);
        }
        rook_targets &= ~own;
        while (rook_targets) {
            const int to64 = bb_pop_lsb(rook_targets);
            const int flags = (enemy & bb_square(to64)) ? F_CAPTURE : 0;
            ml.add(sq88_from_bb_sq(sq64), sq88_from_bb_sq(to64), 0, flags);
        }
        return;
    }
    targets &= ~own;
    while (targets) {
        const int to64 = bb_pop_lsb(targets);
        const int flags = (enemy & bb_square(to64)) ? F_CAPTURE : 0;
        ml.add(sq88_from_bb_sq(sq64), sq88_from_bb_sq(to64), 0, flags);
    }
}

static void bb_gen_pawns(const Position& p, MoveList& ml) {
    const int us = p.sideToMove;
    const int dir = us == WHITE ? 8 : -8;
    const int start_rank = us == WHITE ? 1 : 6;
    const int promo_rank = us == WHITE ? 7 : 0;
    const Bitboard occupied = p.bbOccupiedAll;
    const Bitboard enemy = p.bbOccupied[us ^ 1];
    const Bitboard enemy_pawns = p.bbPieces[us ^ 1][PAWN];
    Bitboard pawns = p.bbPieces[us][PAWN];
    while (pawns) {
        const int from64 = bb_pop_lsb(pawns);
        const int from88 = sq88_from_bb_sq(from64);
        const int from_rank = from64 >> 3;
        const int one64 = from64 + dir;
        if ((unsigned)one64 < 64 && !(occupied & bb_square(one64))) {
            const int one_rank = one64 >> 3;
            if (one_rank == promo_rank) {
                add_promos(ml, from88, sq88_from_bb_sq(one64), 0);
            }
            else {
                ml.add(from88, sq88_from_bb_sq(one64), 0, 0);
                const int two64 = from64 + 2 * dir;
                if (from_rank == start_rank && (unsigned)two64 < 64 &&
                    !(occupied & bb_square(two64))) {
                    ml.add(from88, sq88_from_bb_sq(two64), 0, F_DOUBLE);
                }
            }
        }

        Bitboard captures = bb_pawn_attack_mask(us, from64) & enemy;
        while (captures) {
            const int to64 = bb_pop_lsb(captures);
            const int to_rank = to64 >> 3;
            if (to_rank == promo_rank)
                add_promos(ml, from88, sq88_from_bb_sq(to64), F_CAPTURE);
            else
                ml.add(from88, sq88_from_bb_sq(to64), 0, F_CAPTURE);
        }

        if (p.ep != NO_SQ && on_board(p.ep)) {
            const int ep64 = bb_sq_from_sq88(p.ep);
            const bool rank_ok = us == WHITE ? ((ep64 >> 3) == 5) : ((ep64 >> 3) == 2);
            const int captured64 = ep64 + (us == WHITE ? -8 : 8);
            if (rank_ok && (unsigned)captured64 < 64 &&
                (enemy_pawns & bb_square(captured64)) &&
                (bb_pawn_attack_mask(us, from64) & bb_square(ep64)) &&
                !(occupied & bb_square(ep64))) {
                ml.add(from88, p.ep, 0, F_CAPTURE | F_EP);
            }
        }
    }
}

static void bb_gen_castle(const Position& p, MoveList& ml) {
    const int us = p.sideToMove;
    const int rank = us == WHITE ? 0 : 7;
    const int e = sq88(4, rank);
    const int king64 = rank * 8 + 4;
    if (!(p.bbPieces[us][KING] & bb_square(king64))) return;
    const Bitboard occupied = p.bbOccupiedAll;
    const Bitboard rook = p.bbPieces[us][ROOK];
    const int king_side = us == WHITE ? WK : BK;
    const int queen_side = us == WHITE ? WQ : BQ;
    const int rook_king64 = rank * 8 + 7;
    const int rook_queen64 = rank * 8;
    if ((p.castle & king_side) && (rook & bb_square(rook_king64)) &&
        !(occupied & (bb_square(rank * 8 + 5) | bb_square(rank * 8 + 6)))) {
        ml.add(e, sq88(6, rank), 0, F_CASTLE);
    }
    if ((p.castle & queen_side) && (rook & bb_square(rook_queen64)) &&
        !(occupied & (bb_square(rank * 8 + 1) | bb_square(rank * 8 + 2) |
            bb_square(rank * 8 + 3)))) {
        ml.add(e, sq88(2, rank), 0, F_CASTLE);
    }
}

static void generateMoves(const Position& p, MoveList& ml) {
    ml.clear();
    bb_init_attack_tables();
    bb_gen_pawns(p, ml);
    const int us = p.sideToMove;
    Bitboard pieces = p.bbPieces[us][KNIGHT];
    while (pieces) {
        const int from64 = bb_pop_lsb(pieces);
        bb_add_step_moves(p, from64, KNIGHT, ml);
    }
    pieces = p.bbPieces[us][BISHOP];
    while (pieces) {
        const int from64 = bb_pop_lsb(pieces);
        bb_add_slider_moves(p, from64, BISHOP, ml);
    }
    pieces = p.bbPieces[us][ROOK];
    while (pieces) {
        const int from64 = bb_pop_lsb(pieces);
        bb_add_slider_moves(p, from64, ROOK, ml);
    }
    pieces = p.bbPieces[us][QUEEN];
    while (pieces) {
        const int from64 = bb_pop_lsb(pieces);
        bb_add_slider_moves(p, from64, QUEEN, ml);
    }
    pieces = p.bbPieces[us][KING];
    while (pieces) {
        const int from64 = bb_pop_lsb(pieces);
        bb_add_step_moves(p, from64, KING, ml);
    }
    bb_gen_castle(p, ml);
}

static int generateMoves(const Position& p, Move* out) {
    MoveList ml;
    generateMoves(p, ml);
    for (int i = 0; i < ml.count; ++i) out[i] = ml.m[i];
    return ml.count;
}

// ------------------------------- MAKE / UNMAKE -------------------------------
struct Undo {
    int captured;
    int castle;
    int ep;
    int halfmove;
    int fullmove;
    int ksq;
    uint64_t key;
};
static int castle_mask(int sq) {
    int x = WK | WQ | BK | BQ;
    if (sq == sq88(4, 0) || sq == sq88(7, 0)) x &= ~WK;
    if (sq == sq88(4, 0) || sq == sq88(0, 0)) x &= ~WQ;
    if (sq == sq88(4, 7) || sq == sq88(7, 7)) x &= ~BK;
    if (sq == sq88(4, 7) || sq == sq88(0, 7)) x &= ~BQ;
    return x;
}
// =============================================================================
// REPETITION DETECTION — explicit, caller-owned stack (NOT global, NOT tied to
// makeMove/unmakeMove).
//
// Раньше это было глобальное состояние (position_history[]/history_ply),
// которое makeMove/unmakeMove трогали неявно. Из-за этого:
//   - perft/bench были вынуждены руками save/restore чужой глобал,
//   - несколько независимых поисков (будущий Lazy SMP / self-play) не могли
//     сосуществовать — все писали в один и тот же массив,
//   - при переполнении HISTORY_SIZE счётчик и реальный стек расходились.
//
// Теперь makeMove/unmakeMove абсолютно чистые (только board/key/castle/ep),
// а история повторов — обычный объект, которым явно владеет вызывающий код
// (UCI-слой хранит реальную партию, search() получает свою копию для дерева
// поиска).
// =============================================================================
struct RepetitionStack {
    static constexpr int CAPACITY = 1024;
    uint64_t keys[CAPACITY];
    int count = 0;
    void push(uint64_t k) { if (count < CAPACITY) keys[count++] = k; }
    void pop() { if (count > 0) --count; }
};
void makeMove(Position& p, const Move& m, Undo& u) {
    const int from = m.from, to = m.to, us = p.sideToMove;
    const int pc = bb_piece_ref(p, from);
    u.captured = EMPTY;
    u.castle = p.castle;
    u.ep = p.ep;
    u.halfmove = p.halfmove;
    u.fullmove = p.fullmove;
    u.ksq = p.ksq[us];
    u.key = p.key;
    // build1.1-bitboard stage 9: update the authoritative bitboards and the
    // compact 64-square piece lookup together.
    bb_remove_piece(p, pc, from);

    // build0.3: EP hashing is handled separately from the rest of the
    // incremental key update via hash_ep_square(), which normalizes away any
    // ep square that could not actually be captured. We first strip the OLD
    // (already-normalized) ep component from the key while `p` still fully
    // describes the pre-move position...
    const int old_ep_component = hash_ep_square(p);
    p.key ^= zEp[old_ep_component];
    p.key ^= zSide ^ zCastle[p.castle] ^ zPiece[pc][from];
    p.halfmove = (piece_of(pc) == PAWN || (m.flags & F_CAPTURE)) ? 0 : p.halfmove + 1;
    if (m.flags & F_EP) {
        int cs = to + (us == WHITE ? -16 : 16);
        u.captured = bb_piece_ref(p, cs);
        p.key ^= zPiece[u.captured][cs];
        bb_remove_piece(p, u.captured, cs);
        bb_piece_ref(p, cs) = EMPTY;
    }
    else if (m.flags & F_CAPTURE) {
        u.captured = bb_piece_ref(p, to);
        p.key ^= zPiece[u.captured][to];
        bb_remove_piece(p, u.captured, to);
    }
    bb_piece_ref(p, from) = EMPTY;
    bb_piece_ref(p, to) = (m.flags & F_PROMO) ? make_pc(m.promo, us) : pc;
    bb_add_piece(p, bb_piece_ref(p, to), to);
    if (piece_of(pc) == KING) p.ksq[us] = to;
    p.key ^= zPiece[bb_piece_ref(p, to)][to];
    if (m.flags & F_CASTLE) {
        int r = rank_of(to);
        int ks = file_of(to) == 6;
        int rf = sq88(ks ? 7 : 0, r), rt = sq88(ks ? 5 : 3, r);
        p.key ^= zPiece[bb_piece_ref(p, rf)][rf] ^ zPiece[bb_piece_ref(p, rf)][rt];
        bb_move_piece(p, bb_piece_ref(p, rf), rf, rt);
        bb_piece_ref(p, rt) = bb_piece_ref(p, rf);
        bb_piece_ref(p, rf) = EMPTY;
    }
    p.castle &= castle_mask(from) & castle_mask(to);
    p.key ^= zCastle[p.castle];
    p.ep = (m.flags & F_DOUBLE) ? (from + (us == WHITE ? 16 : -16)) : NO_SQ;
    if (us == BLACK) ++p.fullmove;
    p.sideToMove ^= 1;
    // ...then, once the position (board + side + raw ep) fully describes the
    // POST-move position, add back the NEW normalized ep component. This
    // keeps p.key == computeHash(p) after every makeMove() call.
    const int new_ep_component = hash_ep_square(p);
    p.key ^= zEp[new_ep_component];
}
void unmakeMove(Position& p, const Move& m, const Undo& u) {
    p.sideToMove ^= 1;
    p.key = u.key;
    const int from = m.from, to = m.to, us = p.sideToMove;
    p.castle = u.castle;
    p.ep = u.ep;
    p.halfmove = u.halfmove;
    p.fullmove = u.fullmove;
    if (m.flags & F_CASTLE) {
        int r = rank_of(to);
        int ks = file_of(to) == 6;
        int rf = sq88(ks ? 7 : 0, r), rt = sq88(ks ? 5 : 3, r);
        bb_move_piece(p, bb_piece_ref(p, rt), rt, rf);
        bb_piece_ref(p, rf) = bb_piece_ref(p, rt);
        bb_piece_ref(p, rt) = EMPTY;
    }
    const int moved_pc = bb_piece_ref(p, to);
    bb_remove_piece(p, moved_pc, to);
    bb_piece_ref(p, from) = (m.flags & F_PROMO) ? make_pc(PAWN, us) : moved_pc;
    bb_add_piece(p, bb_piece_ref(p, from), from);
    if (m.flags & F_EP) {
        bb_piece_ref(p, to) = EMPTY;
        bb_piece_ref(p, to + (us == WHITE ? -16 : 16)) = u.captured;
        bb_add_piece(p, u.captured, to + (us == WHITE ? -16 : 16));
    }
    else {
        bb_piece_ref(p, to) = u.captured;
        bb_add_piece(p, u.captured, to);
    }
    p.ksq[us] = u.ksq;
}
// -------------------------- CHECK / LEGAL MOVES -----------------------------
bool isSquareAttacked_0x88(const Position& p, int sq, int byColor) {
    int pdir = (byColor == WHITE) ? -16 : 16;
    int pd[2] = { pdir - 1, pdir + 1 };
    for (int i = 0; i < 2; ++i) {
        int s = sq + pd[i];
        if (on_board(s) && bb_piece_ref(p, s) == make_pc(PAWN, byColor)) return true;
    }
    for (int d : KNIGHT_DIRS) {
        int s = sq + d;
        if (on_board(s) && bb_piece_ref(p, s) == make_pc(KNIGHT, byColor)) return true;
    }
    for (int d : KING_DIRS) {
        int s = sq + d;
        if (on_board(s) && bb_piece_ref(p, s) == make_pc(KING, byColor)) return true;
    }
    for (int d : BISHOP_DIRS) {
        for (int s = sq + d; on_board(s); s += d) {
            int pc = bb_piece_ref(p, s);
            if (!pc) continue;
            if (color_of(pc) == byColor && (piece_of(pc) == BISHOP || piece_of(pc) == QUEEN)) return true;
            break;
        }
    }
    for (int d : ROOK_DIRS) {
        for (int s = sq + d; on_board(s); s += d) {
            int pc = bb_piece_ref(p, s);
            if (!pc) continue;
            if (color_of(pc) == byColor && (piece_of(pc) == ROOK || piece_of(pc) == QUEEN)) return true;
            break;
        }
    }
    return false;
}
static int bb_first_candidate_on_dirs(Bitboard candidates, int target64,
    const int* dirs, int count) {
    const int target88 = sq88_from_bb_sq(target64);
    for (int i = 0; i < count; ++i) {
        for (int sq = target88 + dirs[i]; on_board(sq); sq += dirs[i]) {
            const Bitboard bit = bb_square(bb_sq_from_sq88(sq));
            if (candidates & bit) return sq;
        }
    }
    return NO_SQ;
}

static int bb_first_pawn_attacker(Bitboard candidates, int target64, int side) {
    const int target88 = sq88_from_bb_sq(target64);
    const int fwd = side == WHITE ? 16 : -16;
    for (int df : { -1, 1 }) {
        const int sq = target88 - fwd + df;
        if (on_board(sq) && (candidates & bb_square(bb_sq_from_sq88(sq))))
            return sq;
    }
    return NO_SQ;
}

static bool isSquareAttacked_bitboard(const Position& p, int sq, int byColor) {
    if (!on_board(sq)) return false;
    const int target64 = bb_sq_from_sq88(sq);
    const Bitboard target = bb_square(target64);
    const Bitboard removed = 0;
    const Bitboard live = ~removed;

    Bitboard candidates = p.bbPieces[byColor][PAWN] & live;
    if (bb_first_pawn_attacker(candidates, target64, byColor) != NO_SQ)
        return true;

    candidates = p.bbPieces[byColor][KNIGHT] & live &
        bb_knight_attack_mask(target64);
    if (candidates) return true;
    candidates = p.bbPieces[byColor][KING] & live &
        bb_king_attack_mask(target64);
    if (candidates) return true;

    const Bitboard occupied = p.bbOccupiedAll & live;
    const Bitboard bishop_rays = bb_bishop_attack_mask(target64, occupied);
    const Bitboard rook_rays = bb_rook_attack_mask(target64, occupied);
    if (bishop_rays & (p.bbPieces[byColor][BISHOP] | p.bbPieces[byColor][QUEEN]))
        return true;
    if (rook_rays & (p.bbPieces[byColor][ROOK] | p.bbPieces[byColor][QUEEN]))
        return true;
    return false;
}

bool isSquareAttacked(const Position& p, int sq, int byColor) {
    return isSquareAttacked_bitboard(p, sq, byColor);
}

int findKing(const Position& p, int color) {
    int target = make_pc(KING, color);
    for (int sq = 0; sq < 128; ++sq)
        if (on_board(sq) && bb_piece_ref(p, sq) == target) return sq;
    return NO_SQ;
}
bool inCheck(const Position& p, int color) {
    const int ksq = p.ksq[color];
    return ksq != NO_SQ && isSquareAttacked(p, ksq, color ^ 1);
}

static bool inCheck_0x88(const Position& p, int color) {
    const int ksq = p.ksq[color];
    return ksq != NO_SQ && isSquareAttacked_0x88(p, ksq, color ^ 1);
}

// Preserved legal reference: old pseudo generator + old attack detector.
static void generateLegalMoves_0x88(const Position& p, MoveList& ml) {
    MoveList pseudo;
    generateMoves_0x88(p, pseudo);
    ml.clear();
    Position pos = p;
    const int us = pos.sideToMove;
    for (int i = 0; i < pseudo.count; ++i) {
        const Move& m = pseudo.m[i];
        if (m.flags & F_CASTLE) {
            if (inCheck_0x88(pos, us)) continue;
            if (isSquareAttacked_0x88(pos, (m.from + m.to) / 2, us ^ 1)) continue;
            if (isSquareAttacked_0x88(pos, m.to, us ^ 1)) continue;
        }
        Undo u;
        makeMove(pos, m, u);
        if (!inCheck_0x88(pos, us)) ml.add(m.from, m.to, m.promo, m.flags);
        unmakeMove(pos, m, u);
    }
}

// Active legal filtering: bitboard pseudo moves, bitboard attack tests, and
// the existing make/unmake state transition. Pins and discovered checks are
// rejected by testing the mover's king after every candidate move. Castling
// checks the origin, transit, and destination squares before making the move.
void generateLegalMoves(const Position& p, MoveList& ml) {
    MoveList pseudo;
    generateMoves(p, pseudo);
    ml.clear();
    Position pos = p;
    const int us = pos.sideToMove;
    for (int i = 0; i < pseudo.count; ++i) {
        const Move& m = pseudo.m[i];
        if (m.flags & F_CASTLE) {
            if (inCheck(pos, us)) continue;
            if (isSquareAttacked(pos, (m.from + m.to) / 2, us ^ 1)) continue;
            if (isSquareAttacked(pos, m.to, us ^ 1)) continue;
        }
        Undo u;
        makeMove(pos, m, u);
        if (!inCheck(pos, us)) ml.add(m.from, m.to, m.promo, m.flags);
        unmakeMove(pos, m, u);
    }
}
int generateLegalMoves(const Position& p, Move* out) {
    MoveList ml;
    generateLegalMoves(p, ml);
    if (out) for (int i = 0; i < ml.count; ++i) out[i] = ml.m[i];
    return ml.count;
}
// -------------------------------- SELF-TESTS ---------------------------------
static int g_pass = 0, g_fail = 0;
static void check(bool ok, const char* name) {
    if (ok) ++g_pass; else { ++g_fail; std::printf("FAIL: %s\n", name); }
}
static int sq_of(const char* s) { return sq88(s[0] - 'a', s[1] - '1'); }
static void move_str(const Move& m, char* b) {
    b[0] = (char)('a' + file_of(m.from));
    b[1] = (char)('1' + rank_of(m.from));
    b[2] = (char)('a' + file_of(m.to));
    b[3] = (char)('1' + rank_of(m.to));
    int n = 4;
    if (m.flags & F_PROMO) {
        switch (m.promo) {
        case KNIGHT:
            b[n++] = 'n';
            break;
        case BISHOP:
            b[n++] = 'b';
            break;
        case ROOK:
            b[n++] = 'r';
            break;
        case QUEEN:
            b[n++] = 'q';
            break;
        }
    }
    b[n] = 0;
}
// ------------------------------- PERFT TESTER --------------------------------
// Perft считает количество легальных позиций на глубине depth.
// Это НЕ поиск и НЕ оценка. Нужен только для проверки генератора ходов,
// make/unmake, рокировки, en-passant и превращений.
//
// perft больше НЕ трогает никакой глобал: makeMove/unmakeMove чистые.
static uint64_t perft(Position& pos, int depth) {
    if (depth <= 0)
        return 1ULL;
    MoveList ml;
    generateLegalMoves(pos, ml);
    if (depth == 1)
        return (uint64_t)ml.count;
    uint64_t nodes = 0;
    for (int i = 0; i < ml.count; ++i) {
        Undo u;
        makeMove(pos, ml.m[i], u);
        nodes += perft(pos, depth - 1);
        unmakeMove(pos, ml.m[i], u);
    }
    return nodes;
}
// Divide показывает вклад каждого хода первого слоя.
// Очень удобно, когда общий perft не совпал: видно, на каком ходе ошибка.
static uint64_t perft_divide(Position& pos, int depth) {
    if (depth <= 0) {
        std::printf("info string nodes 1\n");
        std::fflush(stdout);
        return 1ULL;
    }
    MoveList ml;
    generateLegalMoves(pos, ml);
    uint64_t total_nodes = 0;
    for (int i = 0; i < ml.count; ++i) {
        Undo u;
        uint64_t child_nodes;
        if (depth == 1) {
            child_nodes = 1ULL;
        }
        else {
            makeMove(pos, ml.m[i], u);
            child_nodes = perft(pos, depth - 1);
            unmakeMove(pos, ml.m[i], u);
        }
        char buf[8];
        move_str(ml.m[i], buf);
        std::printf("info string %s %llu\n", buf, (unsigned long long)child_nodes);
        total_nodes += child_nodes;
    }
    std::printf("info string total %llu\n", (unsigned long long)total_nodes);
    std::fflush(stdout);
    return total_nodes;
}
static bool perft_check(const char* name, const char* fen, int depth, uint64_t expected) {
    Position p;
    if (!parse_fen(p, fen)) {
        std::printf("info string PERFT FAIL %s depth %d bad fen\n", name, depth);
        std::fflush(stdout);
        return false;
    }
    uint64_t got = perft(p, depth);
    bool ok = (got == expected);
    std::printf("info string PERFT %s depth %d nodes %llu expected %llu %s\n",
        name,
        depth,
        (unsigned long long)got,
        (unsigned long long)expected,
        ok ? "OK" : "FAIL");
    std::fflush(stdout);
    return ok;
}
static void run_perft_tests() {
    int pass = 0;
    int fail = 0;
    const char* STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    const char* KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    auto one = [&](const char* name, const char* fen, int depth, uint64_t expected) {
        if (perft_check(name, fen, depth, expected)) ++pass;
        else ++fail;
        };
    one("startpos", STARTPOS, 1, 20ULL);
    one("startpos", STARTPOS, 2, 400ULL);
    one("startpos", STARTPOS, 3, 8902ULL);
    one("startpos", STARTPOS, 4, 197281ULL);
    one("startpos", STARTPOS, 5, 4865609ULL);
    // Kiwipete: классическая позиция для проверки рокировок, взятий и сложных ходов.
    one("kiwipete", KIWIPETE, 1, 48ULL);
    one("kiwipete", KIWIPETE, 2, 2039ULL);
    one("kiwipete", KIWIPETE, 3, 97862ULL);
    one("kiwipete", KIWIPETE, 4, 4085603ULL);
    std::printf("info string PERFT TESTS DONE pass %d fail %d\n", pass, fail);
    std::fflush(stdout);
}
static int bb_move_key(const Move& m) {
    return int(m.from) | (int(m.to) << 8) |
        (int(m.promo) << 16) | (int(m.flags) << 20);
}

static bool bb_move_lists_equal(const MoveList& old_list, const MoveList& new_list) {
    if (old_list.count != new_list.count) return false;
    std::vector<int> old_keys, new_keys;
    old_keys.reserve(old_list.count);
    new_keys.reserve(new_list.count);
    for (int i = 0; i < old_list.count; ++i)
        old_keys.push_back(bb_move_key(old_list.m[i]));
    for (int i = 0; i < new_list.count; ++i)
        new_keys.push_back(bb_move_key(new_list.m[i]));
    std::sort(old_keys.begin(), old_keys.end());
    std::sort(new_keys.begin(), new_keys.end());
    return old_keys == new_keys;
}

static bool bb_position_core_equal(const Position& a, const Position& b) {
    return std::memcmp(a.piece, b.piece, sizeof a.piece) == 0 &&
        a.sideToMove == b.sideToMove && a.castle == b.castle && a.ep == b.ep &&
        a.halfmove == b.halfmove && a.fullmove == b.fullmove &&
        a.ksq[WHITE] == b.ksq[WHITE] && a.ksq[BLACK] == b.ksq[BLACK] &&
        a.key == b.key &&
        std::memcmp(a.bbPieces, b.bbPieces, sizeof a.bbPieces) == 0 &&
        std::memcmp(a.bbOccupied, b.bbOccupied, sizeof a.bbOccupied) == 0 &&
        a.bbOccupiedAll == b.bbOccupiedAll;
}

static bool bb_walk_make_unmake(Position& p, int depth) {
    if (!bb_state_matches(p)) return false;
    if (depth == 0) return true;
    MoveList ml;
    generateLegalMoves(p, ml);
    for (int i = 0; i < ml.count; ++i) {
        Position before = p;
        Undo u;
        makeMove(p, ml.m[i], u);
        if (!bb_walk_make_unmake(p, depth - 1)) return false;
        unmakeMove(p, ml.m[i], u);
        if (!bb_position_core_equal(p, before)) return false;
    }
    return true;
}

static bool bb_magic_exhaustive_check() {
    if (!bb_init_magic_tables()) return false;
    for (int kind = 0; kind < 2; ++kind) {
        const bool bishop = kind == 0;
        const BBMagicEntry* info = bishop
            ? bb_bishop_magic_info : bb_rook_magic_info;
        for (int sq64 = 0; sq64 < 64; ++sq64) {
            const Bitboard mask = info[sq64].mask;
            const uint32_t variations = 1u << bb_popcountll(mask);
            for (uint32_t i = 0; i < variations; ++i) {
                const Bitboard occupied =
                    bb_magic_subset_from_index(i, mask);
                const Bitboard expected =
                    bb_sliding_attacks_reference(sq64, occupied, bishop);
                const Bitboard actual = bb_magic_lookup(info[sq64], occupied);
                if (expected != actual) return false;
            }
        }
    }
    return true;
}

static void run_bitboard_tests() {
    int pass = 0, fail = 0;
    auto result = [&](bool ok, const char* name) {
        if (ok) { ++pass; std::printf("info string BB %s OK\n", name); }
        else { ++fail; std::printf("info string BB %s FAIL\n", name); }
        };

    bool mapping = true;
    for (int sq64 = 0; sq64 < 64; ++sq64) {
        int sq88 = sq88_from_bb_sq(sq64);
        if (!on_board(sq88) || bb_sq_from_sq88(sq88) != sq64) mapping = false;
    }
    result(mapping, "square_mapping");

    bb_init_attack_tables();
    result(bb_magic_exhaustive_check(), "magic_sliding_tables");
    bool attack_tables = true;
    for (int sq64 = 0; sq64 < 64; ++sq64) {
        if (bb_pawn_attack_mask(WHITE, sq64) !=
            bb_reference_pawn_attacks(WHITE, sq64) ||
            bb_pawn_attack_mask(BLACK, sq64) !=
            bb_reference_pawn_attacks(BLACK, sq64) ||
            bb_knight_attack_mask(sq64) !=
            bb_reference_step_attacks(sq64, KNIGHT_DIRS, 8) ||
            bb_king_attack_mask(sq64) !=
            bb_reference_step_attacks(sq64, KING_DIRS, 8)) {
            attack_tables = false;
            break;
        }
    }
    // Differentially compare every non-sliding table entry with the existing
    // 0x88 isSquareAttacked() implementation using isolated single-attacker
    // positions. This is deliberately independent of the table constructor.
    for (int color = WHITE; color <= BLACK && attack_tables; ++color) {
        for (int type : { PAWN, KNIGHT, KING }) {
            for (int from64 = 0; from64 < 64 && attack_tables; ++from64) {
                for (int target64 = 0; target64 < 64; ++target64) {
                    Position q;
                    clear_pos(q);
                    bb_piece_ref(q, sq88_from_bb_sq(from64)) = make_pc(type, color);
                    const int target88 = sq88_from_bb_sq(target64);
                    const bool old_result = isSquareAttacked_0x88(q, target88, color);
                    const Bitboard target = bb_square(target64);
                    const Bitboard mask = type == PAWN
                        ? bb_pawn_attack_mask(color, from64)
                        : type == KNIGHT
                        ? bb_knight_attack_mask(from64)
                        : bb_king_attack_mask(from64);
                    if (old_result != ((mask & target) != 0)) {
                        attack_tables = false;
                        break;
                    }
                }
            }
        }
    }
    result(attack_tables, "non_sliding_attack_tables");

    // Differential sliding-attack test. For every source square and many
    // deterministic random occupancies, build an isolated 0x88 position with
    // one white slider and compare every target square against the bitboard
    // ray implementation. Black pawns are blockers only; they are never
    // counted as white attackers by isSquareAttacked().
    bool sliding_attacks = true;
    uint64_t random_state = 0x6a09e667f3bcc909ULL;
    for (int kind = 0; kind < 3 && sliding_attacks; ++kind) {
        const int piece = kind == 0 ? BISHOP : kind == 1 ? ROOK : QUEEN;
        for (int from64 = 0; from64 < 64 && sliding_attacks; ++from64) {
            for (int sample = 0; sample < 96 && sliding_attacks; ++sample) {
                Bitboard occupied = splitmix64(random_state);
                occupied &= ~bb_square(from64);
                Position q;
                clear_pos(q);
                const int from88 = sq88_from_bb_sq(from64);
                bb_piece_ref(q, from88) = make_pc(piece, WHITE);
                for (int sq64 = 0; sq64 < 64; ++sq64) {
                    if (occupied & bb_square(sq64))
                        bb_piece_ref(q, sq88_from_bb_sq(sq64)) = make_pc(PAWN, BLACK);
                }
                const Bitboard mask = kind == 0
                    ? bb_bishop_attack_mask(from64, occupied)
                    : kind == 1
                    ? bb_rook_attack_mask(from64, occupied)
                    : bb_queen_attack_mask(from64, occupied);
                for (int target64 = 0; target64 < 64; ++target64) {
                    const bool old_result = isSquareAttacked_0x88(
                        q, sq88_from_bb_sq(target64), WHITE);
                    const bool bb_result = (mask & bb_square(target64)) != 0;
                    if (old_result != bb_result) {
                        sliding_attacks = false;
                        break;
                    }
                }
            }
        }
    }
    result(sliding_attacks, "sliding_attack_reference");

    bool attacked_square = true;
    int attacked_compared = 0;
    const char* attack_fens[] = {
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/8/3p4/2b1r3/3Q4/2N5/4K3/8 w - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    };
    for (const char* fen : attack_fens) {
        Position q;
        if (!parse_fen(q, fen)) { attacked_square = false; break; }
        for (int color = WHITE; color <= BLACK && attacked_square; ++color) {
            for (int target64 = 0; target64 < 64; ++target64) {
                const int target88 = sq88_from_bb_sq(target64);
                const bool old_result = isSquareAttacked_0x88(q, target88, color);
                const bool new_result = isSquareAttacked(q, target88, color);
                ++attacked_compared;
                if (old_result != new_result) {
                    attacked_square = false;
                    break;
                }
            }
        }
        if (!attacked_square) break;
    }
    result(attacked_square && attacked_compared > 0, "attacked_square_differential");

    // Stage 7 differential test: compare the complete pseudo-legal move set
    // from the old 0x88 generator with the active bitboard generator. Flags
    // are part of the key, so captures, double pushes, promotions, castling,
    // and en passant cannot silently compare equal as plain from/to moves.
    bool pseudo_differential = true;
    int pseudo_positions = 0;
    const char* pseudo_fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/P6p/1p6/3pPp2/8/8/p6P/4K2k w - f6 0 1",
        "7k/4P3/8/8/8/8/4p3/K7 b - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "8/8/8/3pP3/8/8/8/4K2k w - d6 0 1",
    };
    for (const char* fen : pseudo_fens) {
        Position q;
        if (!parse_fen(q, fen)) { pseudo_differential = false; break; }
        MoveList old_list, new_list;
        generateMoves_0x88(q, old_list);
        generateMoves(q, new_list);
        ++pseudo_positions;
        if (!bb_move_lists_equal(old_list, new_list)) {
            pseudo_differential = false;
            break;
        }
    }
    // Continue through a deterministic legal walk to cover positions that are
    // not represented by the fixed FENs above.
    if (pseudo_differential) {
        Position q;
        set_start(q);
        uint64_t walk_state = 0xbb67ae8584caa73bULL;
        for (int ply = 0; ply < 120 && pseudo_differential; ++ply) {
            MoveList old_list, new_list;
            generateMoves_0x88(q, old_list);
            generateMoves(q, new_list);
            ++pseudo_positions;
            if (!bb_move_lists_equal(old_list, new_list)) {
                pseudo_differential = false;
                break;
            }
            MoveList legal;
            generateLegalMoves(q, legal);
            if (legal.count == 0) break;
            const int pick = int(splitmix64(walk_state) % (uint64_t)legal.count);
            Undo u;
            makeMove(q, legal.m[pick], u);
        }
    }
    result(pseudo_differential, "pseudo_move_differential");
    std::printf("info string BB pseudo_positions_compared %d\n", pseudo_positions);

    // Stage 8 differential test: active legal filtering must match the old
    // 0x88 legal path, including pins, checks, castling transit squares,
    // en-passant discovered checks, and promotions.
    bool legal_differential = true;
    int legal_positions = 0;
    const char* legal_fens[] = {
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/8/8/r2pP2K/8/8/8/4k3 w - d6 0 1",
        "8/8/8/3pP3/8/8/8/4K2k w - d6 0 1",
        "7k/P7/8/8/8/8/8/4K3 w - - 0 1",
        "4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1",
        "4r1k1/8/8/8/8/8/4K3/8 w - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    };
    for (const char* fen : legal_fens) {
        Position q;
        if (!parse_fen(q, fen)) { legal_differential = false; break; }
        MoveList old_list, new_list;
        generateLegalMoves_0x88(q, old_list);
        generateLegalMoves(q, new_list);
        ++legal_positions;
        if (!bb_move_lists_equal(old_list, new_list)) {
            legal_differential = false;
            break;
        }
    }
    if (legal_differential) {
        Position q;
        set_start(q);
        uint64_t legal_walk_state = 0x3c6ef372fe94f82bULL;
        for (int ply = 0; ply < 120 && legal_differential; ++ply) {
            MoveList old_list, new_list;
            generateLegalMoves_0x88(q, old_list);
            generateLegalMoves(q, new_list);
            ++legal_positions;
            if (!bb_move_lists_equal(old_list, new_list)) {
                legal_differential = false;
                break;
            }
            if (new_list.count == 0) break;
            const int pick = int(splitmix64(legal_walk_state) %
                (uint64_t)new_list.count);
            Undo u;
            makeMove(q, new_list.m[pick], u);
        }
    }
    result(legal_differential, "legal_move_differential");
    std::printf("info string BB legal_positions_compared %d\n", legal_positions);

    const char* fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/P6p/1p6/3pPp2/8/8/p6P/4K2k w - f6 0 1",
        "7k/4P3/8/8/8/8/4p3/K7 b - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    };
    bool rebuilt = true;
    for (const char* fen : fens) {
        Position p;
        if (!parse_fen(p, fen) || !bb_state_matches(p)) rebuilt = false;
    }
    result(rebuilt, "fen_rebuild");

    bool incremental = true;
    for (const char* fen : fens) {
        Position p;
        if (!parse_fen(p, fen) || !bb_walk_make_unmake(p, 2)) {
            incremental = false;
            break;
        }
    }
    result(incremental, "incremental_make_unmake");
    std::printf("info string BB TESTS DONE pass %d fail %d\n", pass, fail);
    std::fflush(stdout);
}

static int gen_fen(const char* fen, MoveList& ml) {
    Position p;
    if (!parse_fen(p, fen)) { ml.clear(); return -1; }
    generateMoves(p, ml);
    return ml.count;
}
static int count_from(const char* fen, const char* sq) {
    MoveList ml; gen_fen(fen, ml);
    int s = sq_of(sq), n = 0;
    for (int i = 0; i < ml.count; ++i) if (ml.m[i].from == s) ++n;
    return n;
}
static int find_move(const char* fen, const char* mv) {
    MoveList ml; gen_fen(fen, ml); char b[8];
    for (int i = 0; i < ml.count; ++i) { move_str(ml.m[i], b); if (!std::strcmp(b, mv)) return 0x100 | ml.m[i].flags; }
    return 0;
}
static bool has(const char* fen, const char* mv) { return find_move(fen, mv) != 0; }
static int total(const char* fen) { MoveList ml; return gen_fen(fen, ml); }
static bool peq(const Position& a, const Position& b) {
    return !std::memcmp(a.piece, b.piece, sizeof a.piece) && a.sideToMove == b.sideToMove
        && a.castle == b.castle && a.ep == b.ep && a.halfmove == b.halfmove && a.fullmove == b.fullmove
        && a.ksq[WHITE] == b.ksq[WHITE] && a.ksq[BLACK] == b.ksq[BLACK];
}
static bool getmv(const Position& p, const char* s, Move& m) {
    MoveList ml; generateMoves(p, ml); char b[8];
    for (int i = 0; i < ml.count; ++i) { move_str(ml.m[i], b); if (!std::strcmp(b, s)) { m = ml.m[i]; return true; } }
    return false;
}
static void rt(const char* fen, const char* s, const char* n) {
    Position p, o; Move m{}; Undo u;
    bool ok = parse_fen(p, fen) && getmv(p, s, m);
    check(ok, n);
    if (!ok) return;
    o = p; makeMove(p, m, u); unmakeMove(p, m, u);
    check(peq(p, o), n);
}
static void test_mu() {
    Position p, o; Move m, m1, m2; Undo u, u1, u2;
    set_start(p); o = p; getmv(p, "e2e4", m); makeMove(p, m, u);
    check(bb_piece_ref(p, sq_of("e4")) == make_pc(PAWN, WHITE) && bb_piece_ref(p, sq_of("e2")) == EMPTY, "e2e4 pcs");
    check(p.ep == sq_of("e3") && p.sideToMove == BLACK && p.halfmove == 0 && p.fullmove == 1, "e2e4 st");
    unmakeMove(p, m, u); check(peq(p, o), "e2e4 rt");
    parse_fen(p, "8/8/8/4p3/3P4/8/8/8 w - - 5 3"); o = p;
    getmv(p, "d4e5", m); makeMove(p, m, u);
    check(bb_piece_ref(p, sq_of("e5")) == make_pc(PAWN, WHITE) && bb_piece_ref(p, sq_of("d4")) == EMPTY && p.halfmove == 0, "cap");
    unmakeMove(p, m, u); check(peq(p, o), "cap rt");
    parse_fen(p, "8/4P3/8/8/8/8/8/8 w - - 0 1"); o = p;
    getmv(p, "e7e8q", m); makeMove(p, m, u);
    check(bb_piece_ref(p, sq_of("e8")) == make_pc(QUEEN, WHITE) && bb_piece_ref(p, sq_of("e7")) == EMPTY, "promo");
    unmakeMove(p, m, u); check(peq(p, o), "promo rt");
    parse_fen(p, "5r2/4P3/8/8/8/8/8/8 w - - 0 1"); o = p;
    getmv(p, "e7f8q", m); makeMove(p, m, u);
    check(bb_piece_ref(p, sq_of("f8")) == make_pc(QUEEN, WHITE), "cpromo");
    unmakeMove(p, m, u); check(peq(p, o) && bb_piece_ref(p, sq_of("f8")) == make_pc(ROOK, BLACK), "cpromo rt");
    parse_fen(p, "8/8/8/3pP3/8/8/8/8 w - d6 0 1"); o = p;
    getmv(p, "e5d6", m); makeMove(p, m, u);
    check(bb_piece_ref(p, sq_of("d6")) == make_pc(PAWN, WHITE) && bb_piece_ref(p, sq_of("d5")) == EMPTY
        && bb_piece_ref(p, sq_of("e5")) == EMPTY && p.ep == NO_SQ, "ep");
    unmakeMove(p, m, u); check(peq(p, o) && bb_piece_ref(p, sq_of("d5")) == make_pc(PAWN, BLACK), "ep rt");
    parse_fen(p, "rnbqkbnr/ppp1p1pp/8/8/3pPp2/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"); o = p;
    getmv(p, "d4e3", m); makeMove(p, m, u);
    check(bb_piece_ref(p, sq_of("e3")) == make_pc(PAWN, BLACK) && bb_piece_ref(p, sq_of("e4")) == EMPTY
        && bb_piece_ref(p, sq_of("d4")) == EMPTY, "epb");
    unmakeMove(p, m, u); check(peq(p, o), "epb rt");
    parse_fen(p, "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"); o = p;
    getmv(p, "e1g1", m); makeMove(p, m, u);
    check(bb_piece_ref(p, sq_of("g1")) == make_pc(KING, WHITE) && bb_piece_ref(p, sq_of("f1")) == make_pc(ROOK, WHITE)
        && bb_piece_ref(p, sq_of("e1")) == EMPTY && bb_piece_ref(p, sq_of("h1")) == EMPTY && !(p.castle & (WK | WQ)), "WOO");
    unmakeMove(p, m, u); check(peq(p, o), "WOO rt");
    parse_fen(p, "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"); o = p;
    getmv(p, "e1c1", m); makeMove(p, m, u);
    check(bb_piece_ref(p, sq_of("c1")) == make_pc(KING, WHITE) && bb_piece_ref(p, sq_of("d1")) == make_pc(ROOK, WHITE)
        && bb_piece_ref(p, sq_of("a1")) == EMPTY && bb_piece_ref(p, sq_of("e1")) == EMPTY, "WOOO");
    unmakeMove(p, m, u); check(peq(p, o), "WOOO rt");
    parse_fen(p, "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"); o = p;
    getmv(p, "e8g8", m); makeMove(p, m, u);
    check(bb_piece_ref(p, sq_of("g8")) == make_pc(KING, BLACK) && bb_piece_ref(p, sq_of("f8")) == make_pc(ROOK, BLACK)
        && bb_piece_ref(p, sq_of("e8")) == EMPTY && bb_piece_ref(p, sq_of("h8")) == EMPTY, "BOO");
    unmakeMove(p, m, u); check(peq(p, o), "BOO rt");
    parse_fen(p, "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"); o = p;
    getmv(p, "e8c8", m); makeMove(p, m, u);
    check(bb_piece_ref(p, sq_of("c8")) == make_pc(KING, BLACK) && bb_piece_ref(p, sq_of("d8")) == make_pc(ROOK, BLACK)
        && bb_piece_ref(p, sq_of("a8")) == EMPTY && bb_piece_ref(p, sq_of("e8")) == EMPTY, "BOOO");
    unmakeMove(p, m, u); check(peq(p, o), "BOOO rt");
    parse_fen(p, "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    getmv(p, "e1e2", m); makeMove(p, m, u);
    check(!(p.castle & (WK | WQ)) && (p.castle & (BK | BQ)) == (BK | BQ), "king rights");
    parse_fen(p, "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    getmv(p, "a1a2", m); makeMove(p, m, u);
    check(!(p.castle & WQ) && (p.castle & WK), "rook a1");
    parse_fen(p, "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    getmv(p, "h1h2", m); makeMove(p, m, u);
    check(!(p.castle & WK) && (p.castle & WQ), "rook h1");
    parse_fen(p, "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
    getmv(p, "a8a7", m); makeMove(p, m, u);
    check(!(p.castle & BQ) && (p.castle & BK), "rook a8");
    parse_fen(p, "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
    getmv(p, "h8h7", m); makeMove(p, m, u);
    check(!(p.castle & BK) && (p.castle & BQ), "rook h8");
    parse_fen(p, "8/8/8/8/8/8/r7/R3K3 b Q - 0 1");
    getmv(p, "a2a1", m); makeMove(p, m, u);
    check(p.castle == 0, "cap a1");
    parse_fen(p, "8/8/8/8/8/8/7r/4K2R b K - 0 1");
    getmv(p, "h2h1", m); makeMove(p, m, u);
    check(p.castle == 0, "cap h1");
    parse_fen(p, "r3k3/8/8/8/8/8/8/R7 w q - 0 1");
    getmv(p, "a1a8", m); makeMove(p, m, u);
    check(p.castle == 0, "cap a8");
    set_start(p); getmv(p, "g1f3", m); makeMove(p, m, u);
    check(p.halfmove == 1 && p.fullmove == 1 && p.sideToMove == BLACK && p.ep == NO_SQ, "nf3 clk");
    getmv(p, "b8c6", m); makeMove(p, m, u2);
    check(p.halfmove == 2 && p.fullmove == 2 && p.sideToMove == WHITE, "nc6 clk");
    unmakeMove(p, m, u2);
    check(p.halfmove == 1 && p.fullmove == 1 && p.sideToMove == BLACK, "clk unmake");
    set_start(p); o = p;
    getmv(p, "e2e4", m1); makeMove(p, m1, u1);
    getmv(p, "e7e5", m2); makeMove(p, m2, u2);
    unmakeMove(p, m2, u2); unmakeMove(p, m1, u1);
    check(peq(p, o), "2ply rt");
    rt("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "b1c3", "rt Nc3");
    rt("rnbqkbnr/ppp1p1pp/8/8/3pPp2/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "f4e3", "rt epb2");
    rt("8/8/8/8/8/8/4p3/8 b - - 0 1", "e2e1q", "rt bpr");
    rt("5r2/4P3/8/8/8/8/8/8 w - - 0 1", "e7f8n", "rt cpn");
    rt("8/8/8/8/8/8/4p3/5R2 b - - 0 1", "e2f1q", "rt bcp");
    rt("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", "rt WOO2");
    rt("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8", "rt BOOO2");
    rt("8/8/8/3pP3/8/8/8/8 w - d6 0 1", "e5d6", "rt ep2");
}
static int legal_total(const char* fen) {
    Position p; if (!parse_fen(p, fen)) return -1;
    MoveList ml; generateLegalMoves(p, ml); return ml.count;
}
static bool has_legal(const char* fen, const char* mv) {
    Position p; if (!parse_fen(p, fen)) return false;
    MoveList ml; generateLegalMoves(p, ml); char b[8];
    for (int i = 0; i < ml.count; ++i) { move_str(ml.m[i], b); if (!std::strcmp(b, mv)) return true; }
    return false;
}
static void test_legal() {
    Position p;
    parse_fen(p, "4k3/8/8/8/8/8/8/4R2K b - - 0 1");
    check(inCheck(p, BLACK) && !inCheck(p, WHITE), "check rook");
    parse_fen(p, "4k3/8/8/1B6/8/8/8/4K3 b - - 0 1");
    check(inCheck(p, BLACK), "check bishop");
    parse_fen(p, "4k3/3N4/8/8/8/8/8/4K3 b - - 0 1");
    check(inCheck(p, BLACK), "check knight");
    parse_fen(p, "4k3/3P4/8/8/8/8/8/4K3 b - - 0 1");
    check(inCheck(p, BLACK), "check pawn w");
    parse_fen(p, "4k3/8/8/8/8/8/3p4/4K3 w - - 0 1");
    check(inCheck(p, WHITE), "check pawn b");
    check(legal_total("8/8/8/8/8/8/r7/4K3 w - - 0 1") == 2, "king attacked sq filtered");
    check(!has_legal("4r3/8/8/8/8/8/4N3/4K3 w - - 0 1", "e2d4"), "pinned knight cannot move");
    check(has_legal("7b/8/8/8/8/2B5/8/K7 w - - 0 1", "c3h8"), "pinned bishop capture legal");
    check(!has_legal("7b/8/8/8/8/2B5/8/K7 w - - 0 1", "c3d4"), "pinned bishop off-diagonal");
    check(has_legal("4r3/8/8/8/8/8/8/1B2K3 w - - 0 1", "b1e4"), "block check legal");
    check(!has_legal("4r3/8/8/8/8/8/8/1B2K3 w - - 0 1", "b1a2"), "non-blocking illegal");
    check(!has_legal("8/8/8/r2pP2K/8/8/8/8 w - d6 0 1", "e5d6"), "ep pin illegal");
    check(has_legal("8/8/8/3pP3/8/8/8/4K2k w - d6 0 1", "e5d6"), "ep legal");
    const char* C_CHK = "r3k2r/8/8/8/8/8/4r3/R3K2R w KQkq - 0 1";
    check(!has_legal(C_CHK, "e1g1") && !has_legal(C_CHK, "e1c1"), "castle out of check illegal");
    const char* C_TRN = "r3k2r/8/8/8/8/5r2/8/R3K2R w KQkq - 0 1";
    check(!has_legal(C_TRN, "e1g1") && has_legal(C_TRN, "e1c1"), "castle through check illegal");
    const char* C_DST = "r3k2r/8/8/8/8/6r1/8/R3K2R w KQkq - 0 1";
    check(!has_legal(C_DST, "e1g1"), "castle into check illegal");
    check(legal_total("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") == 20, "startpos 20 legal moves");
    check(legal_total("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1") == 20, "startpos black 20 legal moves");
}
static void zrt(const char* fen, const char* mv, const char* name) {
    Position p; Move m{}; Undo u;
    bool ok = parse_fen(p, fen) && getmv(p, mv, m);
    check(ok, name);
    if (!ok) return;
    uint64_t h = p.key;
    makeMove(p, m, u);
    check(p.key == computeHash(p), name);
    unmakeMove(p, m, u);
    check(p.key == h, name);
}
static void test_z() {
    Position a, b;
    parse_fen(a, "8/8/8/8/8/8/8/4K3 w - - 0 1");
    parse_fen(b, "8/8/8/8/8/8/8/4K3 w - - 0 1");
    check(a.key == b.key && a.key == computeHash(a), "z same");
    parse_fen(b, "8/8/8/8/8/8/8/4K3 b - - 0 1");
    check(a.key != b.key, "z side");
    parse_fen(b, "8/8/8/8/8/8/8/3K4 w - - 0 1");
    check(a.key != b.key, "z pieces");
    parse_fen(a, "8/8/8/8/8/8/8/R3K2R w KQ - 0 1");
    parse_fen(b, "8/8/8/8/8/8/8/R3K2R w K - 0 1");
    check(a.key != b.key, "z castle");
    // build0.3: a raw ep square with no actual capturing pawn on the board
    // must NOT change the hash. This bare-king position has no pawns at all,
    // so "e3" is a dead ep square and both FENs must hash identically.
    parse_fen(a, "8/8/8/8/8/8/8/4K3 w - - 0 1");
    parse_fen(b, "8/8/8/8/8/8/8/4K3 w - e3 0 1");
    check(a.key == b.key, "z ep dead square canonicalized");
    zrt("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4", "z normal");
    zrt("8/8/8/4p3/3P4/8/8/8 w - - 5 3", "d4e5", "z capture");
    zrt("8/4P3/8/8/8/8/8/8 w - - 0 1", "e7e8q", "z promotion");
    zrt("5r2/4P3/8/8/8/8/8/8 w - - 0 1", "e7f8q", "z cap-promo");
    zrt("8/8/8/3pP3/8/8/8/8 w - d6 0 1", "e5d6", "z en-passant");
    zrt("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", "z castling");
}
// ============================================================================
// SEARCH (build 1.5): поиск переписан с нуля.
// Не менялись: позиция, генератор ходов, make/unmake, SEE-примитивы, оценка
// (A, NNUE), UCI-разбор, наборы тестов. Переписаны: состояние поиска, порядок
// ходов, qsearch, negamax, итеративное углубление, PV, вывод info.
// ============================================================================
// ----------------------------------------------------------------------------
// build 1.5: константы поиска.
// Изменено: добавлены SEARCH_MAX_PLY (лимит глубины стека), SEARCH_CHECK_EXT_PATH_MAX
// (лимит продлений за шах на путь), SEARCH_QS_MAX_PLY (глубина qsearch с тихими шахами).
// Старые константы оценки (SEARCH_FUTILITY_MARGIN, SEARCH_DELTA_*) сохранены по значению.
// ----------------------------------------------------------------------------
constexpr int SEARCH_INFINITE_SCORE = 30000;
constexpr int SEARCH_MATE_SCORE = 29000;
constexpr int SEARCH_MAX_PLY = 96;
static const int search_piece_vals[] = { 0, 100, 320, 330, 500, 900, 20000 };
static const int SEARCH_DELTA_MARGIN = 200;
static const int SEARCH_DELTA_NPM_MIN = 1300;
// Лимит продлений за шах на один путь от корня (1.4: безлимитно, что дало взрыв узлов).
static const int SEARCH_CHECK_EXT_PATH_MAX = 4;
// Глубина qsearch от горизонта; на первом уровне qsearch допускаются тихие шахи.
static const int SEARCH_QS_MAX_PLY = 8;
static const int SEARCH_FUTILITY_MARGIN[3] = { 0, 200, 450 };
// build 1.5: массив оставлен, но в поиске не используется: RFP отключён (см. negamax).
static const int SEARCH_RFP_MARGIN[4] = { 0, 120, 250, 400 };

// ============================================================================
// FROZEN EVALUATION A
// ============================================================================
static const int EV_MG[7] = { 0, 82, 337, 365, 477, 1025, 0 };
static const int EV_EG[7] = { 0, 94, 281, 297, 512, 936, 0 };
static const int EV_PHASE[7] = { 0, 0, 1, 1, 2, 4, 0 };
static inline int ev_sign(int c) {
    return c == WHITE ? 1 : -1;
}
static inline int ev_relrank(int sq, int c) {
    int r = rank_of(sq);
    return c == WHITE ? r : 7 - r;
}
static inline int ev_center(int sq) {
    int f = file_of(sq), r = rank_of(sq);
    int df = f < 4 ? 3 - f : f - 4;
    int dr = r < 4 ? 3 - r : r - 4;
    return 6 - df - dr;
}
static inline int ev_blend(int mg, int eg, int phase) {
    return (mg * phase + eg * (24 - phase)) / 24;
}
static void ev_mark_attack(bool a[128], int sq) {
    if (on_board(sq)) a[sq] = true;
}
static int ev_slider_attacks(const Position& p, int sq, const int* d, int n,
    bool a[128], int us) {
    int mob = 0;
    for (int i = 0; i < n; ++i) {
        for (int to = sq + d[i]; on_board(to); to += d[i]) {
            int pc = bb_piece_ref(p, to);
            ev_mark_attack(a, to);
            if (!pc) {
                ++mob;
                continue;
            }
            if (color_of(pc) != us) ++mob;
            break;
        }
    }
    return mob;
}
static int ev_step_attacks(const Position& p, int sq, const int* d, int n,
    bool a[128], int us) {
    int mob = 0;
    for (int i = 0; i < n; ++i) {
        int to = sq + d[i];
        if (!on_board(to)) continue;
        a[to] = true;
        if (!bb_piece_ref(p, to) || color_of(bb_piece_ref(p, to)) != us) ++mob;
    }
    return mob;
}
// build0.5: passed-pawn geometry uses absolute board ranks. Only enemy
// pawns on the same or adjacent files strictly ahead toward promotion block.
static bool ev_is_passed_pawn(const Position& p, int c, int sq) {
    const int f = file_of(sq);
    const int enemy = c ^ 1;
    const int step = (c == WHITE) ? 1 : -1;

    for (int ef = std::max(0, f - 1);
        ef <= std::min(7, f + 1);
        ++ef) {
        for (int rr = rank_of(sq) + step;
            valid_rank(rr);
            rr += step) {
            if (bb_piece_ref(p, sq88(ef, rr)) == make_pc(PAWN, enemy))
                return false;
        }
    }
    return true;
}

static int ev_material_scale(const int count[2][7]) {
    int pawns = count[WHITE][PAWN] + count[BLACK][PAWN];
    int rq = count[WHITE][ROOK] + count[BLACK][ROOK] +
        count[WHITE][QUEEN] + count[BLACK][QUEEN];
    int minors = count[WHITE][KNIGHT] + count[BLACK][KNIGHT] +
        count[WHITE][BISHOP] + count[BLACK][BISHOP];
    if (pawns == 0 && rq == 0 && minors <= 1) return 0;
    if (pawns == 0 && rq == 0 && minors == 2 &&
        count[WHITE][BISHOP] == 1 && count[BLACK][BISHOP] == 1) return 0;
    return 128;
}
static const short W7_PST_MG[6][64] = {
    {   82,   84,   86,   95,   95,   86,   84,   82,
        95,  103,  109,  103,  111,  144,  155,  103,
        92,   81,  117,  101,  116,  112,  127,  100,
        88,   89,  113,  133,  133,  128,  106,   75,
       112,  108,  115,  128,  140,  132,  116,   93,
       113,  116,  144,  129,  156,  207,  116,  111,
       163,  114,   99,  133,  152,  161,   72,   51,
       135,  137,  139,  155,  155,  139,  137,  135,},
    {  403,  450,  425,  451,  451,  459,  446,  387,
       451,  435,  469,  486,  485,  482,  472,  467,
       452,  472,  494,  490,  503,  502,  505,  458,
       446,  443,  477,  461,  483,  481,  464,  436,
       468,  473,  462,  505,  458,  490,  440,  479,
       441,  463,  468,  508,  533,  562,  495,  423,
       366,  415,  540,  475,  531,  530,  447,  441,
       179,  393,  392,  397,  485,  320,  301,  294,},
    {  453,  428,  459,  453,  439,  433,  438,  455,
       419,  484,  455,  455,  462,  476,  501,  451,
       473,  471,  472,  444,  460,  483,  464,  477,
       437,  438,  441,  460,  465,  442,  451,  424,
       413,  431,  435,  448,  448,  414,  441,  432,
       417,  434,  486,  436,  500,  461,  477,  454,
       393,  436,  406,  412,  424,  486,  408,  450,
       402,  366,  378,  342,  300,  399,  409,  346,},
    {  624,  625,  634,  636,  644,  648,  612,  641,
       600,  606,  601,  615,  619,  631,  660,  577,
       592,  598,  607,  620,  620,  634,  663,  636,
       589,  580,  570,  580,  597,  601,  632,  609,
       575,  600,  606,  617,  595,  626,  644,  646,
       602,  596,  583,  609,  634,  659,  706,  663,
       601,  578,  616,  666,  624,  682,  710,  676,
       643,  681,  657,  648,  675,  729,  707,  632,},
    { 1164, 1153, 1165, 1175, 1164, 1145, 1116, 1181,
      1156, 1163, 1164, 1168, 1175, 1189, 1189, 1163,
      1143, 1168, 1143, 1147, 1147, 1168, 1174, 1162,
      1133, 1128, 1124, 1110, 1134, 1134, 1154, 1137,
      1113, 1117, 1108, 1090, 1098, 1109, 1110, 1148,
      1113, 1106, 1133, 1114, 1146, 1115, 1215, 1127,
      1107, 1096, 1073, 1037, 1041, 1167, 1114, 1266,
      1122, 1114, 1147, 1178, 1199, 1178, 1081, 1122,},
    {  -48,   27,   41,  -41,  -23,   -8,   34,   34,
        13,  -55,  -67, -115,  -97,  -82,  -13,  -15,
        13,   32,  -73, -108,  -96,  -51,  -21,  -63,
        -2,   48,   -2,  -94,  -75,  -78,  -30, -110,
         0,   15,   42,  -19,  -33,   11,   17,  -63,
        70,  187,   48,   73,    3,   79,  116,  -21,
        38,  164,  110,   68,   63,  153,   -4,  -33,
       157,   62,  173,  116,  127,  135,   69,   55,},
};
static const short W7_PST_EG[6][64] = {
    {   94,   95,   96,   97,   97,   96,   95,   94,
       112,  108,  109,  108,  108,  104,   91,   89,
       100,  103,   93,  100,  101,  100,   90,   86,
       112,  111,   92,   83,   89,   89,   99,   96,
       126,  120,  104,   86,   90,   97,  111,  108,
       150,  137,  119,   85,   75,   84,  115,  122,
       211,  212,  195,  153,  142,  148,  203,  200,
       182,  183,  184,  192,  192,  184,  183,  182,},
    {  274,  302,  329,  329,  320,  330,  295,  304,
       304,  321,  321,  330,  328,  329,  314,  298,
       312,  334,  332,  358,  355,  336,  322,  303,
       332,  348,  368,  377,  379,  358,  340,  334,
       334,  356,  376,  388,  394,  383,  367,  331,
       311,  330,  373,  369,  346,  346,  328,  336,
       316,  334,  317,  342,  317,  311,  317,  297,
       320,  283,  318,  336,  317,  325,  315,  249,},
    {  289,  308,  292,  306,  311,  313,  295,  292,
       314,  289,  302,  308,  312,  300,  293,  289,
       293,  317,  311,  327,  320,  308,  302,  294,
       305,  314,  329,  316,  318,  323,  304,  308,
       322,  326,  327,  332,  327,  326,  310,  315,
       324,  313,  309,  334,  311,  330,  318,  313,
       310,  309,  318,  309,  321,  304,  320,  284,
       308,  303,  309,  323,  330,  321,  308,  332,},
    {  597,  596,  594,  595,  585,  586,  593,  563,
       593,  596,  606,  600,  598,  593,  585,  598,
       593,  596,  600,  597,  597,  589,  578,  573,
       605,  611,  619,  620,  614,  609,  593,  593,
       620,  609,  616,  610,  614,  610,  601,  602,
       621,  623,  624,  618,  609,  604,  595,  599,
       624,  642,  630,  613,  617,  605,  594,  599,
       618,  603,  614,  611,  604,  589,  593,  611,},
    {  964,  968,  948,  945,  971,  959,  958,  924,
       979,  984,  972,  978,  970,  928,  926,  963,
       951,  940, 1008,  984,  995,  994, 1007,  981,
       989, 1004, 1009, 1043, 1026, 1012, 1009, 1017,
       984, 1012, 1017, 1061, 1076, 1076, 1057, 1025,
       973,  996, 1018, 1033, 1070, 1052,  988, 1023,
      1014, 1004, 1047, 1107, 1091, 1074, 1063,  920,
       986, 1009, 1004,  999,  982, 1012, 1020, 1030,},
    {    8,   13,   20,   31,   36,   32,    9,  -22,
        -2,   36,   52,   66,   66,   63,   39,   18,
        -9,   19,   53,   69,   69,   59,   41,   32,
        -9,   18,   44,   67,   69,   62,   43,   36,
         4,   32,   40,   57,   56,   55,   51,   45,
         5,   22,   41,   37,   53,   57,   56,   50,
         5,   19,   19,   27,   31,   38,   60,   45,
       -66,    1,  -21,    4,    5,   12,   14,  -12,},
};
static const short W7_MG[100] = {
       -1,  -20,    0,    4,    4,   -5,   10,    3,   46,    0,
        4,   -3,   12,  -15,   58,   47,   16,   11,    9,    4,
       36,   26,   12,  -31, -102, -194, -208, -331, -168,   25,
       19,   48,  -31,   66,   -6,  -25,   -5,    1,    9,    9,
       14,   18,   20,  -41,  -14,   -6,   -1,    3,    3,   11,
       13,   18,   24,   33,   36,   42,   55,  -29,   -5,   -1,
        1,    5,   11,   13,   12,   14,   21,   22,   28,   24,
       38,   56,  -37,    5,   10,    9,   12,   18,   21,   20,
       17,   23,   28,   26,   28,   30,   31,   40,   38,   36,
       47,   47,   34,   48,   56,   75,   39,   72,  269, -231,

};
static const short W7_EG[100] = {
       -9,    0,    0,   26,   14,   21,   33,   92,   75,    0,
      -49,   21,   13,   -5,   57,  -10,   10,   -3,    0,   -2,
      -37,  -10,   -4,    3,   40,   69,   79,  106,   14,   -8,
       -2,    3,    8,  -47,   21,    7,    0,    1,    0,    6,
        4,    2,   -4,  -13,  -23,  -24,  -13,   -4,   10,   12,
       23,   24,   25,   27,   29,   45,   55,   10,  -20,  -13,
       -2,    1,    1,    3,    8,   14,   13,   19,   24,   35,
       31,   30, -165,   21,   17,   15,   21,   47,   45,   54,
       79,   88,   94,  107,  110,  120,  124,  112,  123,  139,
      133,  135,  154,  142,  132,  123,  129,  129,   32,  269,

};

enum {
    E7_DOUBLED = 384, E7_ISOLATED = 385, E7_PASSED = 386, E7_PASS_FREE = 394,
    E7_PASS_FREE_R = 395, E7_PAWN_DEF = 396, E7_PAWN_BLOCKED = 397, E7_BISHOP_PAIR = 398,
    E7_ROOK_OPEN = 399, E7_ROOK_SEMI = 400, E7_KNIGHT_OUT = 401, E7_PIECE_DEF = 402,
    E7_CENTER = 403, E7_KRING = 404, E7_KSHIELD1 = 413, E7_KSHIELD2 = 414,
    E7_KOFFHOME = 415, E7_KCENTRALHOME = 416, E7_CASTLE = 417,
    E7_MOB_N = 418, E7_MOB_B = 427, E7_MOB_R = 441, E7_MOB_Q = 456
};
static const int E7_MOB_MAX[7] = { 0, 0, 8, 13, 14, 27, 0 };
static const int E7_MOB_BASE[7] = { 0, 0, E7_MOB_N, E7_MOB_B, E7_MOB_R, E7_MOB_Q, 0 };
static inline int e7_sq64(int sq, int c) {
    int f = file_of(sq), r = rank_of(sq);
    if (c == BLACK) r = 7 - r;
    return r * 8 + f;
}
// ============================================================================
// BITBOARD EVALUATION (ebb_). Bit-exact replacement for the 0x88 evaluate().
// Every term uses the same definition, weight index and sign as the 0x88
// version (reference copy: archive/evaluate_0x88_ref.inc). Integer math is
// unchanged, so results must match exactly; any mismatch is a bug.
// Requires: W7_MG/W7_EG, E7_* enum, E7_MOB_*, EV_PHASE, WK/WQ/BK/BQ,
// bb_*_attack_mask (magic tables initialised), Position bitboards.
// ============================================================================
int evaluate(const Position& pos);
static const uint64_t EBB_NOT_A = 0xfefefefefefefefeULL;
static const uint64_t EBB_NOT_H = 0x7f7f7f7f7f7f7f7fULL;
static const uint64_t EBB_CENTER = 0x3C3C3C3C0000ULL; // ranks 3-6 x files c-f (0-based 2..5)

static inline uint64_t ebb_file(int f) { return 0x0101010101010101ULL << f; }
static inline uint64_t ebb_rank(int r) { return 0xFFULL << (8 * r); }
static inline uint64_t ebb_adj_files(int f) {
    uint64_t m = 0;
    if (f > 0) m |= ebb_file(f - 1);
    if (f < 7) m |= ebb_file(f + 1);
    return m;
}
static inline uint64_t ebb_pawn_attacks_all(uint64_t p, int c) {
    if (c == WHITE) return ((p << 9) & EBB_NOT_A) | ((p << 7) & EBB_NOT_H);
    return ((p >> 7) & EBB_NOT_A) | ((p >> 9) & EBB_NOT_H);
}

// Passed-pawn masks: same/adjacent files, strictly ahead in the pawn's direction.
static uint64_t ebb_passed_mask[2][64];
static bool ebb_tables_ready = false;
static void ebb_init_tables() {
    if (ebb_tables_ready) return;
    ebb_tables_ready = true;
    for (int c = WHITE; c <= BLACK; ++c)
        for (int sq = 0; sq < 64; ++sq) {
            const int f = sq & 7, r = sq >> 3;
            uint64_t m = 0;
            for (int ff = f - 1; ff <= f + 1; ++ff) {
                if (ff < 0 || ff > 7) continue;
                for (int rr = (c == WHITE) ? r + 1 : 0; rr <= ((c == WHITE) ? 7 : r - 1); ++rr)
                    m |= 1ULL << (rr * 8 + ff);
            }
            ebb_passed_mask[c][sq] = m;
        }
}

static inline void ebb_add(int& mg, int& eg, int sg, int idx, int n) {
    mg += sg * n * W7_MG[idx - 384];
    eg += sg * n * W7_EG[idx - 384];
}

// build 1.5: ручная оценка без изменений (из 1.4). Используется, если сеть не загружена.
static int evaluate_bb(const Position& pos) {
    ebb_init_tables();
    const uint64_t occ = pos.bbOccupiedAll;

    // piece counts and game phase (same rule as 0x88: sum of EV_PHASE, capped at 24)
    int count[2][7] = {};
    int phase = 0;
    for (int c = WHITE; c <= BLACK; ++c)
        for (int pt = PAWN; pt <= KING; ++pt) {
            count[c][pt] = bb_popcountll(pos.bbPieces[c][pt]);
            phase += count[c][pt] * EV_PHASE[pt];
        }
    if (phase > 24) phase = 24;

    // attack sets (attacked[c]) and mobility, computed for both sides before any term
    uint64_t pawnAtt[2], attAll[2] = { 0, 0 };
    int mobility[2][7] = {};
    for (int c = WHITE; c <= BLACK; ++c) {
        pawnAtt[c] = ebb_pawn_attacks_all(pos.bbPieces[c][PAWN], c);
        attAll[c] |= pawnAtt[c];
        const uint64_t own = pos.bbOccupied[c];
        uint64_t b = pos.bbPieces[c][KNIGHT];
        while (b) {
            const int sq = bb_ctzll(b); b &= b - 1;
            const uint64_t a = bb_knight_attack_mask(sq);
            attAll[c] |= a;
            mobility[c][KNIGHT] += bb_popcountll(a & ~own);
        }
        b = pos.bbPieces[c][BISHOP];
        while (b) {
            const int sq = bb_ctzll(b); b &= b - 1;
            const uint64_t a = bb_bishop_attack_mask(sq, occ);
            attAll[c] |= a;
            mobility[c][BISHOP] += bb_popcountll(a & ~own);
        }
        b = pos.bbPieces[c][ROOK];
        while (b) {
            const int sq = bb_ctzll(b); b &= b - 1;
            const uint64_t a = bb_rook_attack_mask(sq, occ);
            attAll[c] |= a;
            mobility[c][ROOK] += bb_popcountll(a & ~own);
        }
        b = pos.bbPieces[c][QUEEN];
        while (b) {
            const int sq = bb_ctzll(b); b &= b - 1;
            const uint64_t a = bb_bishop_attack_mask(sq, occ) | bb_rook_attack_mask(sq, occ);
            attAll[c] |= a;
            mobility[c][QUEEN] += bb_popcountll(a & ~own);
        }
        b = pos.bbPieces[c][KING];
        while (b) {
            const int sq = bb_ctzll(b); b &= b - 1;
            attAll[c] |= bb_king_attack_mask(sq);
        }
    }

    int mg = 0, eg = 0;
    for (int c = WHITE; c <= BLACK; ++c) {
        const int sg = (c == WHITE) ? 1 : -1;
        const int enemy = c ^ 1;
        const uint64_t own = pos.bbOccupied[c];
        const uint64_t pawns = pos.bbPieces[c][PAWN];
        const uint64_t enemyPawns = pos.bbPieces[enemy][PAWN];

        if (count[c][BISHOP] >= 2) ebb_add(mg, eg, sg, E7_BISHOP_PAIR, 1);

        for (int pt = KNIGHT; pt <= QUEEN; ++pt) {
            int m = mobility[c][pt];
            if (m > E7_MOB_MAX[pt]) m = E7_MOB_MAX[pt];
            if (m < 0) m = 0;
            ebb_add(mg, eg, sg, E7_MOB_BASE[pt] + m, 1);
        }

        // piece-square tables: black uses the mirrored square (sq ^ 56)
        for (int pt = PAWN; pt <= KING; ++pt) {
            uint64_t b = pos.bbPieces[c][pt];
            while (b) {
                const int sq = bb_ctzll(b); b &= b - 1;
                const int s64 = (c == WHITE) ? sq : (sq ^ 56);
                mg += sg * W7_PST_MG[pt - 1][s64];
                eg += sg * W7_PST_EG[pt - 1][s64];
            }
        }

        // pawns
        {
            uint64_t b = pawns;
            const int dir = (c == WHITE) ? 8 : -8;
            while (b) {
                const int sq = bb_ctzll(b); b &= b - 1;
                const int f = sq & 7, r = sq >> 3;
                const int rel = (c == WHITE) ? r : 7 - r;
                if (bb_popcountll(pawns & ebb_file(f)) > 1) ebb_add(mg, eg, sg, E7_DOUBLED, 1);
                if ((pawns & ebb_adj_files(f)) == 0) ebb_add(mg, eg, sg, E7_ISOLATED, 1);
                if ((ebb_passed_mask[c][sq] & enemyPawns) == 0) {
                    ebb_add(mg, eg, sg, E7_PASSED + rel, 1);
                    const int front = sq + dir;
                    if ((unsigned)front < 64u && !(occ & (1ULL << front))) {
                        ebb_add(mg, eg, sg, E7_PASS_FREE, 1);
                        ebb_add(mg, eg, sg, E7_PASS_FREE_R, rel);
                    }
                }
                if ((pawnAtt[c] >> sq) & 1ULL) ebb_add(mg, eg, sg, E7_PAWN_DEF, 1);
                const int front = sq + dir;
                if ((unsigned)front < 64u && (occ & (1ULL << front))) ebb_add(mg, eg, sg, E7_PAWN_BLOCKED, 1);
            }
        }

        // rooks on open / semi-open files
        {
            uint64_t b = pos.bbPieces[c][ROOK];
            while (b) {
                const int sq = bb_ctzll(b); b &= b - 1;
                const uint64_t fm = ebb_file(sq & 7);
                const bool ownP = (pawns & fm) != 0;
                const bool enemyP = (enemyPawns & fm) != 0;
                if (!ownP && !enemyP) ebb_add(mg, eg, sg, E7_ROOK_OPEN, 1);
                else if (!ownP) ebb_add(mg, eg, sg, E7_ROOK_SEMI, 1);
            }
        }

        // knights on outposts: relative rank 3..5 and not chaseable by an enemy pawn
        {
            uint64_t b = pos.bbPieces[c][KNIGHT];
            while (b) {
                const int sq = bb_ctzll(b); b &= b - 1;
                const int rel = (c == WHITE) ? (sq >> 3) : 7 - (sq >> 3);
                if (rel >= 3 && rel <= 5) {
                    const bool chase = ((pawnAtt[enemy] >> sq) & 1ULL) != 0;
                    if (!chase) ebb_add(mg, eg, sg, E7_KNIGHT_OUT, 1);
                }
            }
        }

        // defended non-pawn, non-king pieces not attacked by enemy pawns
        {
            const uint64_t pieces = pos.bbPieces[c][KNIGHT] | pos.bbPieces[c][BISHOP] |
                pos.bbPieces[c][ROOK] | pos.bbPieces[c][QUEEN];
            const int n = bb_popcountll(pieces & attAll[c] & ~pawnAtt[enemy]);
            if (n) ebb_add(mg, eg, sg, E7_PIECE_DEF, n);
        }

        // centre control: attacked squares in the central 4x4 block
        {
            const int cc = bb_popcountll(attAll[c] & EBB_CENTER);
            if (cc) ebb_add(mg, eg, sg, E7_CENTER, cc);
        }

        // king terms (only when the king exists; highest-index king like the 0x88 scan)
        if (pos.bbPieces[c][KING]) {
            const int ksq = 63 - bb_clzll(pos.bbPieces[c][KING]);
            const int f = ksq & 7, r = ksq >> 3;
            const int ring = bb_popcountll(bb_king_attack_mask(ksq) & attAll[enemy]);
            ebb_add(mg, eg, sg, E7_KRING + ring, 1);

            const uint64_t files3 = ebb_file(f) | ebb_adj_files(f);
            const int dir = (c == WHITE) ? 1 : -1;
            const int r1 = r + dir, r2 = r + 2 * dir;
            if (r1 >= 0 && r1 <= 7) {
                const int s1 = bb_popcountll(pawns & ebb_rank(r1) & files3);
                if (s1) ebb_add(mg, eg, sg, E7_KSHIELD1, s1);
            }
            if (r2 >= 0 && r2 <= 7) {
                const int s2 = bb_popcountll(pawns & ebb_rank(r2) & files3);
                if (s2) ebb_add(mg, eg, sg, E7_KSHIELD2, s2);
            }
            const int home = (c == WHITE) ? 0 : 7;
            if (r != home) ebb_add(mg, eg, sg, E7_KOFFHOME, 1);
            if (f >= 2 && f <= 5 && r == home) ebb_add(mg, eg, sg, E7_KCENTRALHOME, 1);
        }

        // castling rights still available
        if (c == WHITE) { if (pos.castle & (WK | WQ)) ebb_add(mg, eg, sg, E7_CASTLE, 1); }
        else { if (pos.castle & (BK | BQ)) ebb_add(mg, eg, sg, E7_CASTLE, 1); }
    }

    // material scaling (same rule as ev_material_scale)
    int scale = 128;
    {
        const int pawnsAll = count[WHITE][PAWN] + count[BLACK][PAWN];
        const int rq = count[WHITE][ROOK] + count[BLACK][ROOK] + count[WHITE][QUEEN] + count[BLACK][QUEEN];
        const int minors = count[WHITE][KNIGHT] + count[BLACK][KNIGHT] + count[WHITE][BISHOP] + count[BLACK][BISHOP];
        if (pawnsAll == 0 && rq == 0 && minors <= 1) scale = 0;
        else if (pawnsAll == 0 && rq == 0 && minors == 2 &&
            count[WHITE][BISHOP] == 1 && count[BLACK][BISHOP] == 1) scale = 0;
    }

    int score = (mg * phase + eg * (24 - phase)) / 24;
    score = score * scale / 128;
    score += pos.sideToMove == WHITE ? 8 : -8;
    return pos.sideToMove == WHITE ? score : -score;
}

// ============================================================================
// NNUE (build 1.4, версия Chess768): инференс одной сети из nnue-pytorch (serialize.py).
// Признаки Chess768: 768 = 64 клетки x 12 фигур. Королевских бакетов нет.
// Перспектива чёрных: клетка отражается по вертикали (sq ^ 56), как в C++ загрузчике trainer.
// Архитектура: L1=160 (ft, пары 80+80 -> 160 входов хвоста), L2=32, L3=32, ОДИН стек слоёв.
//   160 -> 32 (sqr+crelu, skip) -> 64 -> 32 (sqr+crelu) -> 128 -> 1 + skip.
// Целочисленная схема та же, что в quantize.py: ft int16 масштаб 256 (активация 0..255);
//   пары: (a*b)>>9 -> масштаб 128, клип 0..127; веса int8 (w*128, w*64, w*128); смещения int32.
//   Выход: trunc(T*9600/16384)/16 сантипешек.
// Полный пересчёт аккумулятора на каждую оценку не нужен: используются слоты по глубине
// и дельты на ход (обычный ход, взятие, рокировка, взятие на проходе, превращение).
// Один поток.
// build 1.5: NNUE-часть без изменений (из 1.4). Инкрементальный аккумулятор проверен
// nnuecheck на 0 расхождений; ожидаемый хеш сети 0x6F45FA7F.
namespace nnue {

constexpr int NUM_FEAT = 768;          // Chess768
constexpr int FT_OUT   = 160;          // L1
constexpr int HALF     = FT_OUT / 2;   // 80
constexpr int L2N      = 32;
constexpr int L3N      = 32;
constexpr int FT_CLIP  = 255;          // ft_quantized_max
constexpr int HID_CLIP = 127;          // hidden_quantized_max

// Один стек слоёв (без выбора по числу фигур и ферзям).
struct Stack {
    int32_t b1[L2N];
    int8_t  w1[L2N][FT_OUT];
    int32_t b2[L3N];
    int8_t  w2[L3N][2 * L2N];
    int32_t b3;
    int8_t  w3[2 * L2N + 2 * L3N];
};

struct Net {
    bool loaded = false;
    int32_t ft_hash = 0;
    int16_t ft_bias[FT_OUT];
    int16_t ft_w[NUM_FEAT][FT_OUT];   // строка = признак, столбцы = выходы ft
    Stack   st;
};

static Net  g_net;
static bool g_nnue_on = false;
static char g_nnue_path[512] = "";
// Хеш архитектуры из заголовка файла. Эталон: Chess768, L1=160 (net_c768_v2 и ep42 совпадают).
static uint32_t g_net_hash = 0;
constexpr uint32_t EXPECTED_NET_HASH = 0x6F45FA7Fu;

// ----- признаки Chess768 -----
// Индекс строки: (тип*2 + (цвет фигуры != перспектива)) * 64 + клетка с учётом перспективы.
// type: 1..6 (пешка..король), как в движке; в загрузчике trainer тип 0..5, порядок тот же.
static inline int feature_row(int persp, int sq, int type, int pc_color) {
    const int o = (persp == BLACK) ? (sq ^ 56) : sq;
    const int p_idx = (type - 1) * 2 + (pc_color != persp ? 1 : 0);
    return p_idx * 64 + o;
}

// ----- стек аккумуляторов по глубине поиска -----
// Слот g_stack[p] хранит аккумулятор обеих перспектив для узла на глубине p.
// Инвариант: перед ходом слот текущего узла соответствует позиции (ключ совпадает),
// иначе он пересчитывается полностью. Нулевой ход слотов не меняет.
constexpr int STACK_MAX = 256;
struct Delta {
    int nsub[2];
    int nadd[2];
    int sub[2][4];          // строки признаков, которые снять
    int add[2][4];          // строки признаков, которые добавить
};
struct AccSlot {
    int32_t v[2][FT_OUT];
    uint64_t key;
};
static AccSlot g_stack[STACK_MAX];
static int g_ply = 0;

static void refresh_persp(const Position& pos, int c, int32_t* out);
static void refresh_acc(const Position& pos, int32_t (*buf)[FT_OUT]);

static Delta make_begin(const Position& pos, const Move& mv) {
    Delta d;
    for (int c = 0; c < 2; ++c) { d.nsub[c] = 0; d.nadd[c] = 0; }
    if (!g_nnue_on) return d;   // без сети слоты не нужны (поиск как в build 1.3)
    if (g_ply < STACK_MAX) {
        AccSlot& s = g_stack[g_ply];
        if (s.key != pos.key) { refresh_acc(pos, s.v); s.key = pos.key; }
    }
    // Ходы хранятся в 0x88, позиция — в 64-клеточной нумерации (bb_sq_from_sq88).
    const int from64 = bb_sq_from_sq88(mv.from);
    const int to64   = bb_sq_from_sq88(mv.to);
    const int pc_from = pos.piece[from64];
    const int pc_to   = pos.piece[to64];
    const int mt = piece_of(pc_from), mcol = color_of(pc_from);
    const int ct = piece_of(pc_to),   ccol = color_of(pc_to);

    struct Chg { int p, col, sq; };
    Chg rem[2]; int nrem = 0;
    Chg add[2]; int nadd = 0;
    rem[nrem++] = {mt, mcol, from64};
    if (mv.flags & F_EP) {
        // взятая на проходе пешка стоит за полем назначения (со стороны хода)
        const int cs = to64 + (mcol == WHITE ? -8 : 8);
        rem[nrem++] = {PAWN, mcol ^ 1, cs};
    } else if (ct != EMPTY) {
        rem[nrem++] = {ct, ccol, to64};
    }
    if (mv.flags & F_CASTLE) {
        // Ладья: короткая рокировка (вертикаль g) h->f, длинная a->d. Ранг = ранг короля.
        const int r = rank_of(mv.to);
        const bool ks = file_of(mv.to) == 6;
        const int rf = r * 8 + (ks ? 7 : 0), rt = r * 8 + (ks ? 5 : 3);
        rem[nrem++] = {ROOK, mcol, rf};
        add[nadd++] = {ROOK, mcol, rt};
    }
    add[nadd++] = {(mv.flags & F_PROMO) ? (int)mv.promo : mt, mcol, to64};

    for (int c = 0; c < 2; ++c) {
        for (int k = 0; k < nrem; ++k)
            d.sub[c][d.nsub[c]++] = feature_row(c, rem[k].sq, rem[k].p, rem[k].col);
        for (int k = 0; k < nadd; ++k)
            d.add[c][d.nadd[c]++] = feature_row(c, add[k].sq, add[k].p, add[k].col);
    }
    return d;
}

// Вызывать после makeMove (позиция уже после хода), только для легального хода.
static void make_commit(const Delta& d, const Position& pos) {
    if (!g_nnue_on) return;
    if (g_ply + 1 < STACK_MAX) {
        const AccSlot& parent = g_stack[g_ply];
        AccSlot& child = g_stack[g_ply + 1];
        const Net& n = g_net;
        for (int c = 0; c < 2; ++c) {
            std::memcpy(child.v[c], parent.v[c], sizeof(child.v[c]));
            for (int k = 0; k < d.nsub[c]; ++k) {
                const int16_t* w = n.ft_w[d.sub[c][k]];
                for (int j = 0; j < FT_OUT; ++j) child.v[c][j] -= w[j];
            }
            for (int k = 0; k < d.nadd[c]; ++k) {
                const int16_t* w = n.ft_w[d.add[c][k]];
                for (int j = 0; j < FT_OUT; ++j) child.v[c][j] += w[j];
            }
        }
        child.key = pos.key;
    }
    ++g_ply;
}

static inline void on_unmake() {
    if (g_ply > 0) --g_ply;
}

// ----- чтение файла -----
struct Reader {
    const uint8_t* p; const uint8_t* end; bool ok = true;
    uint32_t u32() {
        if (end - p < 4) { ok = false; return 0; }
        uint32_t v = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
        p += 4; return v;
    }
    void bytes(void* dst, size_t n) {
        if ((size_t)(end - p) < n) { ok = false; return; }
        std::memcpy(dst, p, n); p += n;
    }
};

// Формат: serialize.py (nnue-pytorch), запуск с --features=Chess768 --l1=160 (L2=L3=32).
// Файл должен занимать ровно столько байт, сколько ожидает эта архитектура.
// ----- пути (UTF-8) и папка exe -----
// GUI и Python передают путь в UTF-8. fopen на Windows понимает только ANSI,
// поэтому путь с кириллицей (например, папка "проверка") без перевода не открывается.
static std::string trim_path(std::string s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' ' || s.back() == '\t')) s.pop_back();
    size_t b = 0;
    while (b < s.size() && (s[b] == ' ' || s[b] == '\t')) ++b;
    s = s.substr(b);
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') s = s.substr(1, s.size() - 2);
    return s;
}
static std::FILE* open_utf8(const std::string& path, const char* mode) {
#ifdef _WIN32
    wchar_t wpath[2048];
    wchar_t wmode[8];
    // Путь приходит из GUI либо в UTF-8, либо в ANSI (Arena шлёт cp1251). Пробуем оба варианта.
    if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path.c_str(), -1, wpath, 2048)) {
        if (!MultiByteToWideChar(CP_ACP, 0, path.c_str(), -1, wpath, 2048)) return nullptr;
    }
    MultiByteToWideChar(CP_UTF8, 0, mode, -1, wmode, 8);
    return _wfopen(wpath, wmode);
#else
    return std::fopen(path.c_str(), mode);
#endif
}
static std::string g_exe_dir;   // папка exe, с завершающим разделителем
static bool is_abs_path(const std::string& p) {
    if (p.empty()) return false;
    if (p[0] == '/' || p[0] == '\\') return true;
    return p.size() > 1 && p[1] == ':';
}
static void init_exe_dir(const char* argv0) {
#ifdef _WIN32
    wchar_t wbuf[2048];
    DWORD n = GetModuleFileNameW(nullptr, wbuf, 2048);
    if (n > 0 && n < 2048) {
        char ubuf[8192];
        int len = WideCharToMultiByte(CP_UTF8, 0, wbuf, (int)n, ubuf, (int)sizeof(ubuf) - 1, nullptr, nullptr);
        if (len > 0) {
            ubuf[len] = 0;
            std::string a(ubuf);
            size_t p = a.find_last_of("/\\");
            if (p != std::string::npos) g_exe_dir = a.substr(0, p + 1);
            return;
        }
    }
#endif
    if (argv0) {
        std::string a(argv0);
        size_t p = a.find_last_of("/\\");
        if (p != std::string::npos) g_exe_dir = a.substr(0, p + 1);
    }
}

static bool load_file(const char* path, char* err, size_t errn) {
    std::FILE* f = open_utf8(path, "rb");
    if (!f) { std::snprintf(err, errn, "cannot open %s (%s)", path, std::strerror(errno)); return false; }
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz <= 0) { std::fclose(f); std::snprintf(err, errn, "empty file"); return false; }
    std::vector<uint8_t> buf((size_t)sz);
    size_t got = std::fread(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    if (got != buf.size()) { std::snprintf(err, errn, "read error"); return false; }

    g_net.loaded = false;
    g_nnue_path[0] = 0;
    // Кэш аккумуляторов (g_stack) относится к прошлой сети; без сброса поиск и nneval
    // после смены сети в том же процессе берут старые значения (найдено тестом nnuecheck).
    std::memset(g_stack, 0, sizeof(g_stack));
    Reader r{buf.data(), buf.data() + buf.size()};
    uint32_t version = r.u32();
    if (version != 0x6A448AFAu) { std::snprintf(err, errn, "bad version 0x%08X", version); return false; }
    g_net_hash = r.u32();                      // хеш архитектуры (проверяется с предупреждением)
    uint32_t desc_len = r.u32();
    if (!r.ok || (size_t)(r.end - r.p) < desc_len) { std::snprintf(err, errn, "truncated header"); return false; }
    r.p += desc_len;                           // описание

    g_net.ft_hash = (int32_t)r.u32();
    r.bytes(g_net.ft_bias, sizeof(g_net.ft_bias));
    r.bytes(g_net.ft_w, sizeof(g_net.ft_w));

    r.u32();                                   // хеш FC слоёв
    r.bytes(g_net.st.b1, sizeof(g_net.st.b1));
    r.bytes(g_net.st.w1, sizeof(g_net.st.w1));
    r.bytes(g_net.st.b2, sizeof(g_net.st.b2));
    r.bytes(g_net.st.w2, sizeof(g_net.st.w2));
    r.bytes(&g_net.st.b3, sizeof(g_net.st.b3));
    r.bytes(g_net.st.w3, sizeof(g_net.st.w3));

    if (!r.ok || r.p != r.end) {
        std::snprintf(err, errn, "size mismatch: parsed %ld of %ld bytes (ожидается Chess768, L1=160, один стек)",
                      (long)(r.p - buf.data()), sz);
        return false;
    }
    g_net.loaded = true;
    std::snprintf(g_nnue_path, sizeof(g_nnue_path), "%s", path);
    return true;
}

// ----- оценка -----
static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Полный пересчёт аккумулятора одной перспективы c.
static void refresh_persp(const Position& pos, int c, int32_t* out) {
    const Net& n = g_net;
    for (int j = 0; j < FT_OUT; ++j) out[j] = n.ft_bias[j];
    for (int col = 0; col < 2; ++col) {
        for (int t = PAWN; t <= KING; ++t) {
            uint64_t bb = pos.bbPieces[col][t];
            while (bb) {
                const int sq = bb_ctzll(bb);
                bb &= bb - 1;
                const int16_t* w = n.ft_w[feature_row(c, sq, t, col)];
                for (int j = 0; j < FT_OUT; ++j) out[j] += w[j];
            }
        }
    }
}

static void refresh_acc(const Position& pos, int32_t (*buf)[FT_OUT]) {
    refresh_persp(pos, 0, buf[0]);
    refresh_persp(pos, 1, buf[1]);
}

static int evaluate_nnue(const Position& pos) {
    const Stack& s = g_net.st;
    int32_t tmp[2][FT_OUT];
    AccSlot* slot = (g_ply < STACK_MAX) ? &g_stack[g_ply] : nullptr;
    int32_t (*acc)[FT_OUT] = slot ? slot->v : tmp;
    if (!slot || slot->key != pos.key) {
        refresh_acc(pos, acc);
        if (slot) slot->key = pos.key;
    }

    const int us = pos.sideToMove;           // WHITE=0
    const int32_t* A = acc[us];              // сторона на ходу
    const int32_t* B = acc[1 - us];          // соперник

    // Пары: первые 80 — сторона на ходу (A0*A80...), затем соперник; сдвиг >>9 -> масштаб 128.
    int32_t h[FT_OUT];
    for (int j = 0; j < HALF; ++j) {
        int x = clampi(A[j], 0, FT_CLIP);
        int y = clampi(A[j + HALF], 0, FT_CLIP);
        h[j] = (x * y) >> 9;
        int u = clampi(B[j], 0, FT_CLIP);
        int v = clampi(B[j + HALF], 0, FT_CLIP);
        h[HALF + j] = (u * v) >> 9;
    }

    // L1: S1 = b1 + W1 h (масштаб 16384). Квадрат и CReLU, затем клип 0..127.
    int32_t S1[L2N];
    for (int i = 0; i < L2N; ++i) {
        int64_t acc1 = s.b1[i];
        for (int j = 0; j < FT_OUT; ++j) acc1 += (int64_t)s.w1[i][j] * h[j];
        S1[i] = (int32_t)acc1;
    }
    int32_t x2[2 * L2N];
    for (int i = 0; i < L2N; ++i) {
        int64_t sq = ((int64_t)S1[i] * S1[i]) >> 21;
        x2[i] = (int32_t)clampi((int)(sq > 1000000 ? 1000000 : sq), 0, HID_CLIP);
        x2[L2N + i] = clampi(S1[i] >> 7, 0, HID_CLIP);
    }
    const int32_t skip1 = S1[L2N - 2] - S1[L2N - 1];   // в единицах 1/16384

    // L2: S2 = b2 + W2 x2 (масштаб 8192). Квадрат и CReLU.
    int32_t S2[L3N];
    for (int i = 0; i < L3N; ++i) {
        int64_t a = s.b2[i];
        for (int j = 0; j < 2 * L2N; ++j) a += (int64_t)s.w2[i][j] * x2[j];
        S2[i] = (int32_t)a;
    }
    int32_t x3[2 * L2N + 2 * L3N];
    for (int j = 0; j < 2 * L2N; ++j) x3[j] = x2[j];
    for (int i = 0; i < L3N; ++i) {
        int64_t sq = ((int64_t)S2[i] * S2[i]) >> 19;
        x3[2 * L2N + i] = clampi((int)(sq > 1000000 ? 1000000 : sq), 0, HID_CLIP);
        x3[2 * L2N + L3N + i] = clampi(S2[i] >> 6, 0, HID_CLIP);
    }

    // Выход: S3 = b3 + W3 x3 (масштаб 16384) + skip.
    int64_t S3 = s.b3;
    for (int j = 0; j < 2 * L2N + 2 * L3N; ++j) S3 += (int64_t)s.w3[j] * x3[j];
    int64_t T = S3 + skip1;                  // единицы 1/16384
    int64_t out_int = (2 * T * 9600) / 32768;
    int64_t cp = out_int / 16;
    if (cp > 30000) cp = 30000;
    if (cp < -30000) cp = -30000;
    return (int)cp;
}

// Публичный вход. Возвращает оценку со стороны на ходу, сп.
static int evaluate(const Position& pos) {
    return evaluate_nnue(pos);
}

// ----- отладка: обход дерева, сравнение слота с полным пересчётом -----
static long long g_chk_nodes = 0, g_chk_fail = 0;
static void debug_walk(Position& p, int depth) {
    if (depth <= 0) return;
    MoveList ml;
    generateLegalMoves(p, ml);
    for (int i = 0; i < ml.count; ++i) {
        Delta d = make_begin(p, ml.m[i]);
        Undo u;
        makeMove(p, ml.m[i], u);
        make_commit(d, p);
        ++g_chk_nodes;
        if (g_ply < STACK_MAX) {
            int32_t ref[2][FT_OUT];
            refresh_acc(p, ref);
            if (std::memcmp(ref, g_stack[g_ply].v, sizeof(ref)) != 0) {
                ++g_chk_fail;
                if (g_chk_fail <= 5)
                    std::printf("nnuecheck FAIL ply %d move %d->%d flags %d\n",
                        g_ply, ml.m[i].from, ml.m[i].to, ml.m[i].flags);
            }
        }
        debug_walk(p, depth - 1);
        unmakeMove(p, ml.m[i], u);
        on_unmake();
    }
}

// ----- загрузка с отчётом в UCI (используют setoption, автозагрузка) -----
static bool load_net_report(const std::string& path, const char* tag) {
    char err[256] = {0};
    if (load_file(path.c_str(), err, sizeof(err))) {
        g_nnue_on = true;
        std::printf("info string %s loaded %s net_hash 0x%08X\n", tag, path.c_str(), g_net_hash);
        if (g_net_hash != EXPECTED_NET_HASH)
            std::printf("info string NNUE warning: net_hash 0x%08X differs from expected 0x%08X (other architecture?)\n",
                        g_net_hash, EXPECTED_NET_HASH);
    } else {
        g_nnue_on = false;
        std::printf("info string %s error %s: %s (classical eval)\n", tag, path.c_str(), err);
    }
    std::fflush(stdout);
    return g_nnue_on;
}

// ----- автозагрузка сети (build 1.5) -----
// Основная сеть build 1.5: net_comp_ft_ep49.nnue (Chess768, L1=160, хеш 0x6F45FA7F).
// Порядок поиска: сначала ep49, затем запасной net_c768.nnue. Каждое имя ищется
// сначала рядом с exe, потом в текущей папке. Переименовывать файл не нужно.
// Неподходящий файл (другая архитектура или битый) не роняет движок: выводится
// сообщение об ошибке и работает классическая оценка.
static const char* const NNUE_DEFAULT_NAMES[] = { "net_comp_ft_ep49.nnue", "net_c768.nnue" };
static void autoload(const char* argv0) {
    (void)argv0;   // папка exe уже определена в init_exe_dir
    for (const char* name : NNUE_DEFAULT_NAMES) {
        const std::string cands[2] = { g_exe_dir + name, std::string(name) };
        for (const std::string& path : cands) {
            std::FILE* f = open_utf8(path, "rb");
            if (!f) continue;
            std::fclose(f);
            load_net_report(path, "NNUE");
            return;
        }
    }
    std::printf("info string NNUE off: no default net (net_comp_ft_ep49.nnue or net_c768.nnue) next to exe or in cwd (classical eval)\n");
    std::fflush(stdout);
}

}  // namespace nnue

// build 1.4: при загруженной сети (setoption name EvalFile) оценка берётся из NNUE.
int evaluate(const Position& pos) { return nnue::g_nnue_on ? nnue::evaluate(pos) : evaluate_bb(pos); }

struct SearchTTEntry {
    uint64_t key = 0;
    Move move = { 0, 0, 0, 0 };
    int32_t score = 0;
    int8_t depth = 0, flag = 0;
};
// v03a: dynamic transposition table.
// The old version had a fixed 1<<16 TT, which is far too small for real games.
// This keeps the same TT logic, but lets GUI/Arena set Hash in MB.
static constexpr int SEARCH_DEFAULT_HASH_MB = 64;
static constexpr int SEARCH_MIN_HASH_MB = 1;
static constexpr int SEARCH_MAX_HASH_MB = 1024;
static int search_hash_mb = SEARCH_DEFAULT_HASH_MB;
static std::vector<SearchTTEntry> search_tt;
static size_t search_tt_mask = 0;
static SearchTTEntry search_tt_dummy;
static int search_clamp_hash_mb(int mb) {
    if (mb < SEARCH_MIN_HASH_MB) return SEARCH_MIN_HASH_MB;
    if (mb > SEARCH_MAX_HASH_MB) return SEARCH_MAX_HASH_MB;
    return mb;
}
static size_t search_floor_power_of_two(size_t x) {
    if (x < 1) return 1;
    size_t p = 1;
    while (p <= x / 2) p <<= 1;
    return p;
}
static size_t search_tt_entries_for_mb(int mb) {
    mb = search_clamp_hash_mb(mb);
    uint64_t bytes = (uint64_t)mb * 1024ULL * 1024ULL;
    size_t entries = (size_t)(bytes / (uint64_t)sizeof(SearchTTEntry));
    if (entries < 1) entries = 1;
    // Important for speed: keep TT size power-of-two so indexing is key & mask,
    // not a slow runtime modulo/division in every searched node.
    return search_floor_power_of_two(entries);
}
static bool resize_search_tt(int mb) {
    mb = search_clamp_hash_mb(mb);
    size_t entries = search_tt_entries_for_mb(mb);
    try {
        std::vector<SearchTTEntry> fresh(entries);
        search_tt.swap(fresh);
        search_tt_mask = search_tt.empty() ? 0 : (search_tt.size() - 1);
        search_hash_mb = mb;
        return true;
    }
    catch (...) {
        return false;
    }
}
static void clear_search_tt() {
    if (search_tt.empty()) {
        resize_search_tt(search_hash_mb);
        return;
    }
    std::fill(search_tt.begin(), search_tt.end(), SearchTTEntry{});
}
static SearchTTEntry& search_tt_entry(uint64_t key) {
    if (search_tt.empty() && !resize_search_tt(search_hash_mb)) {
        search_tt_dummy = SearchTTEntry{};
        return search_tt_dummy;
    }
    return search_tt[(size_t)key & search_tt_mask];
}
// ---- состояние поиска (build 1.5) ----
// build 1.5: состояние поиска по ply (стек).
// PV: треугольная таблица. В 1.4 PV восстанавливалась из TT (ненадёжно при TT-отсечениях);
// треугольная таблица даёт точную главную линию, которую печатает info.
struct SearchPVTable {
    int length = 0;
    Move moves[SEARCH_MAX_PLY];
};
static SearchPVTable search_pv_table[SEARCH_MAX_PLY + 1];
static Move search_killer_moves[2][SEARCH_MAX_PLY];
// build 1.5: история тихих ходов обновляется по формуле gravity
// (см. search_hist_bonus). В 1.4 значение только росло: += depth*depth без ограничения.
static int search_history_moves[16][128];      // [pc][to88], gravity-обновление
// build 1.5: countermove — ход, который лучше всего отвечал на предыдущий ход
// (индекс: фигура и поле предыдущего хода). Нового в 1.4 не было.
static Move search_countermove[16][128];       // [pc предыдущего хода][to предыдущего хода]
static Move search_move_at[SEARCH_MAX_PLY + 2];
static int search_piece_at[SEARCH_MAX_PLY + 2];
// build 1.5: сколько продлений за шах уже использовано на пути от корня до этого ply.
// Нужно для лимита SEARCH_CHECK_EXT_PATH_MAX (в 1.4 лимита не было).
static int search_ext_used[SEARCH_MAX_PLY + 2];
static int search_lmr_table[64][64];
static bool search_lmr_ready = false;
// build 1.5: таблица LMR-редукций: r = 0.75 + ln(depth) * ln(moveNumber) / 2.25.
// Считается один раз. Это стандартная формула, но коэффициенты подобраны без SPRT.
static void search_init_lmr() {
    if (search_lmr_ready) return;
    search_lmr_ready = true;
    for (int d = 1; d < 64; ++d)
        for (int m = 1; m < 64; ++m)
            search_lmr_table[d][m] = (int)(0.75 + std::log((double)d) * std::log((double)m) / 2.25);
}
struct SearchLimitsA {
    long long start_time = 0, time_limit = 0;
    bool check_time = false;
    std::atomic<bool> stopped{ false };
    long long nodes = 0;
    uint64_t node_limit = 0;
    bool check_nodes = false;
};
static SearchLimitsA search_limits;
static long long g_tm_soft_ms = 0;
enum SearchStopReason : int {
    SEARCH_STOP_NONE = 0, SEARCH_STOP_DEPTH, SEARCH_STOP_TIME, SEARCH_STOP_NODES,
    SEARCH_STOP_MATE, SEARCH_STOP_STOPCMD, SEARCH_STOP_TM_SOFT
};
static std::atomic<int> search_stop_reason{ SEARCH_STOP_NONE };
static void search_set_reason_if_none(SearchStopReason reason) {
    int expected = SEARCH_STOP_NONE;
    search_stop_reason.compare_exchange_strong(expected, (int)reason);
}
static const char* search_stop_reason_name() {
    switch (search_stop_reason.load()) {
    case SEARCH_STOP_DEPTH: return "depth";
    case SEARCH_STOP_TIME: return "time";
    case SEARCH_STOP_NODES: return "nodes";
    case SEARCH_STOP_MATE: return "mate";
    case SEARCH_STOP_STOPCMD: return "stop";
    case SEARCH_STOP_TM_SOFT: return "tm_soft";
    default: return "none";
    }
}
// build 1.5: счётчики. Старые поля сохранены (их печатает print_telemetry_line).
// Новые: checkExtensions, mateDistanceCutoffs, pvsResearches, qsChecks, lmpPruned,
// killerCutoffs, countermoveCutoffs. rfpCutoffs остаётся в структуре, но в 1.5 всегда 0
// (RFP отключён).
struct SearchTelemetry {
    uint64_t qnodes = 0, ttProbes = 0, ttHits = 0, ttCutoffs = 0;
    uint64_t betaCutoffs = 0, firstMoveCutoffs = 0;
    uint64_t nullTries = 0, nullCutoffs = 0;
    uint64_t lmrReductions = 0, lmrResearches = 0, aspirationResearches = 0;
    uint64_t seeCalls = 0, seePruned = 0, futilityPruned = 0, deltaPruned = 0;
    uint64_t tmSoftStops = 0, tmFailLowExtends = 0;
    uint64_t checkExtensions = 0, mateDistanceCutoffs = 0, rfpCutoffs = 0;
    uint64_t pvsResearches = 0, qsChecks = 0, lmpPruned = 0;
    uint64_t killerCutoffs = 0, countermoveCutoffs = 0;
};
static SearchTelemetry search_telemetry;
static std::thread g_search_thread;
static std::atomic<bool> g_search_running{ false };
static std::atomic<bool> g_search_initialized{ false };
static void stop_search() {
    search_limits.stopped = true;
    search_set_reason_if_none(SEARCH_STOP_STOPCMD);
}
static inline long long search_now() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}
static inline void search_check_limits() {
    if (search_limits.check_nodes && (uint64_t)search_limits.nodes >= search_limits.node_limit) {
        search_limits.stopped = true;
        search_set_reason_if_none(SEARCH_STOP_NODES);
        return;
    }
    if (search_limits.check_time && search_now() - search_limits.start_time >= search_limits.time_limit) {
        search_limits.stopped = true;
        search_set_reason_if_none(SEARCH_STOP_TIME);
    }
}
static inline bool search_enter_node() {
    if (search_limits.stopped) return false;
    search_limits.nodes++;
    if (((uint64_t)search_limits.nodes & 2047) == 0) search_check_limits();
    return !search_limits.stopped;
}
// build 1.5: суммирование всех счётчиков. Пока не вызывается: uci_bench агрегирует
// поля вручную, как в 1.4. Можно подключить, чтобы включить новые поля в сумму bench.
static void search_telemetry_add(SearchTelemetry& a, const SearchTelemetry& t) {
    a.qnodes += t.qnodes; a.ttProbes += t.ttProbes; a.ttHits += t.ttHits; a.ttCutoffs += t.ttCutoffs;
    a.betaCutoffs += t.betaCutoffs; a.firstMoveCutoffs += t.firstMoveCutoffs;
    a.nullTries += t.nullTries; a.nullCutoffs += t.nullCutoffs;
    a.lmrReductions += t.lmrReductions; a.lmrResearches += t.lmrResearches;
    a.aspirationResearches += t.aspirationResearches;
    a.seeCalls += t.seeCalls; a.seePruned += t.seePruned;
    a.futilityPruned += t.futilityPruned; a.deltaPruned += t.deltaPruned;
    a.tmSoftStops += t.tmSoftStops; a.tmFailLowExtends += t.tmFailLowExtends;
    a.checkExtensions += t.checkExtensions; a.mateDistanceCutoffs += t.mateDistanceCutoffs;
    a.rfpCutoffs += t.rfpCutoffs; a.pvsResearches += t.pvsResearches;
    a.qsChecks += t.qsChecks; a.lmpPruned += t.lmpPruned;
    a.killerCutoffs += t.killerCutoffs; a.countermoveCutoffs += t.countermoveCutoffs;
}

static inline int search_mvv_lva(int attacker, int victim) {
    return search_piece_vals[piece_of(victim)] * 10 - piece_of(attacker);
}
// ============================================================================
// build0.8: SEE — Static Exchange Evaluation, intermediate variant B.
//
// Classic iterative swap algorithm on the 0x88 board:
//   - full attacker discovery for both sides against the target square;
//   - X-RAY SUPPORT: pieces already used in the exchange are marked in
//     removed[] and treated as transparent by the sliding scans, so a
//     rook/queen (on rook lines) or bishop/queen (on diagonals) hiding
//     behind a previous attacker enters the exchange naturally;
//   - NO pin awareness (Level A philosophy, same contract as
//     hash_ep_square() from build0.3): SEE is a cheap local estimate,
//     real legality stays with generateLegalMoves()/search_make_move();
//   - king participation: the king uses search_piece_vals[KING] = 20000,
//     so a king "capture" that gets recaptured folds to a huge loss and
//     is discarded by the minimax fold — the standard safe treatment.
//
// see_least_valuable_attacker(): returns the square of the least valuable
// attacker of `side` against `to`, considering removed[] squares as empty
// (that is exactly what produces x-rays). Scan order = ascending value:
// pawn, knight, bishop, rook, queen, king.
// ============================================================================
static int see_least_valuable_attacker_0x88(const Position& p, const bool removed[128],
    int to, int side) {
    // Pawns: a pawn of `side` attacks `to` from one rank behind (relative
    // to its own pushing direction), files +/-1.
    const int fwd = (side == WHITE) ? 16 : -16;
    for (int df = -1; df <= 1; df += 2) {
        int s = to - fwd + df;
        if (on_board(s) && !removed[s] && bb_piece_ref(p, s) == make_pc(PAWN, side))
            return s;
    }
    for (int d : KNIGHT_DIRS) {
        int s = to + d;
        if (on_board(s) && !removed[s] && bb_piece_ref(p, s) == make_pc(KNIGHT, side))
            return s;
    }
    // Sliders: removed[] squares are transparent — this is the x-ray.
    for (int d : BISHOP_DIRS) {
        for (int s = to + d; on_board(s); s += d) {
            if (removed[s]) continue;
            int pc = bb_piece_ref(p, s);
            if (!pc) continue;
            if (pc == make_pc(BISHOP, side)) return s;
            break;
        }
    }
    for (int d : ROOK_DIRS) {
        for (int s = to + d; on_board(s); s += d) {
            if (removed[s]) continue;
            int pc = bb_piece_ref(p, s);
            if (!pc) continue;
            if (pc == make_pc(ROOK, side)) return s;
            break;
        }
    }
    for (int d : BISHOP_DIRS) {
        for (int s = to + d; on_board(s); s += d) {
            if (removed[s]) continue;
            int pc = bb_piece_ref(p, s);
            if (!pc) continue;
            if (pc == make_pc(QUEEN, side)) return s;
            break;
        }
    }
    for (int d : ROOK_DIRS) {
        for (int s = to + d; on_board(s); s += d) {
            if (removed[s]) continue;
            int pc = bb_piece_ref(p, s);
            if (!pc) continue;
            if (pc == make_pc(QUEEN, side)) return s;
            break;
        }
    }
    for (int d : KING_DIRS) {
        int s = to + d;
        if (on_board(s) && !removed[s] && bb_piece_ref(p, s) == make_pc(KING, side))
            return s;
    }
    return NO_SQ;
}
// search_see(): static exchange value of the capture `mv` in centipawns,
// from the perspective of the side making the move. >= 0 means the capture
// does not lose material by the swap-off estimate.
// build 1.5: SEE без изменений (из 1.4). Тесты SEE (run_see_tests) проходят.
static int search_see_0x88(const Position& pos, const Move& mv) {
    search_telemetry.seeCalls++;
    const int to = mv.to;
    bool removed[128] = {};
    int gain[36];
    int side = color_of(bb_piece_ref(pos, mv.from));
    // First capture: what is being taken off `to`.
    if (mv.flags & F_EP) {
        gain[0] = search_piece_vals[PAWN];
        // The captured pawn is NOT on `to` for en passant.
        removed[to + (side == WHITE ? -16 : 16)] = true;
    }
    else {
        gain[0] = bb_piece_ref(pos, to) ? search_piece_vals[piece_of(bb_piece_ref(pos, to))] : 0;
    }
    // The mover now occupies `to`; its origin square becomes transparent.
    removed[mv.from] = true;
    int occupier_val = search_piece_vals[piece_of(bb_piece_ref(pos, mv.from))];
    side ^= 1;
    int d = 0;
    while (d + 1 < 36) {
        int from = see_least_valuable_attacker_0x88(pos, removed, to, side);
        if (from == NO_SQ) break;
        ++d;
        gain[d] = occupier_val - gain[d - 1];
        occupier_val = search_piece_vals[piece_of(bb_piece_ref(pos, from))];
        removed[from] = true;
        side ^= 1;
    }
    while (d > 0) {
        gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
        --d;
    }
    return gain[0];
}

// Stage 6 SEE attacker selection. The bitboard attack masks provide the
// candidate set after removed exchange pieces are made transparent. The final
// selection follows the old 0x88 direction order so equal-valued attackers
// preserve the original deterministic SEE behavior.
static int bb_first_see_attacker(const Position& p, Bitboard removed,
    int to, int side) {
    const int target64 = bb_sq_from_sq88(to);
    const Bitboard live = ~removed;
    const Bitboard target = bb_square(target64);

    Bitboard candidates = p.bbPieces[side][PAWN] & live;
    int from = bb_first_pawn_attacker(candidates, target64, side);
    if (from != NO_SQ) return from;

    candidates = p.bbPieces[side][KNIGHT] & live &
        bb_knight_attack_mask(target64);
    from = bb_first_candidate_on_dirs(candidates, target64, KNIGHT_DIRS, 8);
    if (from != NO_SQ) return from;

    const Bitboard occupied = p.bbOccupiedAll & live;
    const Bitboard bishop_rays = bb_bishop_attack_mask(target64, occupied);
    const Bitboard rook_rays = bb_rook_attack_mask(target64, occupied);

    candidates = bishop_rays & p.bbPieces[side][BISHOP] & live;
    from = bb_first_candidate_on_dirs(candidates, target64, BISHOP_DIRS, 4);
    if (from != NO_SQ) return from;

    candidates = rook_rays & p.bbPieces[side][ROOK] & live;
    from = bb_first_candidate_on_dirs(candidates, target64, ROOK_DIRS, 4);
    if (from != NO_SQ) return from;

    candidates = bishop_rays & p.bbPieces[side][QUEEN] & live;
    from = bb_first_candidate_on_dirs(candidates, target64, BISHOP_DIRS, 4);
    if (from != NO_SQ) return from;

    candidates = rook_rays & p.bbPieces[side][QUEEN] & live;
    from = bb_first_candidate_on_dirs(candidates, target64, ROOK_DIRS, 4);
    if (from != NO_SQ) return from;

    candidates = p.bbPieces[side][KING] & live &
        bb_king_attack_mask(target64);
    return bb_first_candidate_on_dirs(candidates, target64, KING_DIRS, 8);
}

// Bitboard SEE path. It preserves the original build0.8 contract: x-rays are
// allowed through removed pieces and pins are intentionally not considered.
static int search_see(const Position& pos, const Move& mv) {
    search_telemetry.seeCalls++;
    const int to = mv.to;
    Bitboard removed = 0;
    int gain[36];
    int side = color_of(bb_piece_ref(pos, mv.from));
    const Bitboard from_bit = bb_square(bb_sq_from_sq88(mv.from));
    if (mv.flags & F_EP) {
        gain[0] = search_piece_vals[PAWN];
        const int captured = to + (side == WHITE ? -16 : 16);
        removed |= bb_square(bb_sq_from_sq88(captured));
    }
    else {
        gain[0] = bb_piece_ref(pos, to) ? search_piece_vals[piece_of(bb_piece_ref(pos, to))] : 0;
    }
    removed |= from_bit;
    int occupier_val = search_piece_vals[piece_of(bb_piece_ref(pos, mv.from))];
    side ^= 1;
    int d = 0;
    while (d + 1 < 36) {
        int from = bb_first_see_attacker(pos, removed, to, side);
        if (from == NO_SQ) break;
        ++d;
        gain[d] = occupier_val - gain[d - 1];
        occupier_val = search_piece_vals[piece_of(bb_piece_ref(pos, from))];
        removed |= bb_square(bb_sq_from_sq88(from));
        side ^= 1;
    }
    while (d > 0) {
        gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
        --d;
    }
    return gain[0];
}

// ---- порядок ходов (build 1.5) ----
// Полоса очков: TT 1 000 000 > хорошее взятие 500 000+ > promo 400 000+ >
// killer 300 000 / 299 000 > countermove 298 000 > тихие ходы (история, +-5000)
// > плохие взятия -20 000+ (ниже тихих).
static const int SEARCH_SCORE_TT = 1000000;
static const int SEARCH_SCORE_GOOD_CAP = 500000;
static const int SEARCH_SCORE_PROMO = 400000;
static const int SEARCH_SCORE_KILLER1 = 300000;
static const int SEARCH_SCORE_KILLER2 = 299000;
static const int SEARCH_SCORE_COUNTER = 298000;
static const int SEARCH_SCORE_BAD_CAP = -20000;

static inline int score_to_tt(int score, int ply) {
    if (score >= SEARCH_MATE_SCORE - 100) return score + ply;
    if (score <= -SEARCH_MATE_SCORE + 100) return score - ply;
    return score;
}
static inline int score_from_tt(int score, int ply) {
    if (score >= SEARCH_MATE_SCORE - 100) return score - ply;
    if (score <= -SEARCH_MATE_SCORE + 100) return score + ply;
    return score;
}
// gravity: история не растёт без границ
static inline void search_hist_bonus(int& h, int bonus) {
    h += bonus - h * std::abs(bonus) / 16384;
}

// build 1.5: очки порядка ходов. Здесь же сохранена совместимость с тестами:
// плохие взятия < 20000, хорошие взятия > 30000 (run_see_tests).
// Почему: в 1.4 плохие взятия шли раньше тихих ходов, что тратило узлы на заведомо
// проигрышные обмены; теперь они в конце.
static int search_score_move(const Position& pos, Move mv, int ply, Move tt_move) {
    if (move_equal(mv, tt_move)) return SEARCH_SCORE_TT;
    const int pc = bb_piece_ref(pos, mv.from);
    const bool is_cap = (mv.flags & (F_CAPTURE | F_EP)) != 0;
    if (is_cap) {
        const int victim = (mv.flags & F_EP) ? make_pc(PAWN, color_of(pc) ^ 1) : bb_piece_ref(pos, mv.to);
        const int mvv = search_mvv_lva(pc, victim);
        if (mv.flags & F_PROMO) return SEARCH_SCORE_PROMO + mvv;
        search_telemetry.seeCalls++;
        const int see = search_see(pos, mv);
        if (see >= 0) return SEARCH_SCORE_GOOD_CAP + mvv;
        return SEARCH_SCORE_BAD_CAP + mvv;
    }
    if (mv.flags & F_PROMO) return SEARCH_SCORE_PROMO + search_piece_vals[piece_of(mv.promo)];
    if (ply < SEARCH_MAX_PLY) {
        if (move_equal(search_killer_moves[0][ply], mv)) return SEARCH_SCORE_KILLER1;
        if (move_equal(search_killer_moves[1][ply], mv)) return SEARCH_SCORE_KILLER2;
    }
    if (ply > 0 && search_piece_at[ply - 1] != 0 &&
        move_equal(search_countermove[search_piece_at[ply - 1]][search_move_at[ply - 1].to], mv))
        return SEARCH_SCORE_COUNTER;
    int h = search_history_moves[pc][mv.to];
    if (h > 5000) h = 5000;
    if (h < -5000) h = -5000;
    return h;
}

// Сортировка выбором: лучший ход выносится наверх по очереди (ленивая сортировка).
// build 1.5: сортировка выбором. Очки считаются заранее для всех ходов (SEE для взятий —
// один раз). Это проще, чем ленивая сортировка 1.4; на числе узлов не сказалось,
// nps немного ниже.
static void search_sort_moves(const Position& pos, MoveList& list, int ply, Move tt_move) {
    int scores[MAX_MOVES];
    for (int i = 0; i < list.count; ++i) scores[i] = search_score_move(pos, list.m[i], ply, tt_move);
    for (int i = 0; i < list.count; ++i) {
        int best = i;
        for (int j = i + 1; j < list.count; ++j)
            if (scores[j] > scores[best]) best = j;
        if (best != i) {
            std::swap(scores[i], scores[best]);
            std::swap(list.m[i], list.m[best]);
        }
    }
}

// build 1.5: генерация ходов без изменений (из 1.4). Perft 1..5 (startpos) и 1..4 (kiwipete) совпадают.
static void search_generate_moves(const Position& pos, MoveList& list, bool captures_only = false) {
    // ВАЖНО ДЛЯ СКОРОСТИ:
    // В поиске не надо заранее строить legal moves через generateLegalMoves(),
    // потому что generateLegalMoves() уже делает make/unmake для проверки шаха.
    // Потом search_negamax() всё равно снова делает search_make_move().
    // Поэтому тут генерируем псевдолегальные ходы, а нелегальные отсекаем
    // в search_make_move(). Это уменьшает двойную работу.
    MoveList pseudo;
    generateMoves(pos, pseudo);
    list.clear();
    for (int i = 0; i < pseudo.count; ++i) {
        // В quiescence обычно нужны взятия. Превращения тоже считаем forcing-ходами,
        // даже если это тихое превращение без взятия.
        if (!captures_only || (pseudo.m[i].flags & (F_CAPTURE | F_PROMO))) {
            list.add(pseudo.m[i].from, pseudo.m[i].to, pseudo.m[i].promo, pseudo.m[i].flags);
        }
    }
}
// search_make_move/search_unmake_move теперь опционально принимают
// RepetitionStack*. qsearch их не передаёт (nullptr по умолчанию) —
// репутационная история там никогда не запрашивалась и раньше, так что
// поведение не меняется. search_negamax передаёт указатель на свою копию
// стека явно.
// build 1.5: make/unmake без изменений (из 1.4).
static bool search_make_move(Position& pos, const Move& mv, Undo& undo, RepetitionStack* rep = nullptr) {
    const nnue::Delta nd = nnue::make_begin(pos, mv);
    const int us = pos.sideToMove;
    // Так как поиск теперь получает псевдолегальные ходы, рокировку надо
    // проверять отдельно ДО makeMove(): обычная проверка после makeMove()
    // ловит только шах на конечном поле, но не ловит рокировку из шаха
    // и не ловит проход короля через битое поле.
    if (mv.flags & F_CASTLE) {
        if (inCheck(pos, us))
            return false;
        int through = (mv.from + mv.to) / 2;
        if (isSquareAttacked(pos, through, us ^ 1))
            return false;
        if (isSquareAttacked(pos, mv.to, us ^ 1))
            return false;
    }
    makeMove(pos, mv, undo);
    if (inCheck(pos, us)) {
        unmakeMove(pos, mv, undo);
        return false;
    }
    if (rep) rep->push(pos.key);
    nnue::make_commit(nd, pos);
    return true;
}
static void search_unmake_move(Position& pos, const Move& mv, const Undo& undo, RepetitionStack* rep = nullptr) {
    if (rep) rep->pop();
    unmakeMove(pos, mv, undo);
    nnue::on_unmake();
}
static bool is_draw_by_rules(const Position& pos) {
    if (pos.halfmove >= 100) return true;
    // build1.3: bitboard counts. Kings are excluded, as before.
    const int kings = bb_popcountll(pos.bbPieces[WHITE][KING]) + bb_popcountll(pos.bbPieces[BLACK][KING]);
    const int piece_count = bb_popcountll(pos.bbOccupiedAll) - kings;
    const int knight_count = bb_popcountll(pos.bbPieces[WHITE][KNIGHT] | pos.bbPieces[BLACK][KNIGHT]);
    const int bishop_count = bb_popcountll(pos.bbPieces[WHITE][BISHOP] | pos.bbPieces[BLACK][BISHOP]);
    if (piece_count == 0) return true;
    if (piece_count == 1 && (knight_count == 1 || bishop_count == 1)) return true;
    return false;
}
// Теперь принимает явный стек вместо обращения к глобалу.
static bool is_repetition(const Position& pos, const RepetitionStack& rep) {
    int end = rep.count - 3;
    // FIX (build0.1): ближайший индекс, который в принципе может повторять
    // текущую позицию — это позиция сразу ПОСЛЕ последнего необратимого хода,
    // т.е. (последний индекс) - pos.halfmove = (rep.count - 1) - pos.halfmove.
    // Старая формула (rep.count - pos.halfmove) была сдвинута на единицу и
    // молча исключала эту граничную позицию (в т.ч. root) из сканирования.
    int limit = rep.count - pos.halfmove - 1;
    if (limit < 0) limit = 0;
    for (int i = end; i >= limit; i -= 2) {
        if (rep.keys[i] == pos.key)
            return true;
    }
    return false;
}
// ---- qsearch (build 1.5) ----
// На первом уровне за горизонтом (qdepth==0) разрешены тихие шахи: это закрывает
// горизонтные мат-в-1 после LMR без отключения самой редукции.
// ---- qsearch (build 1.5) ----
// Изменено относительно 1.4:
//   * qdepth (глубина внутри qsearch) отделён от ply. Тихие шахи разрешены только при
//     qdepth == 0, т.е. на первом уровне за горизонтом. Это закрывает мат-в-1 на горизонте.
//     Почему: без этого мат-в-2 позиции 15 не находится даже без RFP (проверено: 15/16).
//   * Дельта-отсечение и SEE-отсечение взятий оставлены, как в 1.4.
//   * Вне шаха: стоячая оценка (stand pat). В шахе: все уклонения, мат, если нет ходов.
static int search_qsearch_impl(Position& pos, int alpha, int beta, int ply, int qdepth) {
    if (search_limits.stopped) return 0;
    search_telemetry.qnodes++;
    if (!search_enter_node()) return 0;
    const bool in_check = inCheck(pos, pos.sideToMove);
    int stand_pat = -SEARCH_INFINITE_SCORE;
    if (!in_check) {
        stand_pat = evaluate(pos);
        if (stand_pat >= beta) return stand_pat;
        if (stand_pat > alpha) alpha = stand_pat;
        if (qdepth >= SEARCH_QS_MAX_PLY || ply >= SEARCH_MAX_PLY - 1) return alpha;
    } else if (qdepth >= SEARCH_QS_MAX_PLY + 4 || ply >= SEARCH_MAX_PLY - 1) {
        return evaluate(pos);
    }

    MoveList list;
    const bool full = in_check || qdepth == 0;
    search_generate_moves(pos, list, !full);
    search_sort_moves(pos, list, ply, Move{ 0, 0, 0, 0 });

    int best = in_check ? -SEARCH_INFINITE_SCORE : stand_pat;
    int legal = 0;
    int npm = -1;
    for (int i = 0; i < list.count; ++i) {
        const Move mv = list.m[i];
        const int pc = bb_piece_ref(pos, mv.from);
        const bool capture = (mv.flags & (F_CAPTURE | F_EP)) != 0;
        const bool promo = (mv.flags & F_PROMO) != 0;
        const bool quiet = !capture && !promo;

        if (!in_check && capture && !promo) {
            if (npm < 0) {
                const int us = pos.sideToMove;
                npm = 0;
                for (int t = KNIGHT; t <= QUEEN; ++t)
                    npm += bb_popcountll(pos.bbPieces[us][t]) * search_piece_vals[t];
            }
            const int victim = (mv.flags & F_EP) ? search_piece_vals[PAWN] : search_piece_vals[piece_of(bb_piece_ref(pos, mv.to))];
            if (npm > SEARCH_DELTA_NPM_MIN && stand_pat + victim + SEARCH_DELTA_MARGIN <= alpha) {
                search_telemetry.deltaPruned++;
                continue;
            }
            search_telemetry.seeCalls++;
            if (search_see(pos, mv) < 0) {
                search_telemetry.seePruned++;
                continue;
            }
        }

        Undo undo;
        if (!search_make_move(pos, mv, undo)) continue;
        if (quiet && !in_check && !inCheck(pos, pos.sideToMove)) {
            search_unmake_move(pos, mv, undo);
            continue;
        }
        if (quiet && !in_check) search_telemetry.qsChecks++;
        ++legal;

        search_move_at[ply] = mv;
        search_piece_at[ply] = pc;
        const int score = -search_qsearch_impl(pos, -beta, -alpha, ply + 1, qdepth + 1);
        search_unmake_move(pos, mv, undo);
        if (search_limits.stopped) return 0;

        if (score > best) best = score;
        if (score > alpha) alpha = score;
        if (score >= beta) return score;
    }
    if (in_check && legal == 0) return -SEARCH_MATE_SCORE + ply;
    return best;
}
static int search_qsearch(Position& pos, int alpha, int beta, int ply) {
    return search_qsearch_impl(pos, alpha, beta, ply, 0);
}

// ---- negamax (build 1.5) ----
// Порядок: остановка -> mate-distance pruning -> draw -> продление за шах (с лимитом
// на путь) -> TT -> RFP/null/futility -> ходы: TT, взятия, killer, countermove,
// история; PVS; LMR (только тихие, не шахи); LMP на малых глубинах.
// ---- negamax (build 1.5) ----
// Порядок шагов (важен, менять осторожно):
//   1. остановка по лимиту, ply-лимит;
//   2. mate-distance pruning (новое в 1.5): мат дальше уже найденного не нужен;
//   3. ничья по правилам/повторению (как в 1.4);
//   4. продление за шах с лимитом на путь (новое: в 1.4 без лимита);
//   5. TT-отсечение (без него в PV-узлах);
//   6. null move (R = 3 + depth/4; в 1.4 R = 3 фиксировано);
//   7. futility для тихих ходов на depth <= 2;
//   8. ходы: PVS, LMR (с ре-серчем), LMP, PV-таблица, история/killer/countermove при отсечении.
// RFP (reverse futility) в 1.5 удалён — см. комментарий в шаге 5 ниже.
static int search_negamax(Position& pos, int alpha, int beta, int depth, int ply, RepetitionStack& rep, bool do_null = true) {
    if (search_limits.stopped) return 0;
    if (ply >= SEARCH_MAX_PLY - 1) return search_qsearch(pos, alpha, beta, ply);
    search_pv_table[ply].length = 0;
    if (!search_enter_node()) return 0;

    // build 1.5 (новое): mate-distance pruning. Если даже лучший мат для нас не лучше
    // alpha (или худший мат для соперника не хуже beta), узел отсекается.
    // Почему: в 1.4 такого не было, и поиск перебирал явно проигранные мат-линии.
    if (ply > 0) {
        const int mdA = std::max(alpha, -SEARCH_MATE_SCORE + ply);
        const int mdB = std::min(beta, SEARCH_MATE_SCORE - ply - 1);
        if (mdA >= mdB) {
            search_telemetry.mateDistanceCutoffs++;
            return mdA;
        }
        alpha = mdA;
        beta = mdB;
    }

    const bool in_check = inCheck(pos, pos.sideToMove);
    if (ply > 0 && (is_draw_by_rules(pos) || is_repetition(pos, rep))) {
        if (!in_check) return 0;
        MoveList legal;
        generateLegalMoves(pos, legal);
        return legal.count == 0 ? -SEARCH_MATE_SCORE + ply : 0;
    }

    // build 1.5: продление за шах с лимитом SEARCH_CHECK_EXT_PATH_MAX на путь.
    // Почему: в 1.4 продление было безлимитным, и на позиции 11 это дало d9 за 180 с
    // (290 млн узлов) против d11 за 5 с без продления. Лимит 4 сохраняет продление
    // для коротких матовых атак и не даёт взрыва.
    const int parent_ext = ply > 0 ? search_ext_used[ply - 1] : 0;
    int ext = 0;
    if (in_check && parent_ext < SEARCH_CHECK_EXT_PATH_MAX) {
        ext = 1;
        search_telemetry.checkExtensions++;
    }
    search_ext_used[ply] = parent_ext + ext;
    depth += ext;
    if (depth <= 0) return search_qsearch(pos, alpha, beta, ply);

    const bool pv_node = (beta - alpha) > 1;
    const uint64_t hash = pos.key;
    search_telemetry.ttProbes++;
    Move tt_move{ 0, 0, 0, 0 };
    {
        const SearchTTEntry& e = search_tt_entry(hash);
        if (e.key == hash) {
            search_telemetry.ttHits++;
            tt_move = e.move;
            if (ply > 0 && !pv_node && e.depth >= depth) {
                const int s = score_from_tt(e.score, ply);
                if (e.flag == 0 || (e.flag == 1 && s <= alpha) || (e.flag == 2 && s >= beta)) {
                    search_telemetry.ttCutoffs++;
                    return s;
                }
            }
        }
    }

    const bool can_prune = !in_check && !pv_node && ply > 0 && std::abs(beta) < SEARCH_MATE_SCORE - 200;
    int static_eval = 0;
    bool have_eval = false;

    // build 1.5: reverse futility pruning отключён (RFP удалён из кода).
    // reverse futility pruning в build 1.5 отключён: при статической оценке, близкой к beta,
    // RFP на глубине 1 отсекает узел соперника, в котором мат в 2 хода (поз. 15 мат-в-2)
    // виден только при полном поиске. Мат в 2 на 16 позициях: 15/16 с RFP, 16/16 без него.

    // null move. build 1.5: R = 3 + depth/4 (в 1.4 фиксировано 3).
    // Почему: на глубоких узлах более агрессивная редукция дешевле, чем полный ход соперника.
    if (can_prune && do_null && depth >= 3) {
        if (!have_eval) { static_eval = evaluate(pos); have_eval = true; }
        const int us = pos.sideToMove;
        const bool major = (pos.bbOccupied[us] & ~(pos.bbPieces[us][PAWN] | pos.bbPieces[us][KING])) != 0;
        if (major && static_eval >= beta) {
            const int R = 3 + depth / 4;
            const uint64_t key_before = pos.key;
            const int saved_ep = pos.ep;
            const int old_ep = hash_ep_square(pos);
            pos.sideToMove ^= 1;
            pos.key ^= zSide;
            pos.ep = NO_SQ;
            const int new_ep = hash_ep_square(pos);
            pos.key ^= zEp[old_ep] ^ zEp[new_ep];
            search_telemetry.nullTries++;
            search_move_at[ply] = Move{ 0, 0, 0, 0 };
            search_piece_at[ply] = 0;
            const int score = -search_negamax(pos, -beta, -beta + 1, depth - 1 - R, ply + 1, rep, false);
            pos.ep = saved_ep;
            pos.sideToMove ^= 1;
            pos.key = key_before;
            if (search_limits.stopped) return 0;
            if (score >= beta) {
                search_telemetry.nullCutoffs++;
                return score >= SEARCH_MATE_SCORE - 100 ? beta : score;
            }
        }
    }

    // futility (как в 1.4): на depth <= 2 тихие ходы без шаха, которые не могут поднять
    // оценку выше alpha, пропускаются.
    bool futile = false;
    if (can_prune && depth <= 2) {
        if (!have_eval) { static_eval = evaluate(pos); have_eval = true; }
        futile = static_eval + SEARCH_FUTILITY_MARGIN[depth] <= alpha;
    }

    MoveList list;
    search_generate_moves(pos, list);
    search_sort_moves(pos, list, ply, tt_move);

    // quiets[] — тихие ходы, опробованные до отсечения; им даётся malus в истории.
    Move quiets[64];
    int nquiets = 0;
    const int alpha_orig = alpha;
    int best_score = -SEARCH_INFINITE_SCORE;
    Move best_move{ 0, 0, 0, 0 };
    int legal = 0;

    for (int i = 0; i < list.count; ++i) {
        const Move mv = list.m[i];
        const int pc = bb_piece_ref(pos, mv.from);
        const bool capture = (mv.flags & (F_CAPTURE | F_EP)) != 0;
        const bool promo = (mv.flags & F_PROMO) != 0;
        const bool quiet = !capture && !promo;

        Undo undo;
        if (!search_make_move(pos, mv, undo, &rep)) continue;
        const bool gives_check = inCheck(pos, pos.sideToMove);
        ++legal;

        if (quiet && !in_check && !gives_check && legal > 1) {
            if (futile) {
                search_telemetry.futilityPruned++;
                search_unmake_move(pos, mv, undo, &rep);
                continue;
            }
            // build 1.5 (новое): late move pruning. На depth <= 3 поздние тихие ходы
            // (номер > 3 + depth^2) не рассматриваются. Не применяется к шахам и PV.
            if (!pv_node && depth <= 3 && legal > 3 + depth * depth) {
                search_telemetry.lmpPruned++;
                search_unmake_move(pos, mv, undo, &rep);
                continue;
            }
        }

        search_move_at[ply] = mv;
        search_piece_at[ply] = pc;
        const int newDepth = depth - 1;
        int score;
        // build 1.5: первый легальный ход — полное окно. Остальные — PVS (нулевое окно),
        // при успехе — полный ре-серч. Почему: PVS дешевле при хорошем порядке ходов.
        if (legal == 1) {
            score = -search_negamax(pos, -beta, -alpha, newDepth, ply + 1, rep);
        } else {
            // LMR (build 1.5, новое): тихие ходы, не шахи, depth >= 3.
            // r уменьшается в PV и для killer-ходов; глубина ребёнка >= 1.
            // Почему такой порог: LMR на глубине 3 с r=2 уводил мат-в-2 в qsearch без
            // поиска (поз. 15); зажим r <= newDepth-1 и отключённый RFP это закрывают.
            int r = 0;
            if (depth >= 3 && quiet && !in_check && !gives_check) {
                r = search_lmr_table[std::min(depth, 63)][std::min(legal, 63)];
                if (pv_node) r--;
                if (ply < SEARCH_MAX_PLY &&
                    (move_equal(search_killer_moves[0][ply], mv) || move_equal(search_killer_moves[1][ply], mv)))
                    r--;
                if (r < 0) r = 0;
                if (r > newDepth - 1) r = newDepth - 1;
                if (r < 0) r = 0;
            }
            if (r > 0) search_telemetry.lmrReductions++;
            score = -search_negamax(pos, -alpha - 1, -alpha, newDepth - r, ply + 1, rep);
            if (r > 0 && score > alpha) {
                search_telemetry.lmrResearches++;
                score = -search_negamax(pos, -alpha - 1, -alpha, newDepth, ply + 1, rep);
            }
            if (score > alpha && score < beta) {
                search_telemetry.pvsResearches++;
                score = -search_negamax(pos, -beta, -alpha, newDepth, ply + 1, rep);
            }
        }
        search_unmake_move(pos, mv, undo, &rep);
        if (search_limits.stopped) return 0;

        if (score > best_score) {
            best_score = score;
            best_move = mv;
        }
        if (score > alpha) {
            alpha = score;
            search_pv_table[ply].moves[0] = mv;
            const int cl = (ply + 1 < SEARCH_MAX_PLY) ? search_pv_table[ply + 1].length : 0;
            for (int j = 0; j < cl && j + 1 < SEARCH_MAX_PLY; ++j)
                search_pv_table[ply].moves[j + 1] = search_pv_table[ply + 1].moves[j];
            search_pv_table[ply].length = 1 + std::min(cl, SEARCH_MAX_PLY - 2);
        }
        if (score >= beta) {
            // build 1.5: при отсечении по тихому ходу обновляются killer, countermove и
            // история: ход получает бонус, опробованные ранее тихие — штраф (malus).
            // В 1.4 штрафа не было, а история росла без границ.
            search_telemetry.betaCutoffs++;
            if (legal == 1) search_telemetry.firstMoveCutoffs++;
            if (quiet && ply < SEARCH_MAX_PLY) {
                if (move_equal(search_killer_moves[0][ply], mv)) search_telemetry.killerCutoffs++;
                if (ply > 0 && search_piece_at[ply - 1] != 0 &&
                    move_equal(search_countermove[search_piece_at[ply - 1]][search_move_at[ply - 1].to], mv))
                    search_telemetry.countermoveCutoffs++;
                if (!move_equal(search_killer_moves[0][ply], mv)) {
                    search_killer_moves[1][ply] = search_killer_moves[0][ply];
                    search_killer_moves[0][ply] = mv;
                }
                const int bonus = std::min(depth * depth, 1200);
                search_hist_bonus(search_history_moves[pc][mv.to], bonus);
                for (int q = 0; q < nquiets; ++q) {
                    const int qpc = bb_piece_ref(pos, quiets[q].from);
                    search_hist_bonus(search_history_moves[qpc][quiets[q].to], -bonus);
                }
                if (ply > 0 && search_piece_at[ply - 1] != 0)
                    search_countermove[search_piece_at[ply - 1]][search_move_at[ply - 1].to] = mv;
            }
            SearchTTEntry& e = search_tt_entry(hash);
            e.key = hash;
            e.move = mv;
            e.depth = (int8_t)depth;
            e.flag = 2;
            e.score = score_to_tt(score, ply);
            return score;
        }
        if (quiet && nquiets < 64) quiets[nquiets++] = mv;
    }

    // build 1.5: сохранение в TT. Флаг: 0 = точная, 1 = верхняя граница (fail-low).
    // Очки мата переводятся в "относительные к узлу" (score_to_tt).
    if (legal == 0) return in_check ? -SEARCH_MATE_SCORE + ply : 0;

    SearchTTEntry& e = search_tt_entry(hash);
    e.key = hash;
    e.move = best_move;
    e.depth = (int8_t)depth;
    e.flag = best_score <= alpha_orig ? 1 : 0;   // 1 = UPPER, 0 = EXACT
    e.score = score_to_tt(best_score, ply);
    return best_score;
}

// ============================================================================
// build0.2 P0 TESTS
//
// Постоянные регрессионные проверки для P0-исправлений. Не заменяют внешнее
// тестирование, но гарантируют, что базовые инварианты (мат важнее draw-return,
// draw rules не считают K+N vs K+N мёртвой позицией) не откатятся незаметно
// в будущих правках.
// ============================================================================
// build 1.5: тесты из 1.4 без изменений. Используют search_negamax с той же сигнатурой.
static void run_p0_tests() {
    int pass = 0;
    int fail = 0;
    auto result = [&](bool ok, const char* name) {
        if (ok) {
            ++pass;
            std::printf("info string P0 %s OK\n", name);
        }
        else {
            ++fail;
            std::printf("info string P0 %s FAIL\n", name);
        }
        };
    Position p;
    RepetitionStack rep;
    bool ok = parse_fen(
        p,
        "7k/5KQ1/8/8/8/8/8/8 b - - 100 60"
    );
    rep.count = 0;
    if (ok) rep.push(p.key);
    search_limits.stopped = false;
    search_limits.check_time = false;
    search_limits.check_nodes = false;
    search_limits.node_limit = 0;
    search_limits.nodes = 0;
    int mate_score = ok
        ? search_negamax(
            p,
            -SEARCH_INFINITE_SCORE,
            SEARCH_INFINITE_SCORE,
            1,
            1,
            rep,
            false
        )
        : 0;
    result(
        ok && mate_score == -SEARCH_MATE_SCORE + 1,
        "mate_before_50move"
    );
    ok = parse_fen(
        p,
        "7k/8/8/8/8/8/6R1/6K1 b - - 100 60"
    );
    rep.count = 0;
    if (ok) rep.push(p.key);
    search_limits.stopped = false;
    search_limits.check_time = false;
    search_limits.check_nodes = false;
    search_limits.node_limit = 0;
    search_limits.nodes = 0;
    int draw_score = ok
        ? search_negamax(
            p,
            -SEARCH_INFINITE_SCORE,
            SEARCH_INFINITE_SCORE,
            1,
            1,
            rep,
            false
        )
        : 1;
    result(ok && draw_score == 0, "fifty_move_draw");
    ok = parse_fen(
        p,
        "7k/5n2/8/8/8/8/2N5/K7 w - - 0 1"
    );
    result(
        ok && !is_draw_by_rules(p),
        "knight_vs_knight_not_dead"
    );
    ok = parse_fen(
        p,
        "7k/8/8/8/8/8/2N5/K7 w - - 0 1"
    );
    result(
        ok && is_draw_by_rules(p),
        "knight_vs_king_dead"
    );
    std::printf(
        "info string P0 TESTS DONE pass %d fail %d\n",
        pass,
        fail
    );
    std::fflush(stdout);
}
// ============================================================================
// build0.3 EP TESTS
//
// Regression coverage for canonical en-passant Zobrist hashing (Level A).
// See hash_ep_square() for the exact normalization contract. These tests are
// intentionally independent of search/TT and only touch parse_fen,
// computeHash, makeMove, generateMoves/generateLegalMoves and
// hash_ep_square directly, plus a standalone replica of the null-move key
// patch pattern used inside search_negamax().
// ============================================================================
static void run_ep_tests() {
    int pass = 0;
    int fail = 0;
    auto result = [&](bool ok, const char* name) {
        if (ok) {
            ++pass;
            std::printf("info string EP %s OK\n", name);
        }
        else {
            ++fail;
            std::printf("info string EP %s FAIL\n", name);
        }
        };
    // A. double_without_candidate_canonical
    {
        Position p, q; Move m{}; Undo u;
        bool ok = parse_fen(p, "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1") && getmv(p, "e2e4", m);
        if (ok) {
            makeMove(p, m, u);
            ok = ok && (p.key == computeHash(p));
            ok = ok && parse_fen(q, "4k3/8/8/8/4P3/8/8/4K3 b - - 0 1");
            ok = ok && (p.key == q.key);
        }
        result(ok, "double_without_candidate_canonical");
    }
    // B. double_with_candidate_hashed
    {
        Position p, q, r; Move m{}; Undo u;
        bool ok = parse_fen(p, "4k3/8/8/8/3p4/8/4P3/4K3 w - - 0 1") && getmv(p, "e2e4", m);
        if (ok) {
            makeMove(p, m, u);
            ok = ok && (p.key == computeHash(p));
            ok = ok && parse_fen(q, "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1");
            ok = ok && (p.key == q.key);
            ok = ok && parse_fen(r, "4k3/8/8/8/3pP3/8/8/4K3 b - - 0 1");
            ok = ok && (p.key != r.key);
        }
        result(ok, "double_with_candidate_hashed");
    }
    // C. malformed_ep_no_move
    {
        const char* fen = "4k3/8/8/4P3/8/8/8/4K3 w - d6 0 1";
        bool ok = !has(fen, "e5d6") && !has_legal(fen, "e5d6");
        result(ok, "malformed_ep_no_move");
    }
    // D. malformed_ep_not_hashed
    {
        Position p, q;
        bool ok = parse_fen(p, "4k3/8/8/4P3/8/8/8/4K3 w - d6 0 1")
            && parse_fen(q, "4k3/8/8/4P3/8/8/8/4K3 w - - 0 1");
        ok = ok && (p.key == q.key) && (hash_ep_square(p) == NO_SQ);
        result(ok, "malformed_ep_not_hashed");
    }
    // E. pinned_ep_level_a_boundary
    // Documented Level A limitation: hash_ep_square() only checks geometry
    // and captured-pawn existence, not pin legality. The capture itself is
    // still correctly rejected as illegal by generateLegalMoves() because it
    // would expose the White king to the rook on a5. This is intentional and
    // not considered a Level A failure.
    {
        const char* fen = "8/8/8/r2pP2K/8/8/8/8 w - d6 0 1";
        Position p;
        bool ok = parse_fen(p, fen);
        ok = ok && (hash_ep_square(p) == sq_of("d6"));
        ok = ok && !has_legal(fen, "e5d6");
        result(ok, "pinned_ep_level_a_boundary");
    }
    // F. null_ep_roundtrip
    // Replicates the exact key-patch pattern used in search_negamax()'s
    // manual null move, standalone (no search/TT/thread involvement).
    {
        Position p;
        bool ok = parse_fen(p, "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1");
        if (ok) {
            const uint64_t original_key = p.key;
            ok = ok && (original_key == computeHash(p));
            const uint64_t key_before_null = p.key;
            const int saved_ep = p.ep;
            const int old_ep_component = hash_ep_square(p);
            p.sideToMove ^= 1;
            p.key ^= zSide;
            p.ep = NO_SQ;
            const int new_ep_component = hash_ep_square(p);
            p.key ^= zEp[old_ep_component] ^ zEp[new_ep_component];
            ok = ok && (p.key == computeHash(p));
            p.ep = saved_ep;
            p.sideToMove ^= 1;
            p.key = key_before_null;
            ok = ok && (p.key == original_key);
            ok = ok && (p.key == computeHash(p));
        }
        else {
            ok = false;
        }
        result(ok, "null_ep_roundtrip");
    }
    std::printf("info string EP TESTS DONE pass %d fail %d\n", pass, fail);
    std::fflush(stdout);
}
// ============================================================================
// build0.8 SEE TESTS
//
// Regression coverage for search_see() (intermediate variant B: swap with
// x-ray, no pins). Pure unit tests: parse_fen + generateMoves + search_see,
// no search/TT/thread involvement. Expected values use search_piece_vals
// (P=100, N=320, B=330, R=500, Q=900, K=20000).
// ============================================================================
static void run_see_tests() {
    int pass = 0;
    int fail = 0;
    auto result = [&](bool ok, const char* name) {
        if (ok) {
            ++pass;
            std::printf("info string SEE %s OK\n", name);
        }
        else {
            ++fail;
            std::printf("info string SEE %s FAIL\n", name);
        }
        };
    auto see_of = [](const char* fen, const char* mv, bool& ok, int& out) {
        Position p; Move m{};
        ok = parse_fen(p, fen) && getmv(p, mv, m);
        if (ok) out = search_see(p, m);
        };
    bool ok; int v;
    // 1. PxP, target undefended: clean +100.
    see_of("7k/8/8/3p4/4P3/8/8/7K w - - 0 1", "e4d5", ok, v);
    result(ok && v == 100, "pxp_undefended_plus100");
    // 2. PxP, target defended by a pawn: even trade, exactly 0.
    see_of("7k/8/2p5/3p4/4P3/8/8/7K w - - 0 1", "e4d5", ok, v);
    result(ok && v == 0, "pxp_defended_zero");
    // 3. QxP, pawn defended by a pawn: queen is lost for a pawn, -800.
    see_of("7k/8/2p5/3p4/8/8/8/3Q3K w - - 0 1", "d1d5", ok, v);
    result(ok && v == -800, "qxp_defended_minus800");
    // 4. X-RAY: white rook battery d1+d2 vs black rooks d5+d8.
    //    RxR, RxR, RxR ends +500 — but ONLY if the d1 rook behind the d2
    //    rook is seen through the vacated square. A no-x-ray SEE returns 0.
    see_of("3r3k/8/8/3r4/8/8/3R4/3R3K w - - 0 1", "d2d5", ok, v);
    result(ok && v == 500, "xray_rook_battery_plus500");
    // 5. King "captures" a defended pawn: recapture of the king folds to a
    //    huge loss, so SEE must be negative (king value treatment).
    see_of("7k/8/5p2/4p3/3K4/8/8/8 w - - 0 1", "d4e5", ok, v);
    result(ok && v < 0, "king_takes_defended_negative");
    // 6. En passant: captured pawn sits BESIDE the target square, +100.
    see_of("7k/8/8/3pP3/8/8/8/7K w - d6 0 1", "e5d6", ok, v);
    result(ok && v == 100, "ep_capture_plus100");
    // 7. Ordering integration: a losing capture must be scored below
    //    killers (i.e. below 20000), a winning one above 30000.
    {
        Position p; Move m{};
        bool o2 = parse_fen(p, "7k/8/2p5/3p4/8/8/8/3Q3K w - - 0 1")
            && getmv(p, "d1d5", m);
        int sc = o2 ? search_score_move(p, m, 0, Move{ 0,0,0,0 }) : 999999;
        result(o2 && sc < 20000, "ordering_bad_capture_below_killers");
    }
    {
        Position p; Move m{};
        bool o2 = parse_fen(p, "7k/8/8/3p4/4P3/8/8/7K w - - 0 1")
            && getmv(p, "e4d5", m);
        int sc = o2 ? search_score_move(p, m, 0, Move{ 0,0,0,0 }) : -1;
        result(o2 && sc > 30000, "ordering_good_capture_above_killers");
    }
    // Stage 6 differential test: every legal capture in representative
    // positions must produce the same value through the new bitboard SEE and
    // the preserved 0x88 reference implementation.
    bool differential = true;
    const char* diff_fens[] = {
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "3r3k/8/8/3r4/8/8/3R4/3R3K w - - 0 1",
        "7k/8/2p5/3p4/8/8/8/3Q3K w - - 0 1",
        "7k/8/8/3pP3/8/8/8/7K w - d6 0 1",
        "8/P6p/1p6/3pPp2/8/8/p6P/4K2k w - f6 0 1",
    };
    int compared = 0;
    for (const char* fen : diff_fens) {
        Position p;
        if (!parse_fen(p, fen)) { differential = false; break; }
        MoveList ml;
        generateLegalMoves(p, ml);
        for (int i = 0; i < ml.count; ++i) {
            if (!(ml.m[i].flags & F_CAPTURE)) continue;
            const int old_value = search_see_0x88(p, ml.m[i]);
            const int new_value = search_see(p, ml.m[i]);
            ++compared;
            if (old_value != new_value) {
                differential = false;
                break;
            }
        }
        if (!differential) break;
    }
    result(differential && compared > 0, "bitboard_see_differential");
    std::printf("info string SEE differential_captures %d\n", compared);
    std::printf("info string SEE TESTS DONE pass %d fail %d\n", pass, fail);
    std::fflush(stdout);
}
static bool search_is_null_move_local(const Move& m) {
    return m.from == 0 && m.to == 0 && m.promo == 0 && m.flags == 0;
}

// ---- вывод info (build 1.5): mate в ходах, PV из треугольной таблицы ----
// ---- ВРЕМЕННАЯ ДИАГНОСТИКА UCI (build 1.5, отладка зависания в Arena) ----
// Пишет в файл uci_log.txt рядом с exe: каждую входящую строку UCI (IN),
// каждый ответ bestmove/readyok/uciok (OUT), каждую итерацию глубины
// (info depth), причину выхода (EXIT/TERMINATE) и исключение Windows (EXC).
// Выключение: UCI_TRACE_ENABLED 0. Внутри дерева поиска ничего не пишется;
// стоимость — одна строка с fflush на команду UCI и на итерацию глубины.
#ifndef UCI_TRACE_ENABLED
#define UCI_TRACE_ENABLED 1
#endif
#if UCI_TRACE_ENABLED
#include <cstdlib>
#include <exception>
#include <mutex>
namespace uci_trace {
static FILE* g_file = nullptr;
static std::mutex g_mutex;
static void log_line(const char* tag, const char* s) {
    if (!g_file) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    const long long ms = (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    std::fprintf(g_file, "%lld %s %s\n", ms, tag, s);
    std::fflush(g_file);
}
#ifdef _WIN32
// Обработчик без блокировки: при падении мьютекс мог быть занят. EXCEPTION_CONTINUE_SEARCH
// сохраняет штатное поведение Windows (сообщение об ошибке и запись в журнал событий).
static LONG WINAPI exc_filter(EXCEPTION_POINTERS* ep) {
    if (g_file && ep && ep->ExceptionRecord) {
        std::fprintf(g_file, "EXC code 0x%08lX address %p\n",
            (unsigned long)ep->ExceptionRecord->ExceptionCode,
            ep->ExceptionRecord->ExceptionAddress);
        std::fflush(g_file);
    }
    return EXCEPTION_CONTINUE_SEARCH;
}
#endif
static void start(const char* argv0) {
    const std::string a = argv0 ? argv0 : "";
    const size_t p = a.find_last_of("/\\");
    const std::string dir = (p == std::string::npos) ? std::string() : a.substr(0, p + 1);
    const std::string path = dir + "uci_log.txt";
    g_file = std::fopen(path.c_str(), "a");
    log_line("START", ("argv0=" + a + " log=" + path).c_str());
    std::set_terminate([]() {
        if (g_file) { std::fprintf(g_file, "TERMINATE std::terminate called\n"); std::fflush(g_file); }
        std::abort();
    });
    std::atexit([]() { log_line("EXIT", "normal exit (atexit)"); });
#ifdef _WIN32
    SetUnhandledExceptionFilter(exc_filter);
#endif
}
} // namespace uci_trace
#define TRACE_START(a) uci_trace::start(a)
#define TRACE_OUT(tag, s) uci_trace::log_line(tag, s)
#define TRACE_IN(s) uci_trace::log_line("IN", s)
#else
#define TRACE_START(a) ((void)0)
#define TRACE_OUT(tag, s) ((void)0)
#define TRACE_IN(s) ((void)0)
#endif
// build 1.5: исправлено. В 1.4 функция всегда печатала "score cp", даже для мата
// (мат выглядел как +28997 сантипешек). Теперь: "score mate N" в полных ходах.
static void search_print_info(int depth, int score) {
    long long elapsed = search_now() - search_limits.start_time;
    if (elapsed < 0) elapsed = 0;
    const long long nodes = search_limits.nodes;
    const long long nps = elapsed > 0 ? nodes * 1000 / elapsed : nodes * 1000;
#if UCI_TRACE_ENABLED
    { char tb[160]; std::snprintf(tb, sizeof(tb), "depth %d score %d nodes %lld time %lld",
        depth, score, nodes, elapsed); TRACE_OUT("OUT info", tb); }
#endif
    if (std::abs(score) >= SEARCH_MATE_SCORE - 100) {
        const int plies = SEARCH_MATE_SCORE - std::abs(score);
        const int moves = (plies + 1) / 2;
        std::printf("info depth %d score mate %d nodes %lld nps %lld time %lld pv",
            depth, score > 0 ? moves : -moves, nodes, nps, elapsed);
    } else {
        std::printf("info depth %d score cp %d nodes %lld nps %lld time %lld pv",
            depth, score, nodes, nps, elapsed);
    }
    const SearchPVTable& pv = search_pv_table[0];
    for (int i = 0; i < pv.length; ++i) {
        const Move& m = pv.moves[i];
        if (search_is_null_move_local(m)) break;
        char buf[8];
        move_str(m, buf);
        std::printf(" %s", buf);
    }
    std::printf("\n");
    std::fflush(stdout);
}

// ---- итеративное углубление с окном аспирации (build 1.5) ----
// ---- итеративное углубление (build 1.5) ----
// Изменено:
//   * аспирация: окно ±25 вокруг прошлой оценки (с глубины 5), при провале окно
//     расширяется удвоением; полное окно — только если провал повторяется;
//   * остановка по мату: если мат найден и он не длиннее текущей глубины, поиск
//     завершается (как в 1.4, но теперь и для проигрышного мата);
//   * PV берётся из треугольной таблицы; лучший ход — PV[0] текущей итерации;
//   * tmFailLowExtends (продление по неудачному окну времени) в 1.5 не переносилось,
//     поэтому всегда 0; мягкий лимит времени работает как в 1.4.
Move search(Position& pos, int max_depth, int time_limit_ms, RepetitionStack rep = RepetitionStack{}, uint64_t node_limit = 0) {
    search_telemetry = SearchTelemetry{};
    search_limits.start_time = search_now();
    search_limits.time_limit = time_limit_ms;
    search_limits.check_time = time_limit_ms > 0;
    search_limits.node_limit = node_limit;
    search_limits.check_nodes = node_limit > 0;
    search_limits.nodes = 0;
    search_limits.stopped = false;
    search_stop_reason = SEARCH_STOP_NONE;
    g_search_initialized = true;
    std::memset(search_killer_moves, 0, sizeof(search_killer_moves));
    std::memset(search_history_moves, 0, sizeof(search_history_moves));
    std::memset(search_countermove, 0, sizeof(search_countermove));
    std::memset(search_move_at, 0, sizeof(search_move_at));
    std::memset(search_piece_at, 0, sizeof(search_piece_at));
    std::memset(search_ext_used, 0, sizeof(search_ext_used));
    for (auto& p : search_pv_table) p.length = 0;
    search_init_lmr();
    nnue::g_ply = 0;

    if (rep.count == 0 || rep.keys[rep.count - 1] != pos.key) rep.push(pos.key);

    MoveList root_moves;
    generateLegalMoves(pos, root_moves);
    if (root_moves.count == 0) {
        search_set_reason_if_none(SEARCH_STOP_DEPTH);
        return Move{ 0, 0, 0, 0 };
    }
    Move best_move = root_moves.m[0];

    const long long tm_soft = g_tm_soft_ms;
    g_tm_soft_ms = 0;
    int last_score = 0;
    bool have_last = false;

    for (int d = 1; d <= max_depth; ++d) {
        if (tm_soft > 0 && d > 1 && search_now() - search_limits.start_time >= tm_soft) {
            search_telemetry.tmSoftStops++;
            search_set_reason_if_none(SEARCH_STOP_TM_SOFT);
            break;
        }
        int score;
        if (have_last && d >= 5 && std::abs(last_score) < SEARCH_MATE_SCORE - 200) {
            int delta = 25;
            int alpha = last_score - delta;
            int beta = last_score + delta;
            for (;;) {
                score = search_negamax(pos, alpha, beta, d, 0, rep);
                if (search_limits.stopped) break;
                if (score > alpha && score < beta) break;
                search_telemetry.aspirationResearches++;
                if (score <= alpha)
                    alpha = (alpha <= -SEARCH_INFINITE_SCORE + 2 * delta) ? -SEARCH_INFINITE_SCORE : alpha - delta;
                if (score >= beta)
                    beta = (beta >= SEARCH_INFINITE_SCORE - 2 * delta) ? SEARCH_INFINITE_SCORE : beta + delta;
                delta *= 2;
            }
        } else {
            score = search_negamax(pos, -SEARCH_INFINITE_SCORE, SEARCH_INFINITE_SCORE, d, 0, rep);
        }
        if (search_limits.stopped) break;

        last_score = score;
        have_last = true;
        if (search_pv_table[0].length > 0) best_move = search_pv_table[0].moves[0];
        search_print_info(d, score);

        // мат найден, и он не длиннее текущей глубины: дальше искать нечего
        if (std::abs(score) >= SEARCH_MATE_SCORE - 100) {
            const int plies = SEARCH_MATE_SCORE - std::abs(score);
            if (plies <= d) {
                search_set_reason_if_none(SEARCH_STOP_MATE);
                break;
            }
        }
    }
    search_set_reason_if_none(SEARCH_STOP_DEPTH);
    return best_move;
}

static bool position_equals(const Position& a, const Position& b) {
    if (a.sideToMove != b.sideToMove || a.castle != b.castle || a.ep != b.ep ||
        a.halfmove != b.halfmove || a.fullmove != b.fullmove || a.key != b.key ||
        a.ksq[WHITE] != b.ksq[WHITE] || a.ksq[BLACK] != b.ksq[BLACK]) return false;
    return std::memcmp(a.piece, b.piece, sizeof(a.piece)) == 0 &&
        std::memcmp(a.bbPieces, b.bbPieces, sizeof(a.bbPieces)) == 0 &&
        std::memcmp(a.bbOccupied, b.bbOccupied, sizeof(a.bbOccupied)) == 0 &&
        a.bbOccupiedAll == b.bbOccupiedAll;
}
static void run_evaluation_tests() {
    Position a, b, before;
    // build0.5: mirrored passed-pawn geometry regression cases.
    parse_fen(a, "4k3/8/4p3/8/4P3/8/8/4K3 w - - 0 1");
    check(!ev_is_passed_pawn(a, WHITE, sq_of("e4")), "passed white same file ahead");
    parse_fen(a, "4k3/8/3p4/8/4P3/8/8/4K3 w - - 0 1");
    check(!ev_is_passed_pawn(a, WHITE, sq_of("e4")), "passed white left file ahead");
    parse_fen(a, "4k3/8/5p2/8/4P3/8/8/4K3 w - - 0 1");
    check(!ev_is_passed_pawn(a, WHITE, sq_of("e4")), "passed white right file ahead");
    parse_fen(a, "4k3/8/8/8/4P3/3p4/8/4K3 w - - 0 1");
    check(ev_is_passed_pawn(a, WHITE, sq_of("e4")), "passed white enemy behind");
    parse_fen(a, "4k3/8/8/4p3/8/3P4/8/4K3 b - - 0 1");
    check(!ev_is_passed_pawn(a, BLACK, sq_of("e5")), "passed black enemy ahead");
    parse_fen(a, "4k3/8/3P4/4p3/8/8/8/4K3 b - - 0 1");
    check(ev_is_passed_pawn(a, BLACK, sq_of("e5")), "passed black enemy behind");
    set_start(a);
    before = a;
    check(evaluate(a) > -30 && evaluate(a) < 30, "eval start balanced");
    check(position_equals(a, before), "eval does not modify position");
    parse_fen(a, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    parse_fen(b, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
    check(evaluate(a) == evaluate(b), "eval start side symmetry");
    parse_fen(a, "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
    parse_fen(b, "4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    check(evaluate(a) > evaluate(b), "eval extra pawn");
    parse_fen(a, "4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
    parse_fen(b, "4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    check(evaluate(a) > evaluate(b) + 350, "eval extra rook");
    parse_fen(a, "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
    parse_fen(b, "4k3/8/8/8/8/8/4P3/4K3 b - - 0 1");
    check(evaluate(a) > 0 && evaluate(b) < 0, "eval side relative");
    parse_fen(a, "4k3/8/4P3/8/8/8/8/4K3 w - - 0 1");
    parse_fen(b, "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
    check(evaluate(a) > evaluate(b), "eval advanced pawn");
    parse_fen(a, "4k3/8/8/8/8/8/P7/R3K3 w - - 0 1");
    parse_fen(b, "4k3/8/8/8/8/8/P7/4K3 w - - 0 1");
    check(evaluate(a) > evaluate(b), "eval rook material");
}
static bool search_move_is_legal_and_roundtrips(Position& pos, const Move& mv) {
    Position copy = pos;
    Undo u;
    if (!search_make_move(pos, mv, u)) return false;
    search_unmake_move(pos, mv, u);
    return position_equals(pos, copy);
}
static void run_search_tests() {
    clear_search_tt();
    Position pos;
    set_start(pos);
    Position before = pos;
    Move m = search(pos, 4, 1000);
    if (m.from == 0 && m.to == 0) std::cout << "SEARCH Test 1: FAIL (no move)" << std::endl;
    else std::cout << "SEARCH Test 1: OK" << std::endl;
    if (position_equals(pos, before)) std::cout << "SEARCH Test 2: OK (start position restored)" << std::endl;
    else std::cout << "SEARCH Test 2: FAIL (position changed)" << std::endl;
    parse_fen(pos, "7k/6Q1/6K1/8/8/8/8/8 w - - 0 1");
    before = pos;
    m = search(pos, 3, 100);
    if (search_move_is_legal_and_roundtrips(pos, m)) std::cout << "SEARCH Test 3: OK" << std::endl;
    else std::cout << "SEARCH Test 3: FAIL" << std::endl;
    if (position_equals(pos, before)) std::cout << "SEARCH Test 4: OK (mate position restored)" << std::endl;
    else std::cout << "SEARCH Test 4: FAIL (mate position changed)" << std::endl;
    parse_fen(pos, "3k4/8/3K4/8/8/8/8/3R4 b - - 0 1");
    before = pos;
    m = search(pos, 3, 100);
    if (search_move_is_legal_and_roundtrips(pos, m)) std::cout << "SEARCH Test 5: OK" << std::endl;
    else std::cout << "SEARCH Test 5: FAIL" << std::endl;
    if (position_equals(pos, before)) std::cout << "SEARCH Test 6: OK (tactical position restored)" << std::endl;
    else std::cout << "SEARCH Test 6: FAIL (tactical position changed)" << std::endl;
}

static void run_repetition_tests() {
    int pass = 0;
    int fail = 0;
    auto result = [&](bool ok, const char* name) {
        if (ok) {
            ++pass;
            std::printf("info string REP %s OK\n", name);
        }
        else {
            ++fail;
            std::printf("info string REP %s FAIL\n", name);
        }
        };
    Position p, q;
    bool ok_p = parse_fen(p, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    bool ok_q = parse_fen(q, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
    result(ok_p && ok_q, "fen_setup");
    if (ok_p && ok_q) {
        RepetitionStack one;
        one.push(p.key);
        result(!is_repetition(p, one), "single_position_not_repeat");

        RepetitionStack prior;
        prior.push(p.key);
        prior.push(q.key);
        prior.push(p.key);
        Position repeated = p;
        repeated.halfmove = 2;
        result(is_repetition(repeated, prior), "same_key_two_plies_back");
        result(!is_repetition(q, prior), "different_key_not_repeat");

        Position after_irreversible = p;
        after_irreversible.halfmove = 0;
        result(!is_repetition(after_irreversible, prior), "irreversible_boundary_respected");
    }
    std::printf("info string REP TESTS DONE pass %d fail %d\n", pass, fail);
    std::fflush(stdout);
}

static void run_tt_tests() {
    int pass = 0;
    int fail = 0;
    auto result = [&](bool ok, const char* name) {
        if (ok) {
            ++pass;
            std::printf("info string TT %s OK\n", name);
        }
        else {
            ++fail;
            std::printf("info string TT %s FAIL\n", name);
        }
        };
    Position p;
    bool ok = parse_fen(p, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    RepetitionStack rep;
    if (ok) rep.push(p.key);
    clear_search_tt();
    Move first = ok ? search(p, 4, 0, rep, 0) : Move{ 0, 0, 0, 0 };
    uint64_t first_hits = search_telemetry.ttHits;
    Move second = ok ? search(p, 4, 0, rep, 0) : Move{ 0, 0, 0, 0 };
    uint64_t second_hits = search_telemetry.ttHits;
    result(ok, "fen_setup");
    result(ok && move_equal(first, second), "repeat_search_same_bestmove");
    result(ok && second_hits > 0, "tt_reused_on_repeat_search");
    std::printf("info string TT first_hits %llu second_hits %llu\n",
        (unsigned long long)first_hits, (unsigned long long)second_hits);
    std::printf("info string TT TESTS DONE pass %d fail %d\n", pass, fail);
    std::fflush(stdout);
}

static void run_final_audit() {
    std::printf("info string AUDIT start\n");
    run_bitboard_tests();

    g_pass = 0;
    g_fail = 0;
    run_evaluation_tests();
    std::printf("info string EVAL TESTS DONE pass %d fail %d\n", g_pass, g_fail);
    std::fflush(stdout);

    run_search_tests();
    run_repetition_tests();
    run_tt_tests();
    run_p0_tests();
    run_ep_tests();
    run_see_tests();
    run_perft_tests();
    std::printf("info string AUDIT done\n");
    std::fflush(stdout);
}
// ============================================================================
// UCI ADAPTER
// ============================================================================
static Position uci_pos;
// Реальная история партии для детекции повторов. Владелец — UCI-слой.
// search() получает КОПИЮ этого стека на каждый вызов.
static RepetitionStack uci_repetition;
static int uci_read_depth_after(const char* p) {
    while (*p == ' ') ++p;
    if (*p < '0' || *p > '9') return -1;
    return std::atoi(p);
}
static void uci_setoption(const char* line) {
    // Supported:
    //   setoption name Hash value 128
    //   setoption name Clear Hash
    // Arena and most GUIs send exactly this form for standard Hash.
    stop_search();
    std::string s(line);
    if (s.find("name Clear Hash") != std::string::npos) {
        clear_search_tt();
        unsigned long long actual_mb = (unsigned long long)((search_tt.size() * sizeof(SearchTTEntry)) / (1024ULL * 1024ULL));
        std::printf("info string hash cleared entries %llu actual %llu MB\n",
            (unsigned long long)search_tt.size(), actual_mb);
        std::fflush(stdout);
        return;
    }
    size_t ev = s.find("name EvalFile");
    if (ev != std::string::npos) {
        // setoption name EvalFile value <путь>; пустой путь, none или off выключают сеть.
        size_t vp = s.find("value", ev + 13);
        std::string path = (vp == std::string::npos) ? std::string() : nnue::trim_path(s.substr(vp + 5));
        if (path.empty() || path == "none" || path == "off") {
            nnue::g_nnue_on = false;
            nnue::g_net.loaded = false;
            std::printf("info string EvalFile off (classical eval)\n");
            std::fflush(stdout);
            return;
        }
        std::string real = path;
        if (!nnue::is_abs_path(path)) {
            // относительный путь: сначала текущая папка, затем папка exe
            std::FILE* t = nnue::open_utf8(path, "rb");
            if (t) std::fclose(t);
            else if (!nnue::g_exe_dir.empty()) real = nnue::g_exe_dir + path;
        }
        nnue::load_net_report(real, "EvalFile");
        return;
    }
    size_t name_pos = s.find("name Hash");
    if (name_pos != std::string::npos) {
        size_t value_pos = s.find("value", name_pos);
        if (value_pos == std::string::npos) return;
        int mb = std::atoi(s.c_str() + value_pos + 5);
        int clamped = search_clamp_hash_mb(mb);
        if (resize_search_tt(clamped)) {
            unsigned long long actual_mb = (unsigned long long)((search_tt.size() * sizeof(SearchTTEntry)) / (1024ULL * 1024ULL));
            std::printf("info string hash set %d MB requested, actual %llu MB entries %llu entry_size %llu\n",
                search_hash_mb,
                actual_mb,
                (unsigned long long)search_tt.size(),
                (unsigned long long)sizeof(SearchTTEntry));
        }
        else {
            std::printf("info string hash allocation failed, keeping %d MB entries %llu\n",
                search_hash_mb,
                (unsigned long long)search_tt.size());
        }
        std::fflush(stdout);
    }
}
static void uci_perft_current_position(int depth, bool divide) {
    stop_search();
    if (depth < 0) {
        std::printf("info string usage: perft <depth>, divide <depth>, or go perft <depth>\n");
        std::fflush(stdout);
        return;
    }
    long long start = search_now();
    uint64_t nodes = divide ? perft_divide(uci_pos, depth) : perft(uci_pos, depth);
    long long elapsed = search_now() - start;
    if (elapsed < 0) elapsed = 0;
    unsigned long long nps = elapsed > 0
        ? (unsigned long long)((nodes * 1000ULL) / (uint64_t)elapsed)
        : (unsigned long long)nodes;
    std::printf("info string perft depth %d nodes %llu time %lld nps %llu\n",
        depth,
        (unsigned long long)nodes,
        elapsed,
        nps);
    std::fflush(stdout);
}
// ============================================================================
// build0.4: telemetry printing helper.
//
// Used for both a single bench position and the aggregate across all bench
// positions. Percentages are always derived from whatever SearchTelemetry
// struct is passed in — for the aggregate call this means the percentages
// are computed from aggregate SUMS (not an average of per-position
// percentages), as required. Zero denominators are handled safely (0%).
// ============================================================================
static double telemetry_pct(uint64_t num, uint64_t den) {
    if (den == 0) return 0.0;
    return (double)num * 100.0 / (double)den;
}
static void print_telemetry_line(const char* label, const SearchTelemetry& t) {
    double ttHitRate = telemetry_pct(t.ttHits, t.ttProbes);
    double ttCutoffRate = telemetry_pct(t.ttCutoffs, t.ttHits);
    double firstMoveCutoffRate = telemetry_pct(t.firstMoveCutoffs, t.betaCutoffs);
    double nullCutoffRate = telemetry_pct(t.nullCutoffs, t.nullTries);
    double lmrResearchRate = telemetry_pct(t.lmrResearches, t.lmrReductions);
    std::printf(
        "info string telemetry %s qnodes %llu ttProbes %llu ttHits %llu ttHitRate %.2f%% "
        "ttCutoffs %llu ttCutoffRate %.2f%% betaCutoffs %llu firstMoveCutoffs %llu "
        "firstMoveCutoffRate %.2f%% nullTries %llu nullCutoffs %llu nullCutoffRate %.2f%% "
        "lmrReductions %llu lmrResearches %llu lmrResearchRate %.2f%% aspirationResearches %llu "
        "seeCalls %llu seePruned %llu futilityPruned %llu deltaPruned %llu "
        "tmSoftStops %llu tmFailLowExtends %llu\n",
        label,
        (unsigned long long)t.qnodes,
        (unsigned long long)t.ttProbes,
        (unsigned long long)t.ttHits,
        ttHitRate,
        (unsigned long long)t.ttCutoffs,
        ttCutoffRate,
        (unsigned long long)t.betaCutoffs,
        (unsigned long long)t.firstMoveCutoffs,
        firstMoveCutoffRate,
        (unsigned long long)t.nullTries,
        (unsigned long long)t.nullCutoffs,
        nullCutoffRate,
        (unsigned long long)t.lmrReductions,
        (unsigned long long)t.lmrResearches,
        lmrResearchRate,
        (unsigned long long)t.aspirationResearches,
        (unsigned long long)t.seeCalls,
        (unsigned long long)t.seePruned,
        (unsigned long long)t.futilityPruned,
        (unsigned long long)t.deltaPruned,
        (unsigned long long)t.tmSoftStops,
        (unsigned long long)t.tmFailLowExtends);
    std::fflush(stdout);
    if (!(t.ttCutoffs <= t.ttHits && t.ttHits <= t.ttProbes)) {
        std::printf(
            "info string telemetry warning %s invariant violated: ttCutoffs<=ttHits<=ttProbes (%llu/%llu/%llu)\n",
            label,
            (unsigned long long)t.ttCutoffs,
            (unsigned long long)t.ttHits,
            (unsigned long long)t.ttProbes);
        std::fflush(stdout);
    }
    if (!(t.firstMoveCutoffs <= t.betaCutoffs)) {
        std::printf(
            "info string telemetry warning %s invariant violated: firstMoveCutoffs<=betaCutoffs (%llu/%llu)\n",
            label,
            (unsigned long long)t.firstMoveCutoffs,
            (unsigned long long)t.betaCutoffs);
        std::fflush(stdout);
    }
    if (!(t.nullCutoffs <= t.nullTries)) {
        std::printf(
            "info string telemetry warning %s invariant violated: nullCutoffs<=nullTries (%llu/%llu)\n",
            label,
            (unsigned long long)t.nullCutoffs,
            (unsigned long long)t.nullTries);
        std::fflush(stdout);
    }
    if (!(t.lmrResearches <= t.lmrReductions)) {
        std::printf(
            "info string telemetry warning %s invariant violated: lmrResearches<=lmrReductions (%llu/%llu)\n",
            label,
            (unsigned long long)t.lmrResearches,
            (unsigned long long)t.lmrReductions);
        std::fflush(stdout);
    }
}
static void uci_bench(int depth) {
    stop_search();
    if (depth <= 0)
        depth = 6;
    struct BenchPosition {
        const char* name;
        const char* fen;
    };
    static const BenchPosition bench_positions[] = {
        {
            "startpos",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        },
        {
            "kiwipete",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        },
        {
            "castles",
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
        },
        {
            "rook_endgame",
            "3k4/8/3K4/8/8/8/8/3R4 b - - 0 1"
        }
    };
    clear_search_tt();
    long long total_nodes = 0;
    long long total_start = search_now();
    SearchTelemetry aggregate{};
    std::printf("info string bench start depth %d positions %d\n",
        depth,
        (int)(sizeof(bench_positions) / sizeof(bench_positions[0])));
    std::fflush(stdout);
    for (const BenchPosition& bp : bench_positions) {
        Position p;
        if (!parse_fen(p, bp.fen)) {
            std::printf("info string bench %s bad fen\n", bp.name);
            continue;
        }
        long long start = search_now();
        Move best = search(p, depth, 0);
        long long elapsed = search_now() - start;
        if (elapsed < 0) elapsed = 0;
        long long nodes = search_limits.nodes;
        total_nodes += nodes;
        long long nps = elapsed > 0 ? (nodes * 1000LL) / elapsed : nodes;
        char best_buf[8];
        if (!search_is_null_move_local(best)) {
            move_str(best, best_buf);
        }
        else {
            // Не используем strcpy: MSVC ругается C4996.
            best_buf[0] = '0';
            best_buf[1] = '0';
            best_buf[2] = '0';
            best_buf[3] = '0';
            best_buf[4] = 0;
        }
        std::printf("info string bench %s depth %d bestmove %s nodes %lld time %lld nps %lld\n",
            bp.name,
            depth,
            best_buf,
            nodes,
            elapsed,
            nps);
        std::fflush(stdout);
        // build0.4: search() is fully finished by this point (called
        // synchronously, not via g_search_thread), so this copy+print never
        // races with an active search and never happens from a UCI thread
        // while search is running.
        SearchTelemetry t = search_telemetry;
        print_telemetry_line(bp.name, t);
        aggregate.qnodes += t.qnodes;
        aggregate.ttProbes += t.ttProbes;
        aggregate.ttHits += t.ttHits;
        aggregate.ttCutoffs += t.ttCutoffs;
        aggregate.betaCutoffs += t.betaCutoffs;
        aggregate.firstMoveCutoffs += t.firstMoveCutoffs;
        aggregate.nullTries += t.nullTries;
        aggregate.nullCutoffs += t.nullCutoffs;
        aggregate.lmrReductions += t.lmrReductions;
        aggregate.lmrResearches += t.lmrResearches;
        aggregate.aspirationResearches += t.aspirationResearches;
        aggregate.seeCalls += t.seeCalls;
        aggregate.seePruned += t.seePruned;
        aggregate.futilityPruned += t.futilityPruned;
        aggregate.deltaPruned += t.deltaPruned;
        aggregate.tmSoftStops += t.tmSoftStops;
        aggregate.tmFailLowExtends += t.tmFailLowExtends;
    }
    long long total_elapsed = search_now() - total_start;
    if (total_elapsed < 0) total_elapsed = 0;
    long long total_nps = total_elapsed > 0 ? (total_nodes * 1000LL) / total_elapsed : total_nodes;
    std::printf("info string bench done depth %d total_nodes %lld total_time %lld total_nps %lld\n",
        depth,
        total_nodes,
        total_elapsed,
        total_nps);
    std::fflush(stdout);
    print_telemetry_line("aggregate", aggregate);
}
static bool get_legal_mv(const Position& pos, const char* s, Move& out) {
    MoveList ml;
    generateLegalMoves(pos, ml);
    char buf[8];
    for (int i = 0; i < ml.count; ++i) {
        move_str(ml.m[i], buf);
        if (!std::strcmp(buf, s)) {
            out = ml.m[i];
            return true;
        }
    }
    return false;
}
// build0.2 FIX 3: убран фиксированный лимит истории партии в 256 полуходов.
// makeMove() применяется напрямую (Undo нам тут больше не нужен за пределами
// самого вызова), а repetition stack сбрасывается на каждом необратимом ходе
// (halfmove == 0 — пешечный ход или взятие), т.к. более старые позиции уже
// не могут участвовать в повторении, и это же не даёт RepetitionStack
// переполниться на длинной партии.
static void uci_apply_move(Position& pos, const char* s) {
    Move m{};
    if (!get_legal_mv(pos, s, m)) {
        std::printf("info string illegal move %s\n", s);
        return;
    }
    Undo scratch;
    makeMove(pos, m, scratch);
    if (pos.halfmove == 0)
        uci_repetition.count = 0;
    uci_repetition.push(pos.key);
}
// build 1.5 (исправление гонки): остановить поиск и дождаться его потока.
// Без join команды position/ucinewgame меняли доску и TT, пока старый поиск
// ещё её читал (в тесте без stop давало невозможный ход e1e2 для чёрных).
static void uci_stop_and_join_search() {
    stop_search();
    if (g_search_thread.joinable()) g_search_thread.join();
}
static void uci_position(const char* line) {
    uci_stop_and_join_search();
    uci_repetition.count = 0;
    // set_start() здесь — только предварительный/дефолтный сброс доски.
    // Его ключ намеренно НЕ пушится в uci_repetition сейчас — корнем
    // repetition-истории должна быть ОКОНЧАТЕЛЬНАЯ позиция (startpos или
    // успешно разобранный FEN), а не этот промежуточный placeholder.
    set_start(uci_pos);
    const char* p = line + 8;
    while (*p == ' ') ++p;
    if (std::strncmp(p, "startpos", 8) == 0) {
        p += 8;
    }
    else if (std::strncmp(p, "fen", 3) == 0) {
        p += 3;
        while (*p == ' ') ++p;
        char fen[256];
        int i = 0;
        while (*p && std::strncmp(p, "moves", 5) != 0 && i < 255) fen[i++] = *p++;
        while (i > 0 && fen[i - 1] == ' ') --i;
        fen[i] = 0;
        if (!parse_fen(uci_pos, fen)) {
            std::printf("info string bad fen\n");
            // build0.1: немедленно выходим при некорректном FEN — не
            // добавляем повреждённую/неопределённую позицию в repetition
            // stack и не пытаемся применять ходы поверх неё.
            return;
        }
    }
    else return;
    // build0.1: root-позиция теперь окончательно определена (startpos или
    // валидный FEN). Пушим её ровно один раз, до применения любых ходов —
    // это гарантирует, что "position startpos" / "position fen X" без
    // списка moves всё равно удовлетворяет
    // rep.keys[rep.count-1] == uci_pos.key.
    uci_repetition.push(uci_pos.key);
    while (*p == ' ') ++p;
    if (std::strncmp(p, "moves", 5) == 0) {
        p += 5;
        while (*p) {
            while (*p == ' ') ++p;
            if (!*p) break;
            char mv[8] = {};
            int j = 0;
            while (*p && *p != ' ' && j < 7) mv[j++] = *p++;
            mv[j] = 0;
            uci_apply_move(uci_pos, mv);
        }
    }
}
static bool is_null_move(const Move& m)
{
    return m.from == 0 &&
        m.to == 0 &&
        m.flags == 0;
}
static void uci_go(const char* line) {
    // build0.6: join any previous search before reading uci_pos. The old
    // order read sideToMove while the search thread could be making moves.
    stop_search();
    int depth = 0, movetime = 0;
    int wtime = 0, btime = 0, winc = 0, binc = 0;
    uint64_t node_limit = 0;
    bool nodes_specified = false;
    const char* p = line + 2;
    while (*p) {
        while (*p == ' ') ++p;
        if (std::strncmp(p, "depth", 5) == 0) {
            p += 5; while (*p == ' ') ++p;
            depth = std::atoi(p);
        }
        else if (std::strncmp(p, "movetime", 8) == 0) {
            p += 8; while (*p == ' ') ++p;
            movetime = std::atoi(p);
        }
        else if (std::strncmp(p, "wtime", 5) == 0) {
            p += 5; while (*p == ' ') ++p;
            wtime = std::atoi(p);
        }
        else if (std::strncmp(p, "btime", 5) == 0) {
            p += 5; while (*p == ' ') ++p;
            btime = std::atoi(p);
        }
        else if (std::strncmp(p, "winc", 4) == 0) {
            p += 4; while (*p == ' ') ++p;
            winc = std::atoi(p);
        }
        else if (std::strncmp(p, "binc", 4) == 0) {
            p += 4; while (*p == ' ') ++p;
            binc = std::atoi(p);
        }
        else if (std::strncmp(p, "nodes", 5) == 0 &&
            (p[5] == ' ' || p[5] == 0)) {
            p += 5; while (*p == ' ') ++p;
            // A present but malformed/negative/overflowing nodes token is
            // safely treated as a zero budget, never as an unlimited search.
            nodes_specified = true;
            node_limit = 0;
            if (*p >= '0' && *p <= '9') {
                errno = 0;
                char* endptr = nullptr;
                unsigned long long parsed = std::strtoull(p, &endptr, 10);
                if (errno != ERANGE && endptr != p &&
                    (*endptr == 0 || *endptr == ' ')) {
                    const unsigned long long maximum =
                        (unsigned long long)LLONG_MAX;
                    if (parsed > maximum) parsed = maximum;
                    node_limit = (uint64_t)parsed;
                    p = endptr;
                }
            }
        }
        while (*p && *p != ' ') ++p;
    }
    // build1.0: soft/hard time budget replaces the flat "available/30 + inc".
    // Soft (don't start a new iteration past it) = available/25 + 0.7*inc.
    // Hard (abort switch in the node loop, as before) = min(4*soft,
    // available/4, available - 50ms lag buffer). The increment is weighted
    // 0.7 and capped by the remaining clock, so the engine can never bet
    // time it has not yet earned (the old formula could flag on
    // "remaining 100ms + inc 1000ms"). Explicit "go movetime" bypasses all
    // of this: g_tm_soft_ms stays 0 and behavior is identical to build0.9.
    g_tm_soft_ms = 0;
    if (movetime <= 0) {
        int available = (uci_pos.sideToMove == WHITE) ? wtime : btime;
        if (available > 0 || winc > 0 || binc > 0) {
            long long inc = (uci_pos.sideToMove == WHITE) ? winc : binc;
            long long soft = (long long)available / 25 + inc * 7 / 10;
            long long hard = soft * 4;
            if (hard > (long long)available / 4) hard = (long long)available / 4;
            long long cap = (long long)available - 50;
            if (cap < 1) cap = 1;
            if (hard > cap) hard = cap;
            if (hard < 5) hard = 5;
            if (soft > hard) soft = hard;
            if (soft < 1) soft = 1;
            movetime = (int)hard;
            g_tm_soft_ms = soft;
        }
    }
    if (depth <= 0) depth = 64;

    // build0.6: an explicit zero (or negative) budget starts no search but
    // still returns one legal fallback move as required by UCI game play.
    if (nodes_specified && node_limit == 0) {
        MoveList legal;
        generateLegalMoves(uci_pos, legal);
        if (legal.count == 0) {
            std::printf("info string node budget requested 0 searched 0 reason nodes\n");
            std::printf("bestmove 0000\n"); TRACE_OUT("OUT bestmove", "0000");
        }
        else {
            char buf[8];
            move_str(legal.m[0], buf);
            std::printf("info string node budget requested 0 searched 0 reason nodes\n");
            std::printf("bestmove %s\n", buf); TRACE_OUT("OUT bestmove", buf);
        }
        std::fflush(stdout);
        return;
    }

    // build 1.5 (исправление падения в Arena): предыдущий поток поиска обязан быть
    // присоединён до присваивания g_search_thread. std::thread::operator= над
    // joinable-объектом вызывает std::terminate, поэтому второй "go" в той же сессии
    // убивал процесс (после первого хода, как в логе uci_log.txt). Если старый поиск
    // ещё идёт, останавливаем его; обычно он уже завершён и bestmove уже напечатан.
    if (g_search_thread.joinable()) {
        search_limits.stopped = true;
        g_search_thread.join();
    }
    search_limits.stopped = false;
    int depth_copy = depth;
    int movetime_copy = movetime;
    uint64_t node_limit_copy = nodes_specified ? node_limit : 0;
    g_search_initialized = false;
    g_search_running = true;
    g_search_thread = std::thread([depth_copy, movetime_copy, node_limit_copy]() {
        Move best = search(uci_pos, depth_copy, movetime_copy,
            uci_repetition, node_limit_copy);
        if (node_limit_copy > 0) {
            std::printf("info string node budget requested %llu searched %lld reason %s\n",
                (unsigned long long)node_limit_copy,
                search_limits.nodes,
                search_stop_reason_name());
        }
        char buf[8];
        if (!is_null_move(best)) {
            move_str(best, buf);
            std::printf("bestmove %s\n", buf); TRACE_OUT("OUT bestmove", buf);
        }
        else {
            std::printf("bestmove 0000\n"); TRACE_OUT("OUT bestmove", "0000");
        }
        std::fflush(stdout);
        g_search_running = false;
        });
    // Do not let the UCI loop consume a following "stop" until search() has
    // completed its one-time initialization, including stopped=false.
    while (!g_search_initialized.load() && g_search_thread.joinable())
        std::this_thread::yield();
}
int uci_main() {
    set_start(uci_pos);
    uci_repetition.count = 0;
    uci_repetition.push(uci_pos.key);
    char line[4096];
    while (std::fgets(line, sizeof(line), stdin)) {
        line[std::strcspn(line, "\r\n")] = 0;
        if (!line[0]) continue;
        TRACE_IN(line);
        if (!std::strcmp(line, "uci")) {
            std::printf("id name MyChessEngine build1.5\n");
            std::printf("id author AI Battle Project\n");
            std::printf("option name Hash type spin default %d min %d max %d\n",
                SEARCH_DEFAULT_HASH_MB, SEARCH_MIN_HASH_MB, SEARCH_MAX_HASH_MB);
            std::printf("option name Clear Hash type button\n");
            std::printf("option name EvalFile type string default net_comp_ft_ep49.nnue\n");
            std::printf("uciok\n"); TRACE_OUT("OUT", "uciok");
        }
        else if (!std::strcmp(line, "isready")) {
            std::printf("readyok\n"); TRACE_OUT("OUT", "readyok");
        }
        else if (std::strncmp(line, "setoption ", 10) == 0) {
            uci_setoption(line);
        }
        else if (!std::strcmp(line, "ucinewgame")) {
            uci_stop_and_join_search();
            clear_search_tt();
            std::memset(search_killer_moves, 0, sizeof(search_killer_moves));
            std::memset(search_history_moves, 0, sizeof(search_history_moves));
            uci_repetition.count = 0;
            set_start(uci_pos);
            uci_repetition.push(uci_pos.key);
        }
        else if (!std::strcmp(line, "d")) {
            stop_search();
            std::printf("info string side %s\ninfo string hash %llu\n",
                uci_pos.sideToMove == WHITE ? "white" : "black",
                (unsigned long long)uci_pos.key);
        }
        else if (!std::strcmp(line, "debug on")) { std::printf("info string debug enabled\n"); }
        else if (!std::strcmp(line, "debug off")) { std::printf("info string debug disabled\n"); }
        else if (!std::strcmp(line, "eval")) {
            stop_search();
            std::printf("info string eval cp %d\n", evaluate(uci_pos));
            std::fflush(stdout);
        }
        else if (!std::strcmp(line, "audit")) {
            stop_search();
            clear_search_tt();
            run_final_audit();
        }
        else if (!std::strcmp(line, "p0test")) {
            stop_search();
            clear_search_tt();
            run_p0_tests();
        }
        else if (!std::strcmp(line, "eptest")) {
            stop_search();
            run_ep_tests();
        }
        else if (!std::strcmp(line, "seetest")) {
            stop_search();
            run_see_tests();
        }
        else if (!std::strcmp(line, "perfttest")) {
            stop_search();
            run_perft_tests();
        }
        else if (!std::strcmp(line, "bbtest")) {
            stop_search();
            run_bitboard_tests();
        }
        else if (std::strncmp(line, "bench", 5) == 0 && (line[5] == ' ' || line[5] == 0)) {
            uci_bench(uci_read_depth_after(line + 5));
        }
        else if (std::strncmp(line, "perft", 5) == 0 && (line[5] == ' ' || line[5] == 0)) {
            uci_perft_current_position(uci_read_depth_after(line + 5), false);
        }
        else if (std::strncmp(line, "divide", 6) == 0 && (line[6] == ' ' || line[6] == 0)) {
            uci_perft_current_position(uci_read_depth_after(line + 6), true);
        }
        else if (std::strncmp(line, "position ", 9) == 0) {
            uci_position(line);
        }
        else if (std::strncmp(line, "go perft", 8) == 0 && (line[8] == ' ' || line[8] == 0)) {
            uci_perft_current_position(uci_read_depth_after(line + 8), false);
        }
        else if (!std::strncmp(line, "go", 2) && (line[2] == ' ' || line[2] == 0)) {
            uci_go(line);
        }
        else if (!std::strcmp(line, "stop")) {
            stop_search();
        }
        else if (!std::strcmp(line, "nnueinfo")) {
            std::printf("nnueinfo %s path=%s hash=0x%08X\n", nnue::g_nnue_on ? "on" : "off",
                        nnue::g_nnue_on ? nnue::g_nnue_path : "-", nnue::g_nnue_on ? nnue::g_net_hash : 0u);
            std::fflush(stdout);
        }
        else if (std::strncmp(line, "nnuecheck ", 10) == 0) {
            int dep = std::atoi(line + 10);
            const char* fen = std::strchr(line + 10, ' ');
            Position cp;
            if (!fen || !parse_fen(cp, fen + 1)) std::printf("nnuecheck bad fen\n");
            else if (!nnue::g_net.loaded) std::printf("nnuecheck net not loaded\n");
            else {
                nnue::g_ply = 0;
                nnue::g_chk_nodes = 0; nnue::g_chk_fail = 0;
                nnue::debug_walk(cp, dep);
                std::printf("nnuecheck depth %d nodes %lld fails %lld\n", dep, nnue::g_chk_nodes, nnue::g_chk_fail);
            }
            std::fflush(stdout);
        }
        else if (std::strncmp(line, "nneval ", 7) == 0) {
            Position np;
            if (!parse_fen(np, line + 7)) std::printf("nneval bad fen\n");
            else if (!nnue::g_net.loaded) std::printf("nneval net not loaded\n");
            else std::printf("nneval %d\n", nnue::evaluate(np));
            std::fflush(stdout);
        }
        else if (!std::strcmp(line, "quit")) {
            stop_search();
            break;
        }
        std::fflush(stdout);
    }
    stop_search();
    // build 1.5 (исправление): поток поиска обязан быть присоединён до выхода из процесса
    // (иначе std::thread::~thread при joinable вызывает terminate).
    if (g_search_thread.joinable()) g_search_thread.join();
    return 0;
}
int main(int argc, char** argv) {
    (void)argc;
    initZobrist();
    bb_init_attack_tables();
    // Build the read-only magic tables before the UCI loop so table
    // construction is not charged to the first search benchmark.
    bb_init_magic_tables();
    resize_search_tt(SEARCH_DEFAULT_HASH_MB);
    nnue::init_exe_dir(argv[0]);
    TRACE_START(argv[0]);
    nnue::autoload(argv[0]);
    return uci_main();
}

