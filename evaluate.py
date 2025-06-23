# Código para carregar policy_mixed de arquivo .pth e executar coleta de estatísticas instrumentadas
import random
import statistics
from collections import Counter
import math

# --- Definições necessárias (Card, build_deck, detect_yaku, score_captures, extract_features, LinearPolicy, heuristic_choose) ---
class Card:
    def __init__(self, month, kind, name):
        self.month = month; self.kind = kind; self.name = name
    def __repr__(self):
        return f"{self.name}({self.month})"
    def __eq__(self, other):
        return isinstance(other, Card) and (self.month, self.kind, self.name) == (other.month, other.kind, other.name)
    def __hash__(self):
        return hash((self.month, self.kind, self.name))

def build_deck():
    cards = []
    cards.extend([Card(1, 'bright', 'Crane'),
                  Card(1, 'ribbon', 'Red Poetry'),
                  Card(1, 'junk', 'Pine Chaff 1'),
                  Card(1, 'junk', 'Pine Chaff 2')])
    cards.extend([Card(2, 'animal', 'Bush Warbler'),
                  Card(2, 'ribbon', 'Red Poetry'),
                  Card(2, 'junk', 'Plum Chaff 1'),
                  Card(2, 'junk', 'Plum Chaff 2')])
    cards.extend([Card(3, 'bright', 'Curtain'),
                  Card(3, 'ribbon', 'Red Poetry'),
                  Card(3, 'junk', 'Cherry Chaff 1'),
                  Card(3, 'junk', 'Cherry Chaff 2')])
    cards.extend([Card(4, 'animal', 'Cuckoo'),
                  Card(4, 'ribbon', 'Purple Ribbon'),
                  Card(4, 'junk', 'Wisteria Chaff 1'),
                  Card(4, 'junk', 'Wisteria Chaff 2')])
    cards.extend([Card(5, 'animal', 'Eight-Plank Bridge'),
                  Card(5, 'ribbon', 'Blue Ribbon'),
                  Card(5, 'junk', 'Iris Chaff 1'),
                  Card(5, 'junk', 'Iris Chaff 2')])
    cards.extend([Card(6, 'animal', 'Butterflies'),
                  Card(6, 'ribbon', 'Blue Ribbon'),
                  Card(6, 'junk', 'Peony Chaff 1'),
                  Card(6, 'junk', 'Peony Chaff 2')])
    cards.extend([Card(7, 'animal', 'Boar'),
                  Card(7, 'ribbon', 'Red Ribbon'),
                  Card(7, 'junk', 'Bush Clover Chaff 1'),
                  Card(7, 'junk', 'Bush Clover Chaff 2')])
    cards.extend([Card(8, 'animal', 'Geese'),
                  Card(8, 'ribbon', 'Red Ribbon'),
                  Card(8, 'junk', 'Pampas Chaff 1'),
                  Card(8, 'junk', 'Pampas Chaff 2')])
    cards.extend([Card(9, 'animal', 'Sake Cup'),
                  Card(9, 'ribbon', 'Blue Ribbon'),
                  Card(9, 'junk', 'Chrysanthemum Chaff 1'),
                  Card(9, 'junk', 'Chrysanthemum Chaff 2')])
    cards.extend([Card(10, 'bright', 'Moon'),
                  Card(10, 'ribbon', 'Blue Ribbon'),
                  Card(10, 'junk', 'Maple Chaff 1'),
                  Card(10, 'junk', 'Maple Chaff 2')])
    cards.extend([Card(11, 'bright', 'Rainman'),
                  Card(11, 'animal', 'Swallow'),
                  Card(11, 'ribbon', 'Purple Ribbon'),
                  Card(11, 'junk', 'Willow Chaff')])
    cards.extend([Card(12, 'bright', 'Phoenix'),
                  Card(12, 'junk', 'Paulownia Chaff 1'),
                  Card(12, 'junk', 'Paulownia Chaff 2'),
                  Card(12, 'junk', 'Paulownia Chaff 3')])
    return cards.copy()

def detect_yaku(captures):
    cnt = Counter(c.kind for c in captures)
    yaku = {}
    if cnt['bright'] >= 5:
        yaku['Goko'] = 10
    elif cnt['bright'] == 4:
        yaku['Shiko'] = 8
    elif cnt['bright'] == 3:
        yaku['Sanko'] = 5
    if cnt['animal'] >= 5:
        yaku['Tane'] = cnt['animal'] - 4
    if cnt['ribbon'] >= 5:
        yaku['Tanzaku'] = cnt['ribbon'] - 4
    if cnt['junk'] >= 10:
        yaku['Kasu'] = cnt['junk'] - 9
    return yaku

def score_captures(captures):
    return sum(detect_yaku(captures).values())

def extract_features(hand, table, caps, remain_counts):
    features = []
    total_remain = sum(remain_counts.values()) + 1e-6
    for c in hand:
        f = [
            1.0 if c.kind == 'bright' else 0.0,
            1.0 if c.kind == 'animal' else 0.0,
            1.0 if c.kind == 'ribbon' else 0.0,
            1.0 if c.kind == 'junk' else 0.0,
            sum(1 for t in table if t.month == c.month),
            remain_counts.get(c, 0) / total_remain
        ]
        features.append(f)
    return features

class LinearPolicy:
    def __init__(self, feature_len, lr=0.005):
        self.w = [random.uniform(-0.1, 0.1) for _ in range(feature_len)]
        self.lr = lr

    def action_probs(self, features):
        scores = [sum(w_i * f_i for w_i, f_i in zip(self.w, f)) for f in features]
        max_s = max(scores)
        exps = [math.exp(s - max_s) for s in scores]
        total = sum(exps)
        return [e / total for e in exps]

    def choose_action(self, features):
        probs = self.action_probs(features)
        r = random.random()
        cum = 0.0
        for i, p in enumerate(probs):
            cum += p
            if r < cum:
                return i, probs
        return len(probs) - 1, probs

    def update(self, trajectories):
        for features, action_index, reward in trajectories:
            probs = self.action_probs(features)
            exp_feat = [0.0] * len(self.w)
            for p, f in zip(probs, features):
                for idx in range(len(self.w)):
                    exp_feat[idx] += p * f[idx]
            f_a = features[action_index]
            for idx in range(len(self.w)):
                grad = f_a[idx] - exp_feat[idx]
                self.w[idx] += self.lr * reward * grad

# Função heurística de referência do oponente
def heuristic_choose(hand, table, caps, remain_counts):
    base = score_captures(caps)
    best, best_val = None, -1e9
    for c in hand:
        temp_caps = caps.copy()
        for t in table:
            if t.month == c.month:
                temp_caps += [c, t]
                break
        delta = score_captures(temp_caps) - base
        risk = sum(1 for t in table if t.month == c.month)
        val = delta - 0.5 * risk
        if val > best_val:
            best_val, best = val, c
    if best is None:
        best = random.choice(hand)
    return best

# --- Função para carregar policy a partir de .pth ---
def load_policy_pth(path):
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch não está disponível neste ambiente.")
    data = torch.load(path)
    w_list = data['w'].tolist() if hasattr(data['w'], 'tolist') else list(data['w'])
    lr = data.get('lr', 0.005)
    policy = LinearPolicy(feature_len=len(w_list), lr=lr)
    policy.w = w_list
    print(f"Política carregada de {path}")
    return policy

# Tente carregar o modelo salvo em /mnt/data/policy_mixed.pth
model_path = 'policy_mixed.pth'
try:
    policy_mixed = load_policy_pth(model_path)
except Exception as e:
    raise RuntimeError(f"Falha ao carregar política de {model_path}: {e}")

# Função de simulação instrumentada
def play_episode_instrumented(policy, opponent_type='heuristic'):
    deck = build_deck()
    random.shuffle(deck)
    hand = [deck.pop() for _ in range(8)]
    hand_o = [deck.pop() for _ in range(8)]
    table = [deck.pop() for _ in range(8)]
    caps = []
    caps_o = []
    remain_counts = Counter(deck)
    first_yaku_recorded = False
    first_yaku_turn = None
    safe_score = None
    first_yaku_types = None
    turn_count = 0
    turn = 0
    while hand or hand_o:
        if turn == 0 and hand:
            features = extract_features(hand, table, caps, remain_counts)
            ai, _ = policy.choose_action(features)
            card = hand.pop(ai)
            turn_count += 1
            matched = False
            for t in table:
                if t.month == card.month:
                    caps += [card, t]; table.remove(t); matched = True; break
            if not matched:
                table.append(card)
            if deck:
                d = deck.pop(0)
                if remain_counts[d] > 0: remain_counts[d] -= 1
                matched = False
                for t in table:
                    if t.month == d.month:
                        caps += [d, t]; table.remove(t); matched = True; break
                if not matched:
                    table.append(d)
            if not first_yaku_recorded:
                yaku = detect_yaku(caps)
                if yaku:
                    first_yaku_recorded = True
                    first_yaku_turn = turn_count
                    safe_score = score_captures(caps)
                    first_yaku_types = tuple(sorted(yaku.keys()))
        elif turn == 1 and hand_o:
            card = heuristic_choose(hand_o, table, caps_o, remain_counts)
            hand_o.remove(card)
            turn_count += 1
            matched = False
            for t in table:
                if t.month == card.month:
                    caps_o += [card, t]; table.remove(t); matched = True; break
            if not matched:
                table.append(card)
            if deck:
                d = deck.pop(0)
                if remain_counts[d] > 0: remain_counts[d] -= 1
                matched = False
                for t in table:
                    if t.month == d.month:
                        caps_o += [d, t]; table.remove(t); matched = True; break
                if not matched:
                    table.append(d)
        else:
            break
        turn ^= 1
    final_score = score_captures(caps)
    opp_score = score_captures(caps_o)
    success_continue = None
    if first_yaku_recorded and safe_score is not None:
        success_continue = (final_score > safe_score)
    return {
        'final_score': final_score,
        'opp_score': opp_score,
        'score_diff': final_score - opp_score,
        'first_yaku_turn': first_yaku_turn,
        'safe_score': safe_score,
        'final_score_after_yaku': final_score if first_yaku_recorded else None,
        'first_yaku_types': first_yaku_types,
        'success_continue': success_continue,
        'turns': turn_count
    }

# Executar múltiplas partidas e coletar estatísticas
random.seed(42)
n_games = 500
results = [play_episode_instrumented(policy_mixed, opponent_type='heuristic') for _ in range(n_games)]

# Agregar estatísticas
wins = sum(1 for r in results if r['score_diff'] > 0)
losses = sum(1 for r in results if r['score_diff'] < 0)
draws = sum(1 for r in results if r['score_diff'] == 0)
avg_score = statistics.mean(r['final_score'] for r in results)
avg_opp_score = statistics.mean(r['opp_score'] for r in results)
first_yaku_records = [r for r in results if r['first_yaku_turn'] is not None]
pct_with_yaku = len(first_yaku_records) / n_games * 100
avg_first_turn = statistics.mean(r['first_yaku_turn'] for r in first_yaku_records) if first_yaku_records else None
std_first_turn = statistics.pstdev([r['first_yaku_turn'] for r in first_yaku_records]) if first_yaku_records else None
avg_safe_score = statistics.mean(r['safe_score'] for r in first_yaku_records) if first_yaku_records else None
avg_final_after = statistics.mean(r['final_score_after_yaku'] for r in first_yaku_records) if first_yaku_records else None
success_list = [r['success_continue'] for r in first_yaku_records if r['success_continue'] is not None]
success_rate = sum(1 for x in success_list if x) / len(success_list) * 100 if success_list else None
yaku_counter = Counter(r['first_yaku_types'] for r in first_yaku_records if r['first_yaku_types'] is not None)
avg_turns = statistics.mean(r['turns'] for r in results)
std_turns = statistics.pstdev(r['turns'] for r in results)

# Mostrar resultados
print(f"Testes contra heurística ({n_games} jogos):")
print(f" Win rate: {wins/n_games*100:.1f}%  Loss rate: {losses/n_games*100:.1f}%  Draw rate: {draws/n_games*100:.1f}%")
print(f" Pontuação média IA: {avg_score:.2f}, heurística: {avg_opp_score:.2f}")
print(f" % partidas com yaku: {pct_with_yaku:.1f}%")
if first_yaku_records:
    print(f" Turno médio do primeiro yaku: {avg_first_turn:.2f} (std {std_first_turn:.2f})")
    print(f" Pontuação segura média ao formar yaku: {avg_safe_score:.2f}")
    print(f" Pontuação média final após formar yaku: {avg_final_after:.2f}")
    print(f" Taxa de sucesso ao continuar (final > segura): {success_rate:.1f}%")
    print("Distribuição de primeiras yaku formados (exemplos de combinações):")
    for yaku_types, cnt in yaku_counter.most_common(10):
        print(f"  {yaku_types}: {cnt} vezes")
print(f" Turnos médios por partida: {avg_turns:.2f} (std {std_turns:.2f})")