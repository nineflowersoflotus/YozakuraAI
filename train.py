import random
import math
from collections import Counter

# --- Definições de carta e baralho ---
class Card:
    def __init__(self, month, kind, name):
        self.month = month
        self.kind = kind
        self.name = name
    def __repr__(self):
        return f"{self.name}({self.month})"
    def __eq__(self, other):
        return isinstance(other, Card) and (self.month, self.kind, self.name) == (other.month, other.kind, other.name)
    def __hash__(self):
        return hash((self.month, self.kind, self.name))

def build_deck():
    cards = []
    # Construção de 48 cartas conforme Hanafuda
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

# --- Funções de detecção de yaku e pontuação ---
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

# --- Heurística simples para usar como referência ---
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

# --- Features e política linear ---
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
    def __init__(self, feature_len, lr=0.01):
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

# --- Simular episódio policy vs opponent (heuristic ou random) ---
def play_episode(policy, opponent_type='heuristic'):
    deck = build_deck()
    random.shuffle(deck)
    hand = [deck.pop() for _ in range(8)]
    if opponent_type == 'heuristic':
        hand_o = [deck.pop() for _ in range(8)]
    else:  # random opponent
        hand_o = [deck.pop() for _ in range(8)]
    table = [deck.pop() for _ in range(8)]
    caps = []
    caps_o = []
    remain_counts = Counter(deck)
    trajectories = []
    turn = 0  # 0: policy, 1: opponent
    while hand or hand_o:
        if turn == 0 and hand:
            features = extract_features(hand, table, caps, remain_counts)
            ai, _ = policy.choose_action(features)
            card = hand.pop(ai)
            trajectories.append((features, ai, None))
            # execução da jogada
            captured = False
            for t in table:
                if t.month == card.month:
                    caps += [card, t]
                    table.remove(t)
                    captured = True
                    break
            if not captured:
                table.append(card)
            # compra
            if deck:
                d = deck.pop(0)
                if remain_counts[d] > 0:
                    remain_counts[d] -= 1
                cap_flag = False
                for t in table:
                    if t.month == d.month:
                        caps += [d, t]
                        table.remove(t)
                        cap_flag = True
                        break
                if not cap_flag:
                    table.append(d)
        elif turn == 1 and hand_o:
            if opponent_type == 'heuristic':
                card = heuristic_choose(hand_o, table, caps_o, remain_counts)
            else:
                card = random.choice(hand_o)
            hand_o.remove(card)
            captured = False
            for t in table:
                if t.month == card.month:
                    caps_o += [card, t]
                    table.remove(t)
                    captured = True
                    break
            if not captured:
                table.append(card)
            if deck:
                d = deck.pop(0)
                if remain_counts[d] > 0:
                    remain_counts[d] -= 1
                cap_flag = False
                for t in table:
                    if t.month == d.month:
                        caps_o += [d, t]
                        table.remove(t)
                        cap_flag = True
                        break
                if not cap_flag:
                    table.append(d)
        else:
            break
        turn ^= 1
    # resultado e recompensa
    s = score_captures(caps)
    s_o = score_captures(caps_o)
    reward = s - s_o
    # atribuir reward às trajetórias
    for i in range(len(trajectories)):
        f, ai, _ = trajectories[i]
        trajectories[i] = (f, ai, reward)
    return reward, trajectories

# --- Treinamento com mistura de oponentes ---
def train_mixed(episodes=1000, lr=0.005):
    random.seed(42)
    feature_len = 6
    policy = LinearPolicy(feature_len, lr=lr)
    for ep in range(1, episodes+1):
        # escolher aleatoriamente oponente: 50% heurística, 50% random
        opponent_type = random.choice(['heuristic', 'random'])
        reward, trajectories = play_episode(policy, opponent_type=opponent_type)
        policy.update(trajectories)
        # opcional: avaliação periódica
        if ep % 200 == 0:
            print(f"Episode {ep}/{episodes} completed")
    return policy

# --- Teste contra heurística e random ---
def test_policy(policy, games=200):
    wins_h = 0
    diffs_h = []
    wins_r = 0
    diffs_r = []
    for _ in range(games):
        r_h, _ = play_episode(policy, opponent_type='heuristic')
        diffs_h.append(r_h)
        if r_h > 0:
            wins_h += 1
        r_r, _ = play_episode(policy, opponent_type='random')
        diffs_r.append(r_r)
        if r_r > 0:
            wins_r += 1
    win_rate_h = wins_h / games * 100
    avg_diff_h = sum(diffs_h) / games
    win_rate_r = wins_r / games * 100
    avg_diff_r = sum(diffs_r) / games
    print(f"Policy vs Heuristic: win rate = {win_rate_h:.1f}%, avg score diff = {avg_diff_h:.2f}")
    print(f"Policy vs Random:    win rate = {win_rate_r:.1f}%, avg score diff = {avg_diff_r:.2f}")
    return {
        'heuristic': (win_rate_h, avg_diff_h),
        'random': (win_rate_r, avg_diff_r)
    }

# Executar treinamento misto e testes
policy_mixed = train_mixed(episodes=50000, lr=0.005)
results = test_policy(policy_mixed, games=200)

import torch

def save_policy_pth(policy, path):
    # Converte lista de pesos para tensor
    state = {
        'w': torch.tensor(policy.w, dtype=torch.float32),
        'lr': policy.lr
    }
    torch.save(state, path)
    print(f"Política salva em {path}")

def load_policy_pth(path):
    data = torch.load(path)
    w_list = data['w'].tolist()
    lr = data.get('lr', 0.005)
    policy = LinearPolicy(feature_len=len(w_list), lr=lr)
    policy.w = w_list
    print(f"Política carregada de {path}")
    return policy

# Uso (após treinar policy_mixed):
save_policy_pth(policy_mixed, 'policy_mixed.pth')