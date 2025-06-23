import tkinter as tk
from tkinter import messagebox
import tkinter.font as tkFont
from PIL import Image, ImageTk
import os
import random
import math
from collections import Counter

# Define Card class, including 'index' for choosing image filename like "January{index}.jpg"
class Card:
    def __init__(self, month, kind, name, index):
        self.month = month
        self.kind = kind
        self.name = name
        self.index = index  # 1..4 for image mapping
        

    def __repr__(self):
        return f"{self.name}({self.month})"

    def __eq__(self, other):
        return isinstance(other, Card) and (
            self.month, self.kind, self.name, self.index
        ) == (other.month, other.kind, other.name, other.index)

    def __hash__(self):
        return hash((self.month, self.kind, self.name, self.index))


# Build the 48-card deck, assigning index per month in the order defined
def build_deck():
    cards = []
    month_defs = {
        1: [('bright', 'Crane'), ('ribbon', 'Red Poetry'), ('junk', 'Pine Chaff 1'), ('junk', 'Pine Chaff 2')],
        2: [('animal', 'Bush Warbler'), ('ribbon', 'Red Poetry'), ('junk', 'Plum Chaff 1'), ('junk', 'Plum Chaff 2')],
        3: [('bright', 'Curtain'), ('ribbon', 'Red Poetry'), ('junk', 'Cherry Chaff 1'), ('junk', 'Cherry Chaff 2')],
        4: [('animal', 'Cuckoo'), ('ribbon', 'Purple Ribbon'), ('junk', 'Wisteria Chaff 1'), ('junk', 'Wisteria Chaff 2')],
        5: [('animal', 'Eight-Plank Bridge'), ('ribbon', 'Blue Ribbon'), ('junk', 'Iris Chaff 1'), ('junk', 'Iris Chaff 2')],
        6: [('animal', 'Butterflies'), ('ribbon', 'Blue Ribbon'), ('junk', 'Peony Chaff 1'), ('junk', 'Peony Chaff 2')],
        7: [('animal', 'Boar'), ('ribbon', 'Red Ribbon'), ('junk', 'Bush Clover Chaff 1'), ('junk', 'Bush Clover Chaff 2')],
        8: [('animal', 'Geese'), ('ribbon', 'Red Ribbon'), ('junk', 'Pampas Chaff 1'), ('junk', 'Pampas Chaff 2')],
        9: [('animal', 'Sake Cup'), ('ribbon', 'Blue Ribbon'), ('junk', 'Chrysanthemum Chaff 1'), ('junk', 'Chrysanthemum Chaff 2')],
        10: [('bright', 'Moon'), ('ribbon', 'Blue Ribbon'), ('junk', 'Maple Chaff 1'), ('junk', 'Maple Chaff 2')],
        11: [('bright', 'Rainman'), ('animal', 'Swallow'), ('ribbon', 'Purple Ribbon'), ('junk', 'Willow Chaff')],
        12: [('bright', 'Phoenix'), ('junk', 'Paulownia Chaff 1'), ('junk', 'Paulownia Chaff 2'), ('junk', 'Paulownia Chaff 3')],
    }
    for month, defs in month_defs.items():
        for idx, (kind, name) in enumerate(defs, start=1):
            cards.append(Card(month, kind, name, idx))
    return cards.copy()

# Yaku detection and scoring, same as before
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

# Feature extraction and LinearPolicy class (matching your training implementation)
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

# Load policy weights from .pth using PyTorch
def load_policy_pth(path):
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch is required to load the model.")
    data = torch.load(path, map_location='cpu')
    w_list = data['w'].tolist() if hasattr(data['w'], 'tolist') else list(data['w'])
    lr = data.get('lr', 0.005)
    policy = LinearPolicy(feature_len=len(w_list), lr=lr)
    policy.w = w_list
    print(f"Policy loaded from {path}")
    return policy

# Month names mapping for image filenames
MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

class HanafudaGUI:
    def __init__(self, root, model_path, images_folder="./images", bg_path="background.png"):
        self.root = root
        self.root.title("Hanafuda Game - You vs AI")

        # === Background image setup ===
        if bg_path:
            try:
                from PIL import Image, ImageTk
                pil_bg = Image.open(bg_path)
                self._original_bg = pil_bg  # keep for resizing
                self.bg_photo = ImageTk.PhotoImage(pil_bg)
                self.bg_label = tk.Label(self.root, image=self.bg_photo)
                # Place behind everything:
                self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
                self.bg_label.lower()
                # If you want resizing:
                self.root.bind("<Configure>", self._on_bg_resize)
            except Exception as e:
                print(f"Warning: could not load background '{bg_path}': {e}")
        else:
            # If no image, you can set a solid color:
            self.root.configure(bg="#ffe4e1")  # e.g., Misty Rose

        # Now proceed with loading policy, images, frames, etc.
        self.images_folder = images_folder
        self.policy = load_policy_pth(model_path)
        self.load_images()
        self.setup_frames()
        self.restart_game()

    def _on_bg_resize(self, event):
        # Resize background when window size changes
        if not hasattr(self, "_original_bg"):
            return
        new_w, new_h = event.width, event.height
        if new_w < 50 or new_h < 50:
            return
        resized = self._original_bg.resize((new_w, new_h), Image.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(resized)
        self.bg_label.configure(image=self.bg_photo)

        

    def load_images(self):
        """Load images for all 48 cards into two caches: full-size e thumbnail.
        Tenta .png, .jpg, .jpeg; se não existir, usa placeholder."""
        self.img_cache = {}
        self.thumb_cache = {}
        # Determinar filtro de reamostragem compatível com a versão de Pillow
        try:
            resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            resample_filter = Image.LANCZOS

        for month_idx, month_name in enumerate(MONTH_NAMES, start=1):
            for i in range(1, 5):
                base = f"{month_name}{i}"
                loaded = False
                for ext in (".png", ".jpg", ".jpeg"):
                    filename = base + ext
                    path = os.path.join(self.images_folder, filename)
                    if os.path.exists(path):
                        try:
                            img = Image.open(path)
                            # Carrega a versão full-size
                            img_full = img.resize((80, 120), resample_filter)
                            photo_full = ImageTk.PhotoImage(img_full)
                            self.img_cache[(month_idx, i)] = photo_full
                            # Gera e armazena thumbnail
                            img_thumb = img.resize((40, 60), resample_filter)
                            photo_thumb = ImageTk.PhotoImage(img_thumb)
                            self.thumb_cache[(month_idx, i)] = photo_thumb
                            loaded = True
                            break
                        except Exception as e:
                            print(f"Warning: Falha ao carregar {filename}: {e}. Tentando próxima extensão ou placeholder.")
                if not loaded:
                    print(f"Warning: Imagem não encontrada para {base}(.png/.jpg). Usando placeholder.")
                    # Placeholder full-size e thumbnail
                    placeholder_full = Image.new("RGB", (80, 120), color=(200, 200, 200))
                    photo_full = ImageTk.PhotoImage(placeholder_full)
                    self.img_cache[(month_idx, i)] = photo_full
                    placeholder_thumb = Image.new("RGB", (40, 60), color=(200, 200, 200))
                    photo_thumb = ImageTk.PhotoImage(placeholder_thumb)
                    self.thumb_cache[(month_idx, i)] = photo_thumb

    def setup_frames(self):
        """Set up the main layout frames."""
        self.table_frame = tk.Frame(self.root, bg="#4E3F33")
        self.table_frame.pack(pady=10)

        self.user_caps_frame = tk.Frame(self.root, bg="#4E3F33")
        self.user_caps_frame.pack(pady=5)

        self.ai_caps_frame = tk.Frame(self.root, bg="#4E3F33")
        self.ai_caps_frame.pack(pady=5)

        self.hand_frame = tk.Frame(self.root, bg="#4E3F33")
        self.hand_frame.pack(pady=10)

        self.status_label = tk.Label(self.root, text="", bg="#4E3F33")
        self.status_label.pack(pady=5)

        self.restart_button = tk.Button(self.root, text="Restart Game", command=self.restart_game, bg="#4E3F33", fg="white")
        self.restart_button.pack(pady=5)

    def restart_game(self):
        """Initialize or reset game state: shuffle, deal, resets captures, etc."""
        deck = build_deck()
        random.shuffle(deck)
        self.user_hand = [deck.pop() for _ in range(8)]
        self.ai_hand = [deck.pop() for _ in range(8)]
        self.table = [deck.pop() for _ in range(8)]
        self.user_caps = []
        self.ai_caps = []
        self.deck = deck
        self.remain_counts = Counter(deck)
        self.current_turn = "user"  # User starts
        self.update_ui()
        self.status_label.config(text="Your turn. Click a card to play.", fg="white")

    def update_ui(self):
        """Redraw table cards, user captures, AI captures (com miniaturas) e user hand."""
        # Limpar widgets anteriores...
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        for widget in self.user_caps_frame.winfo_children():
            widget.destroy()
        for widget in self.ai_caps_frame.winfo_children():
            widget.destroy()
        for widget in self.hand_frame.winfo_children():
            widget.destroy()

        # Table display (igual antes)...
        tk.Label(self.table_frame, text="Table:", bg="#4E3F33", fg="white").pack()
        for card in self.table:
            img = self.img_cache.get((card.month, card.index))
            lbl = tk.Label(self.table_frame, image=img)
            lbl.image = img
            lbl.pack(side=tk.LEFT, padx=2)

        # User captures (igual antes)...
        tk.Label(self.user_caps_frame, text="Your Captures:", bg="#4E3F33", fg="white").pack()
        for card in self.user_caps:
            img = self.img_cache.get((card.month, card.index))
            lbl = tk.Label(self.user_caps_frame, image=img)
            lbl.image = img
            lbl.pack(side=tk.LEFT, padx=2)

        # AI captures: agora mostramos a thumbnail real
        tk.Label(self.ai_caps_frame, text=f"AI Captures ({len(self.ai_caps)}):", bg="#4E3F33", fg="white").pack()
        for card in self.ai_caps:
            img_thumb = self.thumb_cache.get((card.month, card.index))
            # Caso thumb_cache não tenha (teoricamente não vai, pois carregamos placeholder), 
            # podemos cair em um placeholder já salvo em thumb_cache
            lbl = tk.Label(self.ai_caps_frame, image=img_thumb)
            lbl.image = img_thumb
            lbl.pack(side=tk.LEFT, padx=2)

        # User hand (igual antes)...
        tk.Label(self.hand_frame, text="Your Hand:", bg="#4E3F33", fg="white").pack()
        for card in self.user_hand:
            img = self.img_cache.get((card.month, card.index))
            btn = tk.Button(self.hand_frame, image=img, command=lambda c=card: self.on_user_play(c))
            btn.image = img
            btn.pack(side=tk.LEFT, padx=2)

    def on_user_play(self, card):
        """Handle when the user clicks a card to play."""
        if self.current_turn != "user":
            return
        if card not in self.user_hand:
            return

        # Process user play onto table or capture
        self.process_play(card, self.user_hand, self.user_caps)

        # Draw from deck for user
        if self.deck:
            d = self.deck.pop(0)
            # Update remain_counts
            if self.remain_counts[d] > 0:
                self.remain_counts[d] -= 1
            matched = False
            for t in list(self.table):
                if t.month == d.month:
                    self.user_caps += [d, t]
                    self.table.remove(t)
                    matched = True
                    break
            if not matched:
                self.table.append(d)

        # Update UI after user move
        self.update_ui()

        # Check if game over (hand empty)
        if not self.user_hand:
            self.end_game()
            return

        # Trigger AI turn
        self.current_turn = "ai"
        self.status_label.config(text="AI's turn...")
        # slight delay so user sees the change
        self.root.after(500, self.ai_turn)

    def ai_turn(self):
        """AI plays a turn."""
        if not self.ai_hand:
            self.end_game()
            return

        # Choose via policy
        features = extract_features(self.ai_hand, self.table, self.ai_caps, self.remain_counts)
        idx, _ = self.policy.choose_action(features)
        card = self.ai_hand[idx]

        # Process AI play
        self.process_play(card, self.ai_hand, self.ai_caps)

        # Draw from deck for AI
        if self.deck:
            d = self.deck.pop(0)
            if self.remain_counts[d] > 0:
                self.remain_counts[d] -= 1
            matched = False
            for t in list(self.table):
                if t.month == d.month:
                    self.ai_caps += [d, t]
                    self.table.remove(t)
                    matched = True
                    break
            if not matched:
                self.table.append(d)

        # Update UI after AI move
        self.update_ui()

        # Check if game ends
        if not self.user_hand:
            self.end_game()
            return

        # Back to user
        self.current_turn = "user"
        self.status_label.config(text="Your turn. Click a card to play.")

    def process_play(self, card, hand, caps):
        """Common logic: play a card from hand onto table or into captures."""
        if card in hand:
            hand.remove(card)
            matched = False
            for t in list(self.table):
                if t.month == card.month:
                    caps += [card, t]
                    self.table.remove(t)
                    matched = True
                    break
            if not matched:
                self.table.append(card)

    def end_game(self):
        """Compute final scores, show result, allow restart."""
        user_score = score_captures(self.user_caps)
        ai_score = score_captures(self.ai_caps)
        result_msg = f"Game Over!\nYour score: {user_score}\nAI score: {ai_score}\n"
        if user_score > ai_score:
            result_msg += "You win! 🎉"
        elif user_score < ai_score:
            result_msg += "AI wins! 🤖"
        else:
            result_msg += "It's a draw!"
        messagebox.showinfo("Result", result_msg)
        self.status_label.config(text="Game over. You can restart the game.")

if __name__ == "__main__":
    # Adjust these if your model file or images folder are in different locations
    model_path = "policy_mixed.pth"   # Path to your trained .pth file
    images_folder = "./images"        # Folder containing "January1.jpg"/.png etc.
    root = tk.Tk()
    default_font = tkFont.nametofont("TkDefaultFont")
    default_font.configure(family="i ichimaru", size=12)  
    app = HanafudaGUI(root, model_path, images_folder)
    root.mainloop()
