import torch
import numpy as np
import os
import sys
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import copy 

# --- CONFIGURATION ---
NUM_EPOCHS = 50       # Increased slightly to give it time to learn
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
PATIENCE = 10         # Early stopping patience
WEIGHT_DECAY = 1e-4 

# !!! PATHS !!!
VIDEO_FEATURE_DIR = "/Users/yaminmohammadbhat/Desktop/rPPG/marlin_features" 
HB_FEATURE_DIR = "/Users/yaminmohammadbhat/Desktop/rPPG/OUTPUT_HEARTBEAT"
RAW_DATASET_DIR = "/Users/yaminmohammadbhat/Desktop/rPPG/DATASET"

OUTPUT_MODEL_PATH = "fusion_model_best.pth"
OUTPUT_PLOT_PATH = "training_metrics.png"

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- MODEL ---
class FusionDeepfakeDetector(nn.Module):
    def __init__(self, video_input_dim=2048, hidden_dim=256, output_dim=2):
        super(FusionDeepfakeDetector, self).__init__()
        
        self.video_fc = nn.Sequential(
            nn.Linear(video_input_dim, 512), 
            nn.BatchNorm1d(512), 
            nn.ReLU(), 
            nn.Dropout(0.5), 
            nn.Linear(512, hidden_dim), 
            nn.ReLU()
        )
        
        self.hb_cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # 3 Input Channels
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), 
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)
        )
        
        self.hb_fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8 * 32, hidden_dim), nn.ReLU())
        
        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(0.4), 
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.video_classifier = nn.Linear(hidden_dim, output_dim)
        self.hb_classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, video_features, hb_features):
        video_rep = video_features.mean(dim=1) 
        video_emb = self.video_fc(video_rep)
        hb_emb = self.hb_cnn(hb_features)
        hb_emb = self.hb_fc(hb_emb)
        
        fused_input = torch.cat([video_emb, hb_emb], dim=1)
        output = self.fusion_fc(fused_input)
        
        video_pred = self.video_classifier(video_emb)
        hb_pred = self.hb_classifier(hb_emb)
        return output, video_pred, hb_pred

# --- DATASET ---
class DeepfakeDataset(torch.utils.data.Dataset):
    def __init__(self, video_dir, hb_dir, raw_dataset_dir):
        self.video_dir = video_dir; self.hb_dir = hb_dir; self.video_files = []
        self.fake_filenames = set()
        
        fake_path = os.path.join(raw_dataset_dir, "fake")
        if os.path.exists(fake_path):
            for f in os.listdir(fake_path): 
                self.fake_filenames.add(os.path.splitext(f)[0])
                
        for root, _, files in os.walk(video_dir):
            for f in files:
                if f.endswith("_video.npy"):
                    self.video_files.append(os.path.join(root, f))

    def __len__(self): return len(self.video_files)

    def __getitem__(self, idx):
        v_path = self.video_files[idx]
        filename = os.path.basename(v_path)
        base = filename.replace("_video.npy", "")
        
        is_fake = base in self.fake_filenames
        label = 1 if is_fake else 0
        subfolder = "fake" if is_fake else "real"
        hb_path = os.path.join(self.hb_dir, subfolder, base + "_superlet.npy")
        
        try: 
            vid = np.load(v_path)
            if vid.ndim == 1: vid = vid.reshape(1, -1)
        except: vid = np.zeros((30, 2048), dtype=np.float32)
        
        if os.path.exists(hb_path):
            try: 
                hb = np.load(hb_path).astype(np.float32)
                if hb.shape != (3, 64, 256):
                     if hb.shape == (64, 256): hb = np.stack([hb]*3, axis=0)
                     else: hb = np.zeros((3, 64, 256), dtype=np.float32)
            except: hb = np.zeros((3, 64, 256), dtype=np.float32)
        else: hb = np.zeros((3, 64, 256), dtype=np.float32)
            
        return torch.tensor(vid, dtype=torch.float32), torch.tensor(hb, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    v, h, l = zip(*batch)
    return pad_sequence(v, batch_first=True), torch.stack(h), torch.stack(l)

# --- METRIC CALCULATION HELPER ---
def compute_metrics(loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for v, h, l in loader:
            v, h, l = v.to(device), h.to(device), l.to(device)
            out, vp, hp = model(v, h)
            loss = criterion(out, l) + 0.5*criterion(vp, l) + 0.5*criterion(hp, l)
            running_loss += loss.item()
            
            _, predicted = torch.max(out, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(l.cpu().numpy())
            
    avg_loss = running_loss / len(loader) if len(loader) > 0 else 0
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, acc, prec, rec, f1, all_labels, all_preds

# --- MAIN LOOP ---
if __name__ == "__main__":
    full_dataset = DeepfakeDataset(VIDEO_FEATURE_DIR, HB_FEATURE_DIR, RAW_DATASET_DIR)
    
    if len(full_dataset) == 0:
        print("Error: No data found. Check your paths.")
        sys.exit()
        
    labels = [full_dataset[i][2].item() for i in range(len(full_dataset))]
    train_idx, test_idx = train_test_split(range(len(full_dataset)), test_size=0.2, stratify=labels, random_state=42)
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(full_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(full_dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = FusionDeepfakeDetector().to(device)
    
    n_tot = len(full_dataset); n_fake = sum(labels); n_real = n_tot - n_fake
    print(f"Dataset: {n_real} Real, {n_fake} Fake")
    
    if n_real > 0 and n_fake > 0:
        weights = torch.tensor([n_tot/(2*n_real), n_tot/(2*n_fake)], dtype=torch.float32).to(device)
    else:
        weights = torch.tensor([1.0, 1.0]).to(device)
        
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print(f"--- Training for {NUM_EPOCHS} epochs ---")
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | {'Val Acc':<8} | {'Val Prec':<8} | {'Val Rec':<8} | {'Val F1':<8}")
    print("-" * 85)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    for epoch in range(NUM_EPOCHS):
        # TRAIN STEP
        model.train()
        r_loss = 0.0
        for v, h, l in train_loader:
            v, h, l = v.to(device), h.to(device), l.to(device)
            out, vp, hp = model(v, h)
            loss = criterion(out, l) + 0.5*criterion(vp, l) + 0.5*criterion(hp, l)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            r_loss += loss.item()
        
        train_loss = r_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        # VALIDATION STEP (With Full Metrics)
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = compute_metrics(test_loader, model, criterion, device)
        
        scheduler.step(val_loss)
        
        # LOGGING
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"{epoch+1:<6} | {train_loss:<10.4f} | {val_loss:<10.4f} | {val_acc:<8.2%} | {val_prec:<8.2%} | {val_rec:<8.2%} | {val_f1:<8.2%}")

        # CHECKPOINT
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(best_model_wts, OUTPUT_MODEL_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nStop! No improvement for {PATIENCE} epochs.")
                break

    print(f"\nBest Model saved to {OUTPUT_MODEL_PATH}")

    # --- FINAL EVALUATION & CONFUSION MATRIX ---
    model.load_state_dict(best_model_wts)
    _, final_acc, final_prec, final_rec, final_f1, all_lbls, all_preds = compute_metrics(test_loader, model, criterion, device)
    
    print("\n--- FINAL TEST RESULTS ---")
    print(f"Accuracy:  {final_acc:.2%}")
    print(f"Precision: {final_prec:.2%}")
    print(f"Recall:    {final_rec:.2%}")
    print(f"F1 Score:  {final_f1:.2%}")
    
    cm = confusion_matrix(all_lbls, all_preds)
    print("\nConfusion Matrix:")
    print(f"True Real: {cm[0,0]}  | False Fake: {cm[0,1]}")
    print(f"False Real: {cm[1,0]} | True Fake:  {cm[1,1]}")

    # PLOT
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curve'); plt.legend(); plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.plot(history['val_f1'], label='Val F1 Score')
    plt.title('Performance Metrics'); plt.legend(); plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH)
    plt.show()