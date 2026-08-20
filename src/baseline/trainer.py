import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from PIL import Image
import json

class MemotionDataset(Dataset):
    def __init__(self, csv_file, img_dir, processor, tokenizer, max_len=128):
        self.df = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row.get("image_filename", "")
        img_path = self.img_dir / img_name
        
        if img_path.exists():
            image = Image.open(img_path).convert("RGB")
        else:
            image = Image.new("RGB", (224, 224))
            
        text = str(row.get("ocr", ""))
        label = float(row.get("is_humorous", 0.0))
        
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0) if self.processor else torch.zeros((3, 224, 224))
        
        if self.tokenizer:
            encodings = self.tokenizer(
                text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
            )
            input_ids = encodings["input_ids"].squeeze(0)
            attention_mask = encodings["attention_mask"].squeeze(0)
        else:
            input_ids = torch.zeros((self.max_len,), dtype=torch.long)
            attention_mask = torch.zeros((self.max_len,), dtype=torch.long)
            
        return {
            "sample_id": img_name,
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.float)
        }

def train_baseline(model, train_loader, val_loader, epochs, lr, device, save_dir):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    history = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            logits = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / max(1, len(train_loader))
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                logits = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / max(1, len(val_loader))
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        history.append({"epoch": epoch+1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            try:
                torch.save(model.state_dict(), save_dir / "best_model.pth")
                print(f"New best model saved with val_loss: {best_val_loss:.4f}")
            except Exception as e:
                print(f"Failed to save model (disk full?): {e}")
        
    try:
        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Failed to save history: {e}")
