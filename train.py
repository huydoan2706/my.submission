import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils.DetectionLoss import DetectionLoss
from utils.ObjectDetectionDataset import ObjectDetectionDataset
from utils.SimpleDetector import SimpleDetector
from tqdm import tqdm


def collate_fn(batch):
    images, targets, ids, sizes = zip(*batch)
    images = torch.stack(images)
    return images, targets, ids, sizes


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    train_dataset = ObjectDetectionDataset(args.train_data, args.image_dir, img_size=args.img_size, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=args.workers, pin_memory=True)

    val_dataset = ObjectDetectionDataset(args.val_data, args.val_image_dir, img_size=args.img_size, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=True, collate_fn=collate_fn, num_workers=args.workers, pin_memory=True)

    model = SimpleDetector(num_classes=len(train_dataset.classes), grid_size=args.grid_size).to(device)
    criterion = DetectionLoss(num_classes=len(train_dataset.classes), grid_size=args.grid_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # TRAIN
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch: {epoch}/{args.epochs} [Train]')
        for images, targets, _, _ in pbar:
            images = images.to(device)
            preds = model(images)
            loss, loss_dict = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", **{k: f"{v:.3f}" for k, v in loss_dict.items()})

        avg_train_loss = train_loss / len(train_loader)

        # VAL
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets, _, _ in tqdm(val_loader, desc=f'Epoch: {epoch}/{args.epochs} [Val]'):
                images = images.to(device)
                preds = model(images)
                loss, _ = criterion(preds, targets)

                optimizer.zero_grad()
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step()
        print(f'Epoch: {epoch}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')

        # Save checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'classes': train_dataset.classes,
            'img_size': args.img_size,
            'grid_size': args.grid_size,
            'num_classes': len(train_dataset.classes),
            'val_loss': avg_val_loss
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'last.pth'))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(ckpt, os.path.join(args.checkpoint_dir, 'best.pth'))
            print(f"  → Saved best model (val_loss={best_val_loss:.4f})")

    print('Finished Training')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--val_image_dir", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="./models")
    parser.add_argument("--img_size", type=int, default=416)
    parser.add_argument("--grid_size", type=int, default=13)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    train(args)
