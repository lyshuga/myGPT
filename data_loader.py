import os
import torch
from torch.utils.data import Dataset
import tiktoken

class TextDataset(Dataset):
    def __init__(self, file_path, block_size, split='train'):
        self.block_size = block_size
        
        # Read text
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Tokenize
        enc = tiktoken.get_encoding("gpt2")
        data_ids = enc.encode(text)
        self.data = torch.tensor(data_ids, dtype=torch.long)
        
        # Split train/val
        n = int(0.9 * len(self.data))
        if split == 'train':
            self.data = self.data[:n]
        else:
            self.data = self.data[n:]
            
    def __len__(self):
        # We return the number of possible chunks of size block_size
        # Using a stride of block_size to ensure we cover data efficiently
        # If we wanted purely random sampling like the original, we could handle it differently,
        # but for DataLoader, chunking is standard.
        return (len(self.data) - self.block_size) // self.block_size

    def __getitem__(self, idx):
        # Calculate start index based on block_size stride
        start_idx = idx * self.block_size
        # Ensure we don't go out of bounds (though len calculation should prevent this)
        if start_idx + self.block_size + 1 > len(self.data):
             # Wrap around or handle edge case - simple truncation for now or adjust len
             start_idx = len(self.data) - self.block_size - 1

        x = self.data[start_idx : start_idx + self.block_size]
        y = self.data[start_idx + 1 : start_idx + self.block_size + 1]
        return x, y


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    # Configuration
    FILE_PATH = "input.txt"
    BLOCK_SIZE = 64
    BATCH_SIZE = 4
    
    print("=" * 60)
    print("TextDataset Test Cases")
    print("=" * 60)
    
    # Test 1: Basic initialization (train split)
    print("\n[Test 1] Initialize train dataset...")
    try:
        train_dataset = TextDataset(FILE_PATH, BLOCK_SIZE, split='train')
        print(f"  ✓ Train dataset created successfully")
        print(f"  - Total tokens in train split: {len(train_dataset.data)}")
        print(f"  - Block size: {BLOCK_SIZE}")
        print(f"  - Number of samples (chunks): {len(train_dataset)}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Test 2: Basic initialization (val split)
    print("\n[Test 2] Initialize val dataset...")
    try:
        val_dataset = TextDataset(FILE_PATH, BLOCK_SIZE, split='val')
        print(f"  ✓ Val dataset created successfully")
        print(f"  - Total tokens in val split: {len(val_dataset.data)}")
        print(f"  - Number of samples (chunks): {len(val_dataset)}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Test 3: __getitem__ - check shapes and types
    print("\n[Test 3] Test __getitem__ (single sample)...")
    try:
        x, y = train_dataset[0]
        assert x.shape == (BLOCK_SIZE,), f"x shape mismatch: {x.shape}"
        assert y.shape == (BLOCK_SIZE,), f"y shape mismatch: {y.shape}"
        assert x.dtype == torch.long, f"x dtype mismatch: {x.dtype}"
        assert y.dtype == torch.long, f"y dtype mismatch: {y.dtype}"
        print(f"  ✓ Shapes correct: x={x.shape}, y={y.shape}")
        print(f"  ✓ Dtypes correct: x={x.dtype}, y={y.dtype}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Test 4: Verify x and y are offset by 1
    print("\n[Test 4] Verify y is x shifted by 1 position...")
    try:
        x, y = train_dataset[0]
        # y should be x shifted by 1 (next token prediction)
        assert torch.equal(x[1:], y[:-1]), "y is not x shifted by 1"
        print(f"  ✓ y is correctly shifted by 1 from x")
        print(f"  - First 5 tokens of x: {x[:5].tolist()}")
        print(f"  - First 5 tokens of y: {y[:5].tolist()}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Test 5: Test multiple indices
    print("\n[Test 5] Test multiple indices...")
    try:
        for idx in [0, 1, len(train_dataset) - 1]:
            x, y = train_dataset[idx]
            assert x.shape == (BLOCK_SIZE,), f"Index {idx}: x shape mismatch"
            assert y.shape == (BLOCK_SIZE,), f"Index {idx}: y shape mismatch"
        print(f"  ✓ All indices [0, 1, {len(train_dataset)-1}] return correct shapes")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Test 6: DataLoader integration
    print("\n[Test 6] Test with DataLoader...")
    try:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        batch_x, batch_y = next(iter(train_loader))
        assert batch_x.shape == (BATCH_SIZE, BLOCK_SIZE), f"batch_x shape: {batch_x.shape}"
        assert batch_y.shape == (BATCH_SIZE, BLOCK_SIZE), f"batch_y shape: {batch_y.shape}"
        print(f"  ✓ DataLoader works correctly")
        print(f"  - Batch x shape: {batch_x.shape}")
        print(f"  - Batch y shape: {batch_y.shape}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Test 7: Decode sample to verify tokenization
    print("\n[Test 7] Decode sample tokens...")
    try:
        enc = tiktoken.get_encoding("gpt2")
        x, y = train_dataset[0]
        decoded_x = enc.decode(x.tolist())
        print(f"  ✓ Decoding works")
        print(f"  - Sample text (first 100 chars): {repr(decoded_x[:100])}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Test 8: Train/val split ratio
    print("\n[Test 8] Verify train/val split ratio (~90/10)...")
    try:
        train_tokens = len(train_dataset.data)
        val_tokens = len(val_dataset.data)
        total = train_tokens + val_tokens
        train_ratio = train_tokens / total
        val_ratio = val_tokens / total
        print(f"  - Train: {train_tokens} tokens ({train_ratio:.1%})")
        print(f"  - Val: {val_tokens} tokens ({val_ratio:.1%})")
        assert 0.85 < train_ratio < 0.95, "Train ratio not ~90%"
        print(f"  ✓ Split ratio is correct (~90/10)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Test 9: Get a batch from DataLoader
    print("\n[Test 9] Get a batch from DataLoader...")
    try:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        x_batch, y_batch = next(iter(train_loader))
        print(f"  - x_batch shape: {x_batch.shape}")
        print(f"  - y_batch shape: {y_batch.shape}")
        print(f"  - x_batch dtype: {x_batch.dtype}")
        print(f"  - y_batch dtype: {y_batch.dtype}")
        print(f"  - x_batch[0][:10]: {x_batch[0][:10].tolist()}")
        print(f"  - y_batch[0][:10]: {y_batch[0][:10].tolist()}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
