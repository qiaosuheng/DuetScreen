"""
Data Preprocessing Module - Process FASTA files, replace degenerate bases, generate sequence fragments, and add downsampling
"""

import torch
import os
import random
from Bio import SeqIO
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import pickle

# Degenerate base mapping table (IUPAC standard)
degenerate_base_map = {
    'A': ['A'], 'T': ['T'], 'C': ['C'], 'G': ['G'],
    'M': ['A', 'C'], 'R': ['A', 'G'], 'W': ['A', 'T'],
    'S': ['C', 'G'], 'Y': ['C', 'T'], 'K': ['G', 'T'],
    'V': ['A', 'C', 'G'], 'H': ['A', 'C', 'T'],
    'D': ['A', 'G', 'T'], 'B': ['C', 'G', 'T'],
    'N': ['A', 'T', 'C', 'G'],
}


def replace_degenerate_bases(seq: str, rng: random.Random = None) -> str:
    """Replace degenerate bases in the sequence with corresponding standard bases (random selection)

    ``rng`` optionally supplies a dedicated random stream so prepared-split
    preprocessing stays reproducible without touching the global RNG state
    that training seeds own.
    """
    chooser = rng if rng is not None else random
    cleaned = []
    for base in seq.upper():
        possible_bases = degenerate_base_map.get(base)
        if possible_bases:
            cleaned.append(chooser.choice(possible_bases))
        else:
            cleaned.append(chooser.choice(degenerate_base_map['N']))
    return ''.join(cleaned)


def load_fasta_sequences(fasta_path: str, label: int) -> List[Dict]:
    """Load FASTA sequences, replace degenerate bases, and filter"""
    records = []
    try:
        for rec in SeqIO.parse(fasta_path, "fasta"):
            raw_seq = str(rec.seq)
            processed_seq = replace_degenerate_bases(raw_seq)
            
            # Filter sequences that are too short
            if len(processed_seq) >= 3:  # Minimum length, ensure at least one codon
                records.append({
                    "id": rec.id,
                    "seq_str": processed_seq,
                    "original_seq": processed_seq,
                    "label": label,
                    "original_label": label
                })
        return records
    except Exception as e:
        print(f"❌ Failed to load FASTA file: {fasta_path}, Error: {str(e)}")
        return []


def load_labeled_fasta(fasta_path: str, rng: random.Random = None) -> List[Dict]:
    """Load a single FASTA whose record IDs carry the label prefix pos_/neg_.

    This is the input format of nmi_dataset_release: one file per split,
    both classes mixed, labels encoded as pos_/neg_ ID prefixes.
    """
    records = []
    try:
        with open(fasta_path, "rt") as handle:
            for rec in SeqIO.parse(handle, "fasta"):
                rid = rec.id
                if rid.startswith("pos_"):
                    label = 1
                elif rid.startswith("neg_"):
                    label = 0
                else:
                    raise ValueError(
                        f"{fasta_path}: record ID {rid!r} lacks the pos_/neg_ label prefix")
                processed_seq = replace_degenerate_bases(str(rec.seq), rng=rng)
                if len(processed_seq) >= 3:
                    records.append({
                        "id": rid,
                        "seq_str": processed_seq,
                        "original_seq": processed_seq,
                        "label": label,
                        "original_label": label
                    })
        if not records:
            print(f"⚠️  No usable records in FASTA file: {fasta_path}")
        return records
    except ValueError:
        raise
    except Exception as e:
        print(f"❌ Failed to load FASTA file: {fasta_path}, Error: {str(e)}")
        return []


def generate_sequence_segments(full_seqs: List[Dict], max_length: int, min_length: int) -> List[Dict]:
    """Cut long sequences into fixed-length fragments"""
    segments = []
    
    for full_seq in full_seqs:
        orig_id = full_seq["id"]
        orig_seq = full_seq["seq_str"]
        orig_label = full_seq["label"]
        orig_len = len(orig_seq)
        
        start = 0
        seg_idx = 0
        
        while start < orig_len:
            # Calculate the end position of the current fragment
            end = min(start + max_length, orig_len)
            
            # Ensure length is a multiple of 3 (codon requirement)
            remainder = (end - start) % 3
            if remainder != 0:
                end -= remainder
            
            # If adjusted end position is less than or equal to start, cannot form a valid fragment
            if end <= start:
                break
                
            seg_len = end - start
            
            # If fragment length meets minimum requirement, add to results
            if seg_len >= min_length:
                segments.append({
                    "segment_id": f"{orig_id}_seg{seg_idx}",
                    "original_id": orig_id,
                    "original_seq": orig_seq[start:end],
                    "label": orig_label,
                    "original_label": orig_label,
                    "length": seg_len
                })
                seg_idx += 1
            
            # Update start position, prepare for next segment
            start = end
            
            # If remaining sequence length is less than minimum length, break out of loop
            if orig_len - start < min_length:
                break
    
    return segments


def remove_duplicate_segments(segments: List[Dict]) -> Tuple[List[Dict], int]:
    """Remove duplicate sequence fragments"""
    seen_seqs = set()
    unique_segs = []
    dup_count = 0
    
    for seg in segments:
        seq_str = seg["original_seq"]
        if seq_str not in seen_seqs:
            seen_seqs.add(seq_str)
            unique_segs.append(seg)
        else:
            dup_count += 1
    
    return unique_segs, dup_count


def downsample_negative_segments(segments: List[Dict], seed: int = 42, 
                                downsample_ratio: float = 3.0) -> List[Dict]:
    """
    Downsample non-pathogenic fragments to approximately the specified multiple of pathogenic fragments
    
    Args:
        segments: List of all fragments
        seed: Fixed data-sampling seed, independent of the training seed.
            Identical results require identical input contents and ordering.
        downsample_ratio: Downsampling ratio, non-pathogenic fragment count / pathogenic fragment count
    
    Returns:
        Downsampled fragment list
    """
    # Separate fragments by label
    positive_segments = [seg for seg in segments if seg['label'] == 1]
    negative_segments = [seg for seg in segments if seg['label'] == 0]
    
    positive_count = len(positive_segments)
    negative_count = len(negative_segments)
    
    print(f"📊 Pre-downsampling statistics:")
    print(f"  - Pathogenic fragments: {positive_count}")
    print(f"  - Non-pathogenic fragments: {negative_count}")
    print(f"  - Non-pathogenic:Pathogenic ratio: {negative_count/positive_count:.2f}:1")
    
    # If non-pathogenic fragment count is already less than or equal to target count, keep all
    target_negative_count = int(positive_count * downsample_ratio)
    
    if negative_count <= target_negative_count:
        print(f"⚠️  Non-pathogenic fragment count ({negative_count}) is already less than or equal to target ({target_negative_count}), skipping downsampling")
        return segments
    
    # Own the sampling stream: choosing a dataset must not reset the Python,
    # NumPy or PyTorch random states used later by training.
    sampling_rng = random.Random(seed)
    
    # Randomly sample target number of non-pathogenic fragments
    sampled_negative_segments = sampling_rng.sample(negative_segments, target_negative_count)
    
    # Merge pathogenic fragments and sampled non-pathogenic fragments
    downsampled_segments = positive_segments + sampled_negative_segments
    
    # Optional: Shuffle order (maintain randomness but reproducible)
    sampling_rng.shuffle(downsampled_segments)
    
    print(f"✅ Downsampling completed:")
    print(f"  - Pathogenic fragments after sampling: {positive_count}")
    print(f"  - Non-pathogenic fragments after sampling: {len(sampled_negative_segments)}")
    print(f"  - Non-pathogenic:Pathogenic ratio after sampling: {len(sampled_negative_segments)/positive_count:.2f}:1")
    print(f"  - Removed non-pathogenic fragments: {negative_count - target_negative_count}")
    
    return downsampled_segments


def preprocess_data(config) -> List[Dict]:
    """
    Main preprocessing function: Load FASTA files, replace degenerate bases, generate fragments, deduplicate, downsample
    
    Args:
        config: Configuration object (must include downsample_ratio attribute)
        
    Returns:
        List of preprocessed fragments
    """
    print(f"\n🔄 Starting data preprocessing")
    
    # Check if files exist
    if not os.path.exists(config.positive_fasta):
        print(f"⚠️  Positive sample file does not exist: {config.positive_fasta}")
        return []
    
    if not os.path.exists(config.negative_fasta):
        print(f"⚠️  Negative sample file does not exist: {config.negative_fasta}")
        return []
    
    # 1. Load and process original sequences
    print("📥 Loading positive sample sequences...")
    positive_seqs = load_fasta_sequences(config.positive_fasta, label=1)
    
    print("📥 Loading negative sample sequences...")
    negative_seqs = load_fasta_sequences(config.negative_fasta, label=0)
    
    all_seqs = positive_seqs + negative_seqs
    print(f"✅ Loading completed: {len(positive_seqs)} positive samples, {len(negative_seqs)} negative samples")
    
    # 2. Generate fragments
    print("✂️  Generating sequence fragments...")
    all_segments = generate_sequence_segments(
        all_seqs, 
        max_length=config.max_seq_length,
        min_length=config.min_segment_len
    )
    print(f"✅ Fragment generation: {len(all_segments)} fragments total")
    
    # 3. Deduplicate
    print("🔄 Removing duplicate fragments...")
    unique_segments, dup_count = remove_duplicate_segments(all_segments)
    print(f"✅ Deduplication completed: {len(all_segments)} before, {len(unique_segments)} after, removed {dup_count}")
    
    # 4. Downsample non-pathogenic fragments
    if hasattr(config, 'downsample_ratio') and config.downsample_ratio > 0:
        print(f"\n📉 Downsampling non-pathogenic fragments (target ratio non-pathogenic:pathogenic = {config.downsample_ratio}:1)...")
        downsampled_segments = downsample_negative_segments(
            unique_segments, 
            # Keep data selection independent of config.seed (training randomness).
            seed=getattr(config, 'sampling_seed', 42),
            downsample_ratio=config.downsample_ratio
        )
        unique_segments = downsampled_segments
    
    # 5. Statistics
    positive_count = sum(1 for seg in unique_segments if seg['label'] == 1)
    negative_count = len(unique_segments) - positive_count
    avg_length = sum(len(seg['original_seq']) for seg in unique_segments) / len(unique_segments) if unique_segments else 0
    
    print(f"\n📊 Final preprocessing statistics:")
    print(f"  - Total fragments: {len(unique_segments)}")
    print(f"  - Pathogenic fragments: {positive_count} ({positive_count/len(unique_segments)*100:.1f}%)")
    print(f"  - Non-pathogenic fragments: {negative_count} ({negative_count/len(unique_segments)*100:.1f}%)")
    print(f"  - Non-pathogenic:Pathogenic ratio: {negative_count/positive_count:.2f}:1" if positive_count > 0 else "  - No pathogenic fragments")
    print(f"  - Average sequence length: {avg_length:.1f} bp")
    
    return unique_segments


def preprocess_prepared_split(config) -> Dict[str, List[Dict]]:
    """Preprocess the delivered train/val FASTA pair without re-splitting.

    Reads one mixed-class FASTA per split (labels from pos_/neg_ ID prefixes),
    applies the same segmentation rules as the legacy pipeline
    (<= max_seq_length, segment lengths kept to multiples of 3), and tags every
    segment with its "split" so downstream stages never re-partition.
    Exact-content dedup and negative downsampling default OFF to preserve the
    delivered record composition; both remain configurable.

    Returns {"train": segments, "val": segments}.
    """
    print("\n🔄 Starting prepared-split preprocessing (delivered train/val files)")

    rng = random.Random(getattr(config, "sampling_seed", 42))
    result: Dict[str, List[Dict]] = {}

    for split_name, path in (("train", config.prepared_train_fasta),
                             ("val", config.prepared_val_fasta)):
        if not os.path.exists(path):
            print(f"⚠️  Prepared-split {split_name} file does not exist: {path}")
            return {}

        seqs = load_labeled_fasta(path, rng=rng)
        segs = generate_sequence_segments(
            seqs,
            max_length=config.max_seq_length,
            min_length=config.min_segment_len
        )

        if getattr(config, "remove_duplicate_segments", False):
            before = len(segs)
            segs, dup_count = remove_duplicate_segments(segs)
            print(f"  [{split_name}] dedup: {before} -> {len(segs)} (removed {dup_count})")

        if getattr(config, "downsample_ratio", 0) > 0 and split_name == "train":
            segs = downsample_negative_segments(
                segs,
                seed=getattr(config, "sampling_seed", 42),
                downsample_ratio=config.downsample_ratio
            )

        for seg in segs:
            seg["split"] = split_name

        pos = sum(1 for s in segs if s["label"] == 1)
        neg = len(segs) - pos
        if pos == 0 or neg == 0:
            print(f"❌ Prepared {split_name} split lacks one class (pos={pos}, neg={neg})")
            return {}
        avg_len = (sum(len(s["original_seq"]) for s in segs) / len(segs)) if segs else 0
        print(f"  [{split_name}] records={len(seqs)} pos={pos} neg={neg} "
              f"avg_len={avg_len:.1f}bp")
        result[split_name] = segs

    return result


def save_preprocessed_data(segments: List[Dict], cache_path: str):
    """Save preprocessed data to cache file"""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    # Ensure data structure is correct
    clean_segments = []
    for seg in segments:
        clean_seg = {
            'original_seq': seg.get('original_seq', ''),
            'label': seg.get('label', 0),
            'segment_id': seg.get('segment_id', f"seq_{len(clean_segments)}")
        }
        clean_segments.append(clean_seg)
    
    # Save as pickle file
    with open(cache_path, 'wb') as f:
        pickle.dump(clean_segments, f)
    
    print(f"💾 Preprocessed data saved to: {cache_path}")
    return cache_path


def load_preprocessed_data(cache_path: str) -> List[Dict]:
    """Load preprocessed data from cache file"""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                segments = pickle.load(f)
            print(f"✅ Loaded preprocessed data from cache: {cache_path} ({len(segments)} fragments)")
            return segments
        except Exception as e:
            print(f"❌ Cache loading failed: {str(e)}")
    return None


# Test function
if __name__ == "__main__":
    print("🧪 Testing data preprocessing functionality...")
    
    # Create test configuration
    class TestConfig:
        positive_fasta = "test_positive.fasta"
        negative_fasta = "test_negative.fasta"
        max_seq_length = 300
        min_segment_len = 30
        downsample_ratio = 3.0  # Non-pathogenic fragments are 3x pathogenic fragments
        seed = 42  # Random seed
    
    config = TestConfig()
    
    # Test degenerate base replacement
    test_seq = "ATNGCATMRWSN"
    cleaned_seq = replace_degenerate_bases(test_seq)
    print(f"Degenerate base replacement test: {test_seq} -> {cleaned_seq}")
    
    # Test downsampling function
    test_segments = [
        {"segment_id": f"pos_{i}", "original_seq": "ATCG"*10, "label": 1} for i in range(10)
    ] + [
        {"segment_id": f"neg_{i}", "original_seq": "GCTA"*10, "label": 0} for i in range(100)
    ]
    
    print(f"\nTesting downsampling: 10 pathogenic fragments, 100 non-pathogenic fragments")
    downsampled = downsample_negative_segments(test_segments, seed=42, downsample_ratio=3.0)
    
    pos_count = sum(1 for seg in downsampled if seg['label'] == 1)
    neg_count = sum(1 for seg in downsampled if seg['label'] == 0)
    print(f"After downsampling: pathogenic fragments={pos_count}, non-pathogenic fragments={neg_count}, ratio={neg_count/pos_count}:1")
    
    # Test preprocessing pipeline
    try:
        segments = preprocess_data(config)
        if segments:
            print(f"Preprocessing test successful: generated {len(segments)} fragments")
    except Exception as e:
        print(f"Preprocessing test failed: {e}")
    
    print("✅ Data preprocessing test completed")
