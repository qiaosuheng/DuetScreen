import random
import math
import warnings
import multiprocessing
from typing import List, Dict, Any, Tuple
import numpy as np
from collections import defaultdict

# --------------------------
# Constants: Amino acid substitution rules and codon mapping
# --------------------------
conservative_aa_replacements = {
    'A': ['V', 'L', 'I', 'G'], 'V': ['A', 'L', 'I', 'M'], 'L': ['A', 'V', 'I', 'M', 'F'], 
    'I': ['A', 'V', 'L', 'M'], 'M': ['V', 'L', 'I', 'F'], 'F': ['Y', 'W', 'L', 'M'], 
    'Y': ['F', 'W', 'S', 'T'], 'W': ['F', 'Y', 'M'], 'S': ['T', 'N', 'Q', 'C'], 
    'T': ['S', 'N', 'Q', 'C'], 'N': ['Q', 'S', 'T', 'D'], 'Q': ['N', 'S', 'T', 'E'], 
    'C': ['S', 'T', 'A'], 'K': ['R', 'H', 'Q'], 'R': ['K', 'H', 'Q'], 
    'H': ['K', 'R', 'Y'], 'D': ['E', 'N', 'S'], 'E': ['D', 'Q', 'T'], 
    'G': ['A', 'P', 'S'], 'P': ['G', 'A', 'S'], '*': []
}

codon_to_aa = {
    'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L', 'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
    'UAU': 'Y', 'UAC': 'Y', 'UAA': '*', 'UAG': '*', 'UGU': 'C', 'UGC': 'C', 'UGA': '*', 'UGG': 'W',
    'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L', 'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q', 'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M', 'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K', 'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V', 'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E', 'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}

# Amino acid to codon reverse mapping
aa_to_codons = defaultdict(list)
for codon, aa in codon_to_aa.items():
    if aa != '*':
        aa_to_codons[aa].append(codon)

all_valid_aas = [aa for aa in aa_to_codons.keys() if aa != '*']

# --------------------------
# Dinucleotide free energy data (kcal/mol)
# --------------------------
dimer_free_energy = {
    'AA': -1.00, 'AT': -0.88, 'AC': -1.45, 'AG': -1.30,
    'TA': -0.58, 'TT': -1.00, 'TC': -1.30, 'TG': -1.45,
    'CA': -1.45, 'CT': -1.30, 'CC': -2.17, 'CG': -2.24,
    'GA': -1.30, 'GT': -1.45, 'GC': -2.24, 'GG': -2.17,
    # RNA dinucleotide free energy
    'UU': -1.00, 'UA': -0.88, 'UC': -1.45, 'UG': -1.30,
    'AU': -0.58, 'AA': -1.00, 'AC': -1.30, 'AG': -1.45,
    'CU': -1.45, 'CA': -1.30, 'CC': -2.17, 'CG': -2.24,
    'GU': -1.30, 'GA': -1.45, 'GC': -2.24, 'GG': -2.17
}

# --------------------------
# Helper functions
# --------------------------
def calculate_sequence_differences(orig_seq: str, var_seq: str) -> Dict[str, float]:
    """Calculate sequence difference statistics"""
    min_base_len = min(len(orig_seq), len(var_seq))
    base_diff_count = sum(1 for o, v in zip(orig_seq[:min_base_len], var_seq[:min_base_len]) if o != v)
    base_diff_ratio = base_diff_count / min_base_len if min_base_len > 0 else 0.0

    def dna_to_aa(dna: str) -> str:
        rna = dna.replace('T', 'U')
        codons = [rna[i:i+3] for i in range(0, len(rna), 3) if len(rna[i:i+3]) == 3]
        return ''.join([codon_to_aa.get(c, '?') for c in codons])
    
    orig_aa = dna_to_aa(orig_seq)
    var_aa = dna_to_aa(var_seq)
    min_aa_len = min(len(orig_aa), len(var_aa))
    aa_diff_count = sum(1 for o, v in zip(orig_aa[:min_aa_len], var_aa[:min_aa_len]) if o != v)
    aa_diff_ratio = aa_diff_count / min_aa_len if min_aa_len > 0 else 0.0

    return {
        "base_diff_count": base_diff_count, "base_diff_ratio": round(base_diff_ratio, 4),
        "aa_diff_count": aa_diff_count, "aa_diff_ratio": round(aa_diff_ratio, 4),
        "orig_base_len": len(orig_seq), "var_base_len": len(var_seq)
    }


def calculate_dimer_free_energy(sequence: str) -> float:
    """Accurate calculation based on dinucleotide free energy"""
    rna_seq = sequence.replace('T', 'U').upper()
    
    if len(rna_seq) < 2:
        return 0.0
    
    total_energy = 0.0
    dimer_count = 0
    
    for i in range(len(rna_seq) - 1):
        dimer = rna_seq[i:i+2]
        if dimer in dimer_free_energy:
            total_energy += dimer_free_energy[dimer]
            dimer_count += 1
    
    return total_energy / dimer_count if dimer_count > 0 else 0.0


def calculate_conservative_energy_change(original_codon: str, new_codon: str) -> float:
    """Calculate free energy change caused by conservative substitution"""
    if len(original_codon) != 3 or len(new_codon) != 3:
        return 0.0
    
    # Calculate free energy of original codon
    original_energy = calculate_dimer_free_energy(original_codon)
    
    # Calculate free energy of new codon
    new_energy = calculate_dimer_free_energy(new_codon)
    
    # Return free energy change
    return abs(new_energy - original_energy)


def balanced_codon_sampling(
    target_codons: List[str],
    used_codons: List[str] = None
) -> str:
    """Balanced codon sampling"""
    used_codons = used_codons or []
    available_codons = [c for c in target_codons if c not in used_codons]
    if not available_codons:
        available_codons = target_codons
    
    # Uniform sampling
    return random.choice(available_codons)


def get_codon_base_diff(codon1: str, codon2: str) -> int:
    """Calculate the number of base differences between two codons"""
    if len(codon1) != 3 or len(codon2) != 3:
        return 3
    return sum(c1 != c2 for c1, c2 in zip(codon1, codon2))


def reverse_complement(dna_sequence: str) -> str:
    """Generate reverse complement of DNA sequence"""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(dna_sequence.upper()))


# --------------------------
# Core: Semantic similarity variant generation (first positive sample)
# --------------------------

def generate_positive_variant(
    seq_str: str,
    conservative_energy_tolerance: float = 1.0,  # Relaxed energy tolerance
    min_pos_base_diff: float = 0.02,  # Reduced minimum difference
    max_pos_base_diff: float = 0.25,  # Increased maximum difference
    pos_conservative_ratio: float = 0.05,  # Reduced conservative substitution ratio
    pos_syn_ratio: float = 0.8,  # Increased synonymous substitution ratio
    max_replace_ratio: float = 0.4,  # Increased maximum substitution ratio
    max_attempts: int = 500,  # Increased number of attempts
    min_conservative_changes: int = 1  # Minimum number of conservative substitutions
) -> str:
    """Generate a semantic similarity positive sample variant for a sequence (optimized version, increased success rate)"""
    
    def get_replaceable_positions(codons):
        """Precompute positions that can be substituted"""
        syn_positions = []
        conservative_positions = []
        
        for i, codon in enumerate(codons):
            original_aa = codon_to_aa.get(codon)
            if not original_aa or original_aa == '*':
                continue
                
            # Check if synonymous substitution is possible
            if len(aa_to_codons.get(original_aa, [])) > 1:
                syn_positions.append(i)
                
            # Check if conservative substitution is possible
            if original_aa in conservative_aa_replacements:
                conservative_positions.append(i)
        
        return syn_positions, conservative_positions
    
    def fallback_strategy(codons, used_codons):
        """Fallback strategy: use only synonymous substitutions"""
        variant_codons = codons.copy()
        codon_count = len(codons)
        
        # Calculate number of substitutions needed
        min_changes = max(1, int(codon_count * 0.05))  # At least 5% of codons
        max_changes = int(codon_count * 0.25)  # At most 25% of codons
        replace_count = random.randint(min_changes, max_changes)
        
        # Select replaceable positions
        candidate_positions = []
        for i, codon in enumerate(codons):
            original_aa = codon_to_aa.get(codon)
            if original_aa and original_aa != '*' and len(aa_to_codons.get(original_aa, [])) > 1:
                candidate_positions.append(i)
        
        if len(candidate_positions) < min_changes:
            return None
        
        random.shuffle(candidate_positions)
        selected_positions = candidate_positions[:replace_count]
        changes_made = 0
        
        for pos in selected_positions:
            current_codon = variant_codons[pos]
            original_aa = codon_to_aa.get(current_codon)
            target_codons = [c for c in aa_to_codons.get(original_aa, []) 
                            if c != current_codon and codon_to_aa[c] != '*']
            
            if target_codons:
                selected_codon = balanced_codon_sampling(target_codons, used_codons[pos])
                variant_codons[pos] = selected_codon
                changes_made += 1
        
        return ''.join(variant_codons).replace('U', 'T') if changes_made > 0 else None

    # Main function starts
    seq_rna = seq_str.replace('T', 'U')
    if len(seq_rna) % 3 != 0:
        raise ValueError(f"Sequence length must be a multiple of 3 (current: {len(seq_rna)})")
    
    codon_count = len(seq_rna) // 3
    if codon_count < 3:
        raise ValueError(f"Sequence too short ({codon_count} codons), cannot generate valid variant")

    seq_codons = [seq_rna[i:i+3] for i in range(0, len(seq_rna), 3)]
    
    # Precompute replaceable positions
    all_syn_positions, all_conservative_positions = get_replaceable_positions(seq_codons)
    
    # If no replaceable positions, directly use fallback strategy
    if len(all_syn_positions) < 2 and len(all_conservative_positions) < 1:
        fallback_variant = fallback_strategy(seq_codons, defaultdict(list))
        if fallback_variant:
            diff_stats = calculate_sequence_differences(seq_str, fallback_variant)
            print(f"✅ Direct fallback strategy: Sequence difference={diff_stats['base_diff_ratio']:.3f}")
            return fallback_variant
        else:
            warnings.warn(f"Sequence {seq_str[:20]}... has no replaceable positions")
            return ""

    for attempt in range(max_attempts):
        # Calculate number of substitution positions
        max_replace_count = max(1, int(codon_count * max_replace_ratio))
        min_replace_count = max(1, int(codon_count * min_pos_base_diff / 3))
        replace_count = random.randint(min_replace_count, max_replace_count)
        
        # Allocate substitution types
        conservative_count = max(0, int(round(replace_count * pos_conservative_ratio)))
        syn_count = max(0, replace_count - conservative_count)
        
        # Ensure sufficient replaceable positions
        available_conservative = min(conservative_count, len(all_conservative_positions))
        available_syn = min(syn_count, len(all_syn_positions))
        
        # If one substitution type lacks positions, adjust the other type
        if available_conservative < conservative_count:
            available_syn += (conservative_count - available_conservative)
            available_conservative = 0
        if available_syn < syn_count:
            available_conservative = min(available_conservative + (syn_count - available_syn), len(all_conservative_positions))
            available_syn = len(all_syn_positions)
        
        # Select substitution positions
        conservative_positions = random.sample(all_conservative_positions, available_conservative) if available_conservative > 0 else []
        syn_positions = random.sample(all_syn_positions, available_syn) if available_syn > 0 else []

        variant_codons = seq_codons.copy()
        position_used_codons = defaultdict(list)
        valid_replace = False
        total_conservative_energy_change = 0.0
        conservative_changes_made = 0

        # Phase 1: Synonymous substitutions (higher success rate)
        for pos in syn_positions:
            current_codon = variant_codons[pos]
            original_aa = codon_to_aa.get(current_codon)
            if not original_aa or original_aa == '*':
                continue
            
            target_codons = [c for c in aa_to_codons.get(original_aa, []) 
                            if c != current_codon and codon_to_aa[c] != '*']
            
            if target_codons:
                selected_codon = balanced_codon_sampling(target_codons, position_used_codons[pos])
                variant_codons[pos] = selected_codon
                position_used_codons[pos].append(selected_codon)
                valid_replace = True

        # Phase 2: Conservative substitutions (more lenient conditions)
        for pos in conservative_positions:
            # If minimum requirement met and energy change is large, stop conservative substitutions
            if (conservative_changes_made >= min_conservative_changes and 
                total_conservative_energy_change > conservative_energy_tolerance * 0.7):
                break
                
            current_codon = variant_codons[pos]
            original_aa = codon_to_aa.get(current_codon)
            if not original_aa or original_aa == '*':
                continue
                
            # Expand acceptable substitution options
            candidate_codons = []
            if original_aa in conservative_aa_replacements:
                target_aas = conservative_aa_replacements[original_aa]
                for aa in target_aas:
                    candidate_codons.extend([c for c in aa_to_codons.get(aa, []) 
                                          if codon_to_aa[c] != '*'])
            
            # If no conservative substitution options, skip
            if not candidate_codons:
                continue
                
            candidate_codons = [c for c in candidate_codons if c != current_codon]
            
            if candidate_codons:
                # Prioritize substitutions with smaller energy change
                energy_sorted_candidates = sorted(
                    candidate_codons,
                    key=lambda c: calculate_conservative_energy_change(current_codon, c)
                )
                
                # Try the first 3 candidates with smallest energy
                for selected_codon in energy_sorted_candidates[:3]:
                    energy_change = calculate_conservative_energy_change(current_codon, selected_codon)
                    
                    # Check if cumulative energy change is within tolerance
                    if total_conservative_energy_change + energy_change <= conservative_energy_tolerance:
                        variant_codons[pos] = selected_codon
                        position_used_codons[pos].append(selected_codon)
                        total_conservative_energy_change += energy_change
                        conservative_changes_made += 1
                        valid_replace = True
                        break

        if not valid_replace:
            continue

        # Check generated variant
        variant_rna_str = ''.join(variant_codons)
        variant_dna_str = variant_rna_str.replace('U', 'T')
        
        diff_stats = calculate_sequence_differences(seq_str, variant_dna_str)
        base_diff_ratio = diff_stats['base_diff_ratio']

        # Relaxed validation conditions
        if (total_conservative_energy_change <= conservative_energy_tolerance
            and min_pos_base_diff <= base_diff_ratio <= max_pos_base_diff
            and variant_dna_str != seq_str
            and conservative_changes_made >= min_conservative_changes):
            print(f"✅ Semantic variant generated successfully: Conservative substitutions={conservative_changes_made}, "
                  f"Free energy change={total_conservative_energy_change:.3f}, "
                  f"Sequence difference={base_diff_ratio:.3f}")
            return variant_dna_str

    # Primary strategy failed, try fallback strategy
    print(f"⚠️ Primary strategy failed, trying fallback strategy...")
    fallback_variant = fallback_strategy(seq_codons, defaultdict(list))
    if fallback_variant:
        diff_stats = calculate_sequence_differences(seq_str, fallback_variant)
        print(f"✅ Fallback strategy successful: Sequence difference={diff_stats['base_diff_ratio']:.3f}")
        return fallback_variant

    # If still failing, return warning
    warnings.warn(f"Cannot generate semantic similarity variant for sequence {seq_str[:20]}... (attempted {max_attempts} times + fallback strategy)")
    return ""

# --------------------------
# Core: DNA confusion variant generation (second positive sample) - free energy not considered
# --------------------------
def generate_dna_confusion_variant(
    original_seq: str,
    min_fragment_len: int = 30,
    max_fragment_len: int = 100,
    min_fragments: int = 3,
    max_fragments: int = 8,
    flip_ratio: float = 0.3,
    max_attempts: int = 50
) -> str:
    """
    Generate positive sample variant through DNA confusion
    Strategy: Fragmentation + random reverse complement
    Note: Free energy changes not considered
    """
    seq_len = len(original_seq)
    
    # Dynamically adjust parameters based on sequence length
    if seq_len < 100:
        min_fragment_len = max(15, seq_len // 4)
        max_fragment_len = min(60, seq_len // 2)
        min_fragments = 2
        max_fragments = min(4, seq_len // min_fragment_len)
        flip_ratio = 0.4
    elif seq_len > 1000:
        min_fragment_len = 80
        max_fragment_len = min(300, seq_len // 4)
        min_fragments = 4
        max_fragments = min(15, seq_len // min_fragment_len)
        flip_ratio = 0.25
    
    # Check if sequence is still too short
    if seq_len < min_fragment_len * min_fragments:
        warnings.warn(f"Sequence too short ({seq_len}), cannot perform effective DNA confusion (need at least {min_fragment_len * min_fragments} bp)")
        return ""
    
    for attempt in range(max_attempts):
        try:
            # 1. Determine number of fragments
            num_fragments = random.randint(min_fragments, max_fragments)
            
            # 2. Generate fragment boundaries
            fragment_boundaries = []
            remaining_len = seq_len
            current_pos = 0
            
            for i in range(num_fragments - 1):
                max_possible_len = min(
                    max_fragment_len,
                    remaining_len - (num_fragments - i - 1) * min_fragment_len
                )
                
                if max_possible_len < min_fragment_len:
                    num_fragments = i + 1
                    break
                
                fragment_len = random.randint(min_fragment_len, max_possible_len)
                fragment_boundaries.append((current_pos, current_pos + fragment_len))
                current_pos += fragment_len
                remaining_len -= fragment_len
            
            # Add the last fragment
            if current_pos < seq_len:
                last_fragment_len = seq_len - current_pos
                if last_fragment_len >= min_fragment_len:
                    fragment_boundaries.append((current_pos, seq_len))
                else:
                    if fragment_boundaries:
                        last_start, last_end = fragment_boundaries.pop()
                        fragment_boundaries.append((last_start, seq_len))
                    else:
                        continue
            
            if not fragment_boundaries:
                continue
            
            # 3. Randomly select fragments to reverse complement
            flip_indices = set()
            num_to_flip = max(1, int(len(fragment_boundaries) * flip_ratio))
            if len(fragment_boundaries) > 1:
                flip_indices = set(random.sample(range(len(fragment_boundaries)), num_to_flip))
            else:
                flip_indices = {0}
            
            # 4. Build confused sequence
            result_fragments = []
            for i, (start, end) in enumerate(fragment_boundaries):
                fragment = original_seq[start:end]
                
                if i in flip_indices:
                    fragment = reverse_complement(fragment)
                
                result_fragments.append(fragment)

            # 5. Randomly shuffle fragment order
            if random.random() < 0.5 and len(result_fragments) > 1:
                random.shuffle(result_fragments)
            
            confused_seq = ''.join(result_fragments)
            
            # 6. Validate generated sequence
            if len(confused_seq) != seq_len:
                continue
            
            # Calculate sequence difference
            diff_stats = calculate_sequence_differences(original_seq, confused_seq)
            base_diff_ratio = diff_stats['base_diff_ratio']
            
            # DNA confusion variant does not check free energy, return as long as generation is successful
            if confused_seq != original_seq:  # Ensure not original sequence
                print(f"✅ DNA confusion variant generated successfully: Sequence difference={base_diff_ratio:.3f}")
                return confused_seq
                
        except Exception as e:
            if attempt == 0:
                print(f"❌ DNA confusion generation failed: {str(e)}")
            continue
    
    print(f"❌ Cannot generate DNA confusion variant for sequence (attempted {max_attempts} times)")
    return ""


# --------------------------
# Core: Negative sample variant generation (random base substitution) - free energy not considered
# --------------------------
def generate_negative_variant(
    original_seq: str,
    mutation_rate: float = 0.5,
    max_attempts: int = 20
) -> str:
    """
    Generate negative sample variant through random base substitution
    Note: Free energy changes not considered
    """
    seq_len = len(original_seq)
    bases = ['A', 'T', 'C', 'G']
    
    for attempt in range(max_attempts):
        try:
            # Calculate number of bases to substitute
            num_mutations = max(1, int(seq_len * mutation_rate))
            
            # Randomly select substitution positions
            mutation_positions = random.sample(range(seq_len), num_mutations)
            
            # Build negative sample sequence
            negative_seq_list = list(original_seq)
            
            for pos in mutation_positions:
                original_base = negative_seq_list[pos]
                possible_bases = [b for b in bases if b != original_base]
                if possible_bases:
                    new_base = random.choice(possible_bases)
                    negative_seq_list[pos] = new_base
            
            negative_seq = ''.join(negative_seq_list)
            
            # Validate generated sequence
            if len(negative_seq) != seq_len:
                continue
                
            # Calculate difference from original sequence
            diff_count = sum(1 for a, b in zip(original_seq, negative_seq) if a != b)
            diff_ratio = diff_count / seq_len
            
            # Negative sample only checks sequence difference, does not consider free energy
            if 0.3 <= diff_ratio <= 0.7 and negative_seq != original_seq:
                print(f"✅ Negative sample generated successfully: Difference rate={diff_ratio:.3f}")
                return negative_seq
            else:
                if attempt == max_attempts - 1:
                    print(f"⚠️ Negative sample difference rate out of range: {diff_ratio:.3f}")
                
        except Exception as e:
            if attempt == 0:
                print(f"❌ Negative sample generation failed: {str(e)}")
            continue
    
    print(f"❌ Cannot generate negative sample variant for sequence (attempted {max_attempts} times)")
    return ""


# --------------------------
# Top-level: Triplet views generation (original sequence + two positive samples + one negative sample)
# --------------------------
def generate_triplet_views(
    current_seg: dict,
    # Semantic variant parameters
    conservative_energy_tolerance: float = 0.5,
    min_pos_base_diff: float = 0.05,
    max_pos_base_diff: float = 0.2,
    pos_conservative_ratio: float = 0.3,
    pos_syn_ratio: float = 0.7,
    max_replace_ratio: float = 0.3,
    max_semantic_attempts: int = 200,
    # DNA confusion parameters
    min_fragment_len: int = 50,
    max_fragment_len: int = 200,
    min_fragments: int = 3,
    max_fragments: int = 10,
    flip_ratio: float = 0.3,
    max_confusion_attempts: int = 50,
    # Negative sample parameters
    negative_mutation_rate: float = 0.5,
    max_negative_attempts: int = 20
) -> Dict[str, Any]:
    """
    Generate triplet views for a sequence: original sequence + two positive samples + one negative sample
    Semantic variant only calculates free energy change from conservative substitutions, other variants do not consider free energy
    """
    current_seq = current_seg['original_seq']
    current_label = current_seg['label']
    seq_len = len(current_seq)
    
    positive_views = []
    negative_views = []
    view_types = []
    
    print(f"🔧 Generating triplet views for sequence (length: {seq_len} bp)")

    # Generate semantic similarity variant (first positive sample) - only calculates free energy change from conservative substitutions
    semantic_variant = generate_positive_variant(
        seq_str=current_seq,
        conservative_energy_tolerance=conservative_energy_tolerance,
        min_pos_base_diff=min_pos_base_diff,
        max_pos_base_diff=max_pos_base_diff,
        pos_conservative_ratio=pos_conservative_ratio,
        pos_syn_ratio=pos_syn_ratio,
        max_replace_ratio=max_replace_ratio,
        max_attempts=max_semantic_attempts
    )
    
    if semantic_variant:
        positive_views.append(semantic_variant)
        view_types.append("semantic")
        print(f"✅ Semantic variant generated successfully")
    else:
        print(f"❌ Semantic variant generation failed")
    
    # Generate DNA confusion variant (second positive sample) - free energy not considered
    min_required_length = min_fragment_len * min_fragments
    if seq_len >= min_required_length:
        confusion_variant = generate_dna_confusion_variant(
            original_seq=current_seq,
            min_fragment_len=min_fragment_len,
            max_fragment_len=max_fragment_len,
            min_fragments=min_fragments,
            max_fragments=max_fragments,
            flip_ratio=flip_ratio,
            max_attempts=max_confusion_attempts
        )
        
        if confusion_variant:
            positive_views.append(confusion_variant)
            view_types.append("confusion")
            print(f"✅ DNA confusion variant generated successfully")
        else:
            print(f"❌ DNA confusion variant generation failed")
    else:
        print(f"⚠️ Sequence too short ({seq_len} bp < {min_required_length} bp), skipping DNA confusion variant generation")
    
    # Generate negative sample variant (random base substitution) - free energy not considered
    negative_variant = generate_negative_variant(
        original_seq=current_seq,
        mutation_rate=negative_mutation_rate,
        max_attempts=max_negative_attempts
    )
    
    if negative_variant:
        negative_views.append(negative_variant)
        print(f"✅ Negative sample variant generated successfully")
    else:
        print(f"❌ Negative sample variant generation failed")
    
    # Build result - ensure using correct key names
    result = {
        'original_seq': current_seq,
        'original_label': current_label,
        'positive_views': positive_views,  # Semantic and confusion variants
        'contrastive_negative_views': negative_views,  # Random mutation variants
        'view_types': view_types,
        'segment_id': current_seg.get('segment_id', 'unknown'),
        'num_positive_views': len(positive_views),
        'num_contrastive_negative_views': len(negative_views),  # Using new key name
        'has_variants': current_label == 1 and len(positive_views) > 0
    }
    
    # Add statistics information
    if positive_views or negative_views:
        view_diff_details = []
        
        # Positive sample differences
        for i, view in enumerate(positive_views):
            diff_stats = calculate_sequence_differences(current_seq, view)
            view_detail = {
                'type': view_types[i] if i < len(view_types) else 'unknown',
                'base_diff_ratio': diff_stats['base_diff_ratio'],
                'aa_diff_ratio': diff_stats['aa_diff_ratio'],
                'base_diff_count': diff_stats['base_diff_count']
            }
            view_diff_details.append(view_detail)
        
        # Negative sample differences
        for view in negative_views:
            diff_stats = calculate_sequence_differences(current_seq, view)
            view_detail = {
                'type': 'negative',
                'base_diff_ratio': diff_stats['base_diff_ratio'],
                'aa_diff_ratio': diff_stats['aa_diff_ratio'],
                'base_diff_count': diff_stats['base_diff_count']
            }
            view_diff_details.append(view_detail)
        
        result['view_diff_details'] = view_diff_details
    
    print(f"📊 Triplet generation result: {len(positive_views)} positive samples, {len(negative_views)} negative samples")
    
    return result


# --------------------------
# Parallel generation function
# --------------------------
def _triplet_views_wrapper(args: Tuple) -> Dict[str, Any]:
    """Parallel wrapper for triplet view generation"""
    (current_seg, conservative_energy_tolerance, min_pos_base_diff,
     max_pos_base_diff, pos_conservative_ratio, pos_syn_ratio, max_replace_ratio,
     max_semantic_attempts, min_fragment_len, max_fragment_len,
     min_fragments, max_fragments, flip_ratio, max_confusion_attempts,
     negative_mutation_rate, max_negative_attempts) = args
     
    try:
        return generate_triplet_views(
            current_seg=current_seg,
            conservative_energy_tolerance=conservative_energy_tolerance,
            min_pos_base_diff=min_pos_base_diff,
            max_pos_base_diff=max_pos_base_diff,
            pos_conservative_ratio=pos_conservative_ratio,
            pos_syn_ratio=pos_syn_ratio,
            max_replace_ratio=max_replace_ratio,
            max_semantic_attempts=max_semantic_attempts,
            min_fragment_len=min_fragment_len,
            max_fragment_len=max_fragment_len,
            min_fragments=min_fragments,
            max_fragments=max_fragments,
            flip_ratio=flip_ratio,
            max_confusion_attempts=max_confusion_attempts,
            negative_mutation_rate=negative_mutation_rate,
            max_negative_attempts=max_negative_attempts
        )
    except Exception as e:
        warnings.warn(f"Failed to process sequence {current_seg.get('segment_id', 'unknown')}: {str(e)}")
        return {
            'original_seq': current_seg['original_seq'],
            'original_label': current_seg['label'],
            'positive_views': [],
            'contrastive_negative_views': [],  # Using new key name
            'view_types': [],
            'segment_id': current_seg.get('segment_id', 'unknown'),
            'num_positive_views': 0,
            'num_contrastive_negative_views': 0,  # Using new key name
            'has_variants': False,
            'error': str(e)
        }

def generate_triplet_views_parallel(
    all_segments: List[dict],
    # Semantic variant parameters
    conservative_energy_tolerance: float = 0.5,
    min_pos_base_diff: float = 0.05,
    max_pos_base_diff: float = 0.2,
    pos_conservative_ratio: float = 0.3,
    pos_syn_ratio: float = 0.7,
    max_replace_ratio: float = 0.3,
    max_semantic_attempts: int = 200,
    # DNA confusion parameters
    min_fragment_len: int = 50,
    max_fragment_len: int = 200,
    min_fragments: int = 3,
    max_fragments: int = 10,
    flip_ratio: float = 0.3,
    max_confusion_attempts: int = 50,
    # Negative sample parameters
    negative_mutation_rate: float = 0.5,
    max_negative_attempts: int = 20,
    # Parallel parameters
    num_workers: int = None
) -> List[Dict[str, Any]]:
    """
    Generate triplet views for all sequences in parallel
    Only generate variants for positive samples, keep negative samples unchanged
    """
    args_list = [
        (seg, conservative_energy_tolerance, min_pos_base_diff,
         max_pos_base_diff, pos_conservative_ratio, pos_syn_ratio, max_replace_ratio,
         max_semantic_attempts, min_fragment_len, max_fragment_len,
         min_fragments, max_fragments, flip_ratio, max_confusion_attempts,
         negative_mutation_rate, max_negative_attempts)
        for seg in all_segments
    ]

    num_workers = num_workers or max(1, int(multiprocessing.cpu_count() * 0.8))
    print(f"🚀 Starting triplet view generation (processes: {num_workers}, total sequences: {len(all_segments)})")
    print(f"📋 Target: Generate 2 positive sample variants + 1 random mutation variant (contrastive learning negative sample) for each positive sample sequence")
    print(f"📋 Negative sample sequences do not generate any variants")
    print(f"🔋 Conservative substitution free energy tolerance: ±{conservative_energy_tolerance} kcal/mol")
    print(f"🔍 Semantic variant difference range: [{min_pos_base_diff:.2f}, {max_pos_base_diff:.2f}]")
    print(f"🧬 DNA confusion: fragments {min_fragment_len}-{max_fragment_len} bp, {min_fragments}-{max_fragments} fragments")
    print(f"❌ Random mutation variant: random mutation {negative_mutation_rate:.1%} bases (only used for contrastive learning negative samples)")

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(_triplet_views_wrapper, args_list)

    # Count generation results - using new key names
    successful_positives = sum(r['num_positive_views'] for r in results)
    successful_contrastive_negatives = sum(r['num_contrastive_negative_views'] for r in results)  # Modified
    
    semantic_success = sum(1 for r in results if any(t == "semantic" for t in r.get('view_types', [])))
    confusion_success = sum(1 for r in results if any(t == "confusion" for t in r.get('view_types', [])))
    
    # Using new key names
    contrastive_negative_success = sum(1 for r in results if r['num_contrastive_negative_views'] > 0)  # Modified
    
    # Full triplet: positive sample with semantic + confusion + random mutation
    full_triplet_success = sum(1 for r in results if r['num_positive_views'] >= 2 and r['num_contrastive_negative_views'] >= 1)  # Modified
    
    # Count number of positive and negative samples
    positive_segments_count = sum(1 for seg in all_segments if seg.get('label', seg.get('original_label', 0)) == 1)
    negative_segments_count = len(all_segments) - positive_segments_count
    
    # Guard the percentage denominators: symmetric augmentation calls this
    # function with negatives-only batches, where positive_segments_count = 0.
    denom = max(positive_segments_count, 1)
    print(f"\n📊 Triplet generation completion statistics:")
    print(f"   Total processed sequences: {len(all_segments)}")
    print(f"   Positive sample sequences: {positive_segments_count}")
    print(f"   Negative sample sequences: {negative_segments_count}")
    print(f"   Total positive sample variants: {successful_positives}")
    print(f"   Total contrastive learning negative samples: {successful_contrastive_negatives}")
    print(f"   Semantic variant success: {semantic_success}/{positive_segments_count} ({semantic_success/denom*100:.1f}%)")
    print(f"   DNA confusion success: {confusion_success}/{positive_segments_count} ({confusion_success/denom*100:.1f}%)")
    print(f"   Random mutation variant (contrastive learning negative sample) success: {contrastive_negative_success}/{positive_segments_count} ({contrastive_negative_success/denom*100:.1f}%)")
    print(f"   Full triplet success: {full_triplet_success}/{positive_segments_count} ({full_triplet_success/denom*100:.1f}%)")

    return results

# --------------------------
# Test function
# --------------------------
def test_variant_generation():
    """Test variant generation functionality"""
    # Test sequence
    test_seq = "ATGGCCATTGAATGGGCCGCTGCTTCTGGTGCTGCCGGTAGCGCAGTCCGTGGCGGTGCTGGTGCTGGTGCTGGCCAGCGTGGTGCTGCCG"
    
    print("🧪 Testing variant generation functionality (only calculating conservative substitution free energy)...")
    
    # Test semantic similarity variant
    print("\n1. Testing semantic similarity variant generation:")
    semantic_variant = generate_positive_variant(test_seq, max_attempts=10)
    if semantic_variant:
        diff_stats = calculate_sequence_differences(test_seq, semantic_variant)
        print(f"   ✅ Generation successful! Sequence difference: {diff_stats['base_diff_ratio']:.3f}")
        print(f"   Original sequence: {test_seq[:30]}...")
        print(f"   Semantic variant: {semantic_variant[:30]}...")
    else:
        print("   ❌ Generation failed")
    
    # Test DNA confusion variant
    print("\n2. Testing DNA confusion variant generation:")
    confusion_variant = generate_dna_confusion_variant(test_seq, max_attempts=10)
    if confusion_variant:
        diff_stats = calculate_sequence_differences(test_seq, confusion_variant)
        print(f"   ✅ Generation successful! Sequence difference: {diff_stats['base_diff_ratio']:.3f}")
        print(f"   Original sequence: {test_seq[:30]}...")
        print(f"   DNA confusion: {confusion_variant[:30]}...")
    else:
        print("   ❌ Generation failed")
    
    # Test negative sample variant
    print("\n3. Testing negative sample variant generation:")
    negative_variant = generate_negative_variant(test_seq, max_attempts=10)
    if negative_variant:
        diff_stats = calculate_sequence_differences(test_seq, negative_variant)
        print(f"   ✅ Generation successful! Sequence difference: {diff_stats['base_diff_ratio']:.3f}")
        print(f"   Original sequence: {test_seq[:30]}...")
        print(f"   Negative sample: {negative_variant[:30]}...")
    else:
        print("   ❌ Generation failed")
    
    # Test triplet view generation
    print("\n4. Testing triplet view generation:")
    test_segment = {
        'original_seq': test_seq,
        'label': 1,
        'segment_id': 'test_seq'
    }
    triplet_result = generate_triplet_views(test_segment, 
                                          max_semantic_attempts=10, 
                                          max_confusion_attempts=10,
                                          max_negative_attempts=10)
    print(f"   Generation result: {triplet_result['num_positive_views']} positive samples, {triplet_result['num_negative_views']} negative samples")
    print(f"   View types: {triplet_result.get('view_types', [])}")


if __name__ == "__main__":
    test_variant_generation()
