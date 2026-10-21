"""Partition cached records while keeping shared complete content together.

Each original and its attached views form an indivisible record. Comparisons
use normalized complete content; partial similarity is outside this contract.
Declared original_id values are reported as provenance metadata because their
meaning depends on the evaluation protocol.
"""

import hashlib
import json
import numbers
from pathlib import Path
import random


class DataLeakageError(ValueError):
    """Input validation, class coverage, or training/test separation failed."""


_NAMES = ("train", "validation", "test")
_RATIOS = (0.8, 0.1, 0.1)


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _json_digest(value):
    return _digest(json.dumps(value, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=True).encode("utf-8"))


def _content_digest(value):
    if not isinstance(value, str):
        raise DataLeakageError("Cached originals and existing views must be strings.")
    canonical = "".join(value.split()).upper()
    if not canonical:
        raise DataLeakageError("An original or existing view has empty normalized content.")
    return _digest(canonical.encode("utf-8"))


def _optional_id(value):
    if value is None or isinstance(value, str) and not value.strip():
        return None
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise DataLeakageError("Provided record and parent IDs must be strings or integers.")
    # Hash the explicit value and its type, without guessing or stripping suffixes.
    return _json_digest(value)


def _prepare_records(records):
    if not isinstance(records, (list, tuple)) or not records:
        raise DataLeakageError("A nonempty list of cached records is required.")
    prepared, original_targets = [], {}
    for record in records:
        if not isinstance(record, dict):
            raise DataLeakageError("Each cached record must be a dictionary.")
        targets = [record[key] for key in ("original_label", "label") if key in record]
        if (not targets or any(isinstance(x, bool) or not isinstance(x, numbers.Real)
                               or x not in (0, 1) for x in targets)
                or any(x != targets[0] for x in targets)):
            raise DataLeakageError("Each original requires consistent, explicit binary labels.")
        label = int(targets[0])
        original = _content_digest(record.get("original_seq"))
        if original_targets.setdefault(original, label) != label:
            raise DataLeakageError("Identical original content has conflicting labels.")
        payload = {"original": original, "label": label,
                   "segment_id": _optional_id(record.get("segment_id")),
                   "original_id": _optional_id(record.get("original_id"))}
        contents = {original}
        for key in ("positive_views", "contrastive_negative_views"):
            views = record.get(key, [])
            if not isinstance(views, (list, tuple)):
                raise DataLeakageError("Existing view collections must be lists or tuples.")
            payload[key] = [_content_digest(view) for view in views]
            contents.update(payload[key])
        # View roles never supply classification targets. Keep explicit role
        # metadata in the fingerprint without imposing model-specific categories.
        view_types = record.get("view_types", [])
        if not isinstance(view_types, (list, tuple)) or any(not isinstance(x, str) for x in view_types):
            raise DataLeakageError("Provided view_types must be a list of strings.")
        payload["view_types"] = list(view_types)
        prepared.append({"record": record, "label": label, "contents": contents,
                         "parent": payload["original_id"], "id": _json_digest(payload)})
    return sorted(prepared, key=lambda item: item["id"])


def _read_external_files(files):
    if not isinstance(files, dict) or not files:
        raise DataLeakageError("A nonempty external_test_files mapping is required.")
    if any(not isinstance(name, str) or not name.strip() for name in files):
        raise DataLeakageError("External dataset names must be nonempty strings.")
    result = {}
    for name in sorted(files):
        try:
            path = Path(files[name])
            if not path.is_file():
                raise OSError("not a regular file")
            file_hash = hashlib.sha256()
            contents, chunks, started, count = set(), [], False, 0
            with path.open("rb") as handle:
                for line in handle:
                    file_hash.update(line)
                    if line.startswith(b">"):
                        if started:
                            contents.add(_content_digest(b"".join(chunks).decode("utf-8")))
                            count += 1
                        started, chunks = True, []
                    elif line.strip():
                        if not started:
                            raise DataLeakageError("External file is not valid FASTA.")
                        chunks.append(line)
            if not started:
                raise DataLeakageError("External FASTA has no records.")
            contents.add(_content_digest(b"".join(chunks).decode("utf-8")))
            count += 1
        except (OSError, TypeError, UnicodeError) as exc:
            raise DataLeakageError("Cannot read a configured external FASTA file.") from exc
        result[name] = {"contents": contents, "sha256": file_hash.hexdigest(),
                        "records": count, "filename": path.name}
    return result


def _group_records(prepared):
    parent = list(range(len(prepared)))
    size = [1] * len(prepared)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owners = {}
    for index, item in enumerate(prepared):
        keys = [("content", value) for value in sorted(item["contents"])]
        for key in keys:
            if key in owners:
                left, right = find(index), find(owners[key])
                if left != right:
                    if size[left] < size[right]:
                        left, right = right, left
                    parent[right] = left
                    size[left] += size[right]
            else:
                owners[key] = index
    components = {}
    for index in range(len(prepared)):
        components.setdefault(find(index), []).append(index)
    groups = []
    for members in components.values():
        positives = sum(prepared[index]["label"] for index in members)
        groups.append({"members": members, "counts": [len(members), len(members) - positives, positives],
                       "id": _json_digest([prepared[index]["id"] for index in members])})
    return sorted(groups, key=lambda group: group["id"])


def _integer_quotas(total, minima, maxima, targets=None):
    """Nearest squared-error integer counts under explicit feasible bounds."""
    counts = list(minima)
    targets = [total * ratio for ratio in _RATIOS] if targets is None else targets
    if sum(minima) > total or sum(maxima) < total:
        raise DataLeakageError("Insufficient records for three subsets containing both classes.")
    while sum(counts) < total:
        eligible = [i for i in range(3) if counts[i] < maxima[i]]
        chosen = min(eligible, key=lambda i: (2 * (counts[i] - targets[i]) + 1, i))
        counts[chosen] += 1
    return counts


def _target_record_counts(total):
    """Round target proportions while retaining the exact total record count."""
    raw = [total * ratio for ratio in _RATIOS]
    counts = [int(value) for value in raw]
    order = sorted(range(3), key=lambda i: (-(raw[i] - counts[i]), i))
    for index in order[:total - sum(counts)]:
        counts[index] += 1
    return counts


def _allocate(groups, prepared, seed):
    totals = [len(prepared), sum(item["label"] == 0 for item in prepared),
              sum(item["label"] == 1 for item in prepared)]
    sizes = _target_record_counts(totals[0])
    if any(size < 2 for size in sizes) or min(totals[1:]) < 3:
        raise DataLeakageError("Target integer 8:1:1 counts cannot provide both classes in every subset.")
    pools = {"mixed": [], "zero": [], "one": []}
    for group in groups:
        kind = "mixed" if group["counts"][1] and group["counts"][2] else (
            "zero" if group["counts"][1] else "one")
        pools[kind].append(group)
    # This is a necessary and sufficient condition for three class-complete
    # subsets of indivisible binary-labelled groups. The requested integer
    # record counts are checked separately and are never relaxed.
    if len(pools["mixed"]) + min(len(pools["zero"]), len(pools["one"])) < 3:
        raise DataLeakageError("Cannot form three indivisible subsets each containing both classes.")
    if all(group["counts"][0] == 1 for group in groups):
        zeros = _integer_quotas(totals[1], [1, 1, 1], [size - 1 for size in sizes])
        quotas = [zeros, [sizes[i] - zeros[i] for i in range(3)]]
        rng, assigned = random.Random(seed), [[], [], []]
        for label, kind in enumerate(("zero", "one")):
            candidates = list(pools[kind])
            rng.shuffle(candidates)
            offset = 0
            for subset in range(3):
                end = offset + quotas[label][subset]
                assigned[subset].extend(group["members"][0] for group in candidates[offset:end])
                offset = end
        return [sorted(indices) for indices in assigned]

    rng = random.Random(seed)
    multiple = [(group, rng.random()) for group in groups if group["counts"][0] > 1]
    multiple.sort(key=lambda pair: (-pair[0]["counts"][0], pair[1], pair[0]["id"]))
    multiple = [group for group, _ in multiple]
    singles = [[group["members"][0] for group in pools[kind] if group["counts"][0] == 1]
               for kind in ("zero", "one")]
    capacities, counts = list(sizes), [[0, 0, 0] for _ in range(3)]
    targets = [[ratio * total for total in totals] for ratio in _RATIOS]
    remaining_labels = [[len(singles[0]), len(singles[1])]]
    for group in reversed(multiple):
        remaining_labels.append([remaining_labels[-1][k] + group["counts"][k + 1] for k in (0, 1)])
    remaining_labels.reverse()

    def state_key(depth):
        # For feasibility, only remaining capacities and whether each label has
        # appeared matter; exact class counts only rank feasible destinations.
        masks = tuple(int(row[1] > 0) + 2 * int(row[2] > 0) for row in counts)
        return depth, tuple(capacities), masks

    def adjust(group, subset, direction):
        capacities[subset] -= direction * group["counts"][0]
        for kind in range(3):
            counts[subset][kind] += direction * group["counts"][kind]

    def candidate_subsets(depth):
        group, options = multiple[depth], []
        for subset in range(3):
            if group["counts"][0] > capacities[subset]:
                continue
            delta = sum(((counts[subset][k] + group["counts"][k] - targets[subset][k]) ** 2
                         - (counts[subset][k] - targets[subset][k]) ** 2)
                        / max(targets[subset][k], 1.0) for k in range(3))
            adjust(group, subset, 1)
            # Each missing class requires at least one remaining record slot.
            fits = all(capacities[s] >= sum(counts[s][k] == 0 for k in (1, 2)) for s in range(3))
            fits = fits and all(remaining_labels[depth + 1][k - 1] >= sum(row[k] == 0 for row in counts)
                                for k in (1, 2))
            adjust(group, subset, -1)
            if fits:
                options.append((delta, rng.random(), subset))
        return [entry[2] for entry in sorted(options)]

    def finish_singletons():
        minimum_zero = [int(row[1] == 0) for row in counts]
        maximum_zero = [capacities[s] - int(counts[s][2] == 0) for s in range(3)]
        if (any(minimum_zero[s] > maximum_zero[s] for s in range(3))
                or sum(minimum_zero) > len(singles[0]) or sum(maximum_zero) < len(singles[0])):
            return None
        desired_zero = [targets[s][1] - counts[s][1] for s in range(3)]
        zeros = _integer_quotas(len(singles[0]), minimum_zero, maximum_zero, desired_zero)
        return [zeros, [capacities[s] - zeros[s] for s in range(3)]]

    # Place large content groups before singletons, enforcing integer capacities
    # throughout. Iterative backtracking can undo an unlucky early choice without
    # recursion-depth limits. A bounded search may still fail on difficult group
    # packings; such a failure never causes groups or requested ratios to change.
    failed_states, placements = set(), []
    stack = [(state_key(0), candidate_subsets(0))]
    visited, visit_limit = 0, max(50000, 4 * len(multiple))
    while stack and visited < visit_limit:
        depth = len(stack) - 1
        key, options = stack[-1]
        if not options:
            failed_states.add(key)
            stack.pop()
            if placements:
                adjust(multiple[len(stack) - 1], placements.pop(), -1)
            continue
        subset = options.pop(0)
        group = multiple[depth]
        adjust(group, subset, 1)
        placements.append(subset)
        visited += 1
        next_depth = depth + 1
        if next_depth == len(multiple):
            quotas = finish_singletons()
            if quotas is not None:
                assigned = [[], [], []]
                for placed_group, destination in zip(multiple, placements):
                    assigned[destination].extend(placed_group["members"])
                for label, candidates in enumerate(singles):
                    rng.shuffle(candidates)
                    offset = 0
                    for destination in range(3):
                        end = offset + quotas[label][destination]
                        assigned[destination].extend(candidates[offset:end])
                        offset = end
                return [sorted(indices) for indices in assigned]
        else:
            next_key = state_key(next_depth)
            if next_key not in failed_states:
                stack.append((next_key, candidate_subsets(next_depth)))
                continue
        adjust(group, placements.pop(), -1)
    raise DataLeakageError(
        "Current content-group allocation did not meet target integer 8:1:1 counts "
        + json.dumps(sizes) + "; groups were not split. The bounded search does not prove that no feasible allocation exists.")


def prepare_checked_split(records, external_test_files, seed=42):
    """Return train/validation/test records and aggregate, non-content metadata.

    Partition membership depends only on existing records and ``seed``. External
    complete-content matches in train or validation cause an error; matches only
    in the reserved internal test are allowed and reported as correlated tests.
    No records are removed or reassigned in response to external test contents.
    """
    if type(seed) is not int:
        raise DataLeakageError("The split seed must be an integer.")
    prepared = _prepare_records(records)
    external = _read_external_files(external_test_files)
    groups = _group_records(prepared)
    partitions = _allocate(groups, prepared, seed)
    overlaps = {
        name: {subset: sum(bool(prepared[index]["contents"] & info["contents"]) for index in indices)
               for subset, indices in zip(_NAMES, partitions)}
        for name, info in external.items()
    }
    failures = [{"file": external[name]["filename"], "train_records": counts["train"],
                 "validation_records": counts["validation"]}
                for name, counts in overlaps.items() if counts["train"] or counts["validation"]]
    if failures:
        raise DataLeakageError("Complete external-test content occurs in train/validation: "
                               + json.dumps(failures, ensure_ascii=True))
    counts = {
        subset: {"records": len(indices), "label_0": sum(prepared[i]["label"] == 0 for i in indices),
                 "label_1": sum(prepared[i]["label"] == 1 for i in indices)}
        for subset, indices in zip(_NAMES, partitions)
    }
    memberships = {subset: [prepared[index]["id"] for index in indices]
                   for subset, indices in zip(_NAMES, partitions)}
    parents = [{prepared[index]["parent"] for index in indices if prepared[index]["parent"] is not None}
               for indices in partitions]
    parent_overlap = {f"{_NAMES[left]}_{_NAMES[right]}": len(parents[left] & parents[right])
                      for left, right in ((0, 1), (0, 2), (1, 2))}
    summary = {
        "dataset_fingerprint": _json_digest([item["id"] for item in prepared]),
        "split_seed": seed, "target_ratios": dict(zip(_NAMES, _RATIOS)),
        "target_record_counts": dict(zip(_NAMES, _target_record_counts(len(prepared)))),
        "actual_ratios": {name: len(indices) / len(prepared) for name, indices in zip(_NAMES, partitions)},
        "counts": counts, "record_fingerprints": memberships,
        "external_test_sha256": {name: info["sha256"] for name, info in external.items()},
        "external_test_record_counts": {name: info["records"] for name, info in external.items()},
        "external_overlap_counts": overlaps, "group_count": len(groups),
        "source_metadata_complete": all(item["parent"] is not None for item in prepared),
        "records_missing_original_id": sum(item["parent"] is None for item in prepared),
        "declared_parent_overlap_counts": parent_overlap,
        "source_grouping_note": "Only complete content is grouped. Shared declared original_id values are reported, not classified as leakage; homology is not verified.",
        "test_overlap_note": "Internal/external test-only matches are allowed; affected test results are correlated.",
    }
    summary["split_fingerprint"] = _json_digest({
        "dataset_fingerprint": summary["dataset_fingerprint"], "seed": seed, "memberships": memberships})
    result = [[prepared[index]["record"] for index in indices] for indices in partitions]
    return result[0], result[1], result[2], summary


def verify_prepared_split(records, external_test_files):
    """Validate fixed-membership splits and external isolation.

    ``records`` must each carry ``split`` = "train" or "val" (variants already
    attached). Membership is fixed by the delivered dataset, so this function
    only verifies: binary labels, both classes on each side, and that no
    complete fitting-side content (original or any attached view) occurs in
    an external test file. With fixed membership there is no reserved subset
    to absorb correlated matches, so any overlap raises immediately.

    Returns a manifest summary for the training log and data_split.json.
    """
    prepared = _prepare_records(records)
    external = _read_external_files(external_test_files)

    by_split = {"train": [], "val": []}
    for item in prepared:
        split = item["record"].get("split")
        if split not in by_split:
            raise DataLeakageError(
                "Prepared-split records require split='train' or 'val'; got %r" % (split,))
        by_split[split].append(item)

    counts = {}
    for split, items in by_split.items():
        label_0 = sum(item["label"] == 0 for item in items)
        label_1 = sum(item["label"] == 1 for item in items)
        if label_0 == 0 or label_1 == 0:
            raise DataLeakageError(
                f"Prepared {split} split lacks one of the two classes "
                f"(label_0={label_0}, label_1={label_1}).")
        counts[split] = {"records": len(items), "label_0": label_0, "label_1": label_1}

    fitting_contents = set()
    for item in prepared:
        fitting_contents.update(item["contents"])

    overlaps = {}
    failures = []
    for name, info in sorted(external.items()):
        shared = fitting_contents & info["contents"]
        overlaps[name] = {
            "filename": info["filename"],
            "records": info["records"],
            "sha256": info["sha256"],
            "complete_content_matches": len(shared),
        }
        if shared:
            failures.append({"file": info["filename"], "matching_contents": len(shared)})
    if failures:
        raise DataLeakageError(
            "Complete external-test content occurs in the prepared fitting side "
            "(train+val): " + json.dumps(failures, ensure_ascii=True))

    memberships = {split: sorted(item["id"] for item in items)
                   for split, items in by_split.items()}
    summary = {
        "mode": "prepared_split",
        "counts": counts,
        "records_with_variants": sum(
            1 for item in prepared if item["record"].get("has_variants")),
        "dataset_fingerprint": _json_digest([item["id"] for item in prepared]),
        "record_fingerprints": memberships,
        "external_test_files": overlaps,
        "external_test_sha256": {name: info["sha256"]
                                 for name, info in sorted(external.items())},
        "isolation": "pass",
        "split_membership_note": (
            "Membership fixed by the delivered dataset; this pipeline never "
            "re-partitions prepared-split records."),
        "grouping_note": (
            "Only complete content is compared against external files; "
            "homology is not verified here (the delivered dataset carries its "
            "own audited species/family holdouts)."),
    }
    summary["split_fingerprint"] = _json_digest({
        "dataset_fingerprint": summary["dataset_fingerprint"],
        "memberships": memberships,
        "external": summary["external_test_sha256"],
    })
    return summary
