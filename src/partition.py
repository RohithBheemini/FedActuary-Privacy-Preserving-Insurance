"""
Phase 2: non-IID partitioning (PRD Section 7).

Provides Dirichlet partitioning over arbitrary labels/features:
- Label shift: Dirichlet partitioning over ClaimNb categories {0, 1, 2, 3, 4}
- Feature shift: Dirichlet partitioning over Region categories (22 regions)
"""
import numpy as np


def dirichlet_partition_by_label(labels, num_clients, alpha, seed=42, min_size=10):
    """Splits row indices across num_clients insurers using a Dirichlet
    draw PER DISTINCT LABEL VALUE.
    Low alpha = extreme skew (label rows concentrate in a few insurers);
    high alpha = close to uniform. Returns a list of num_clients index arrays."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    n_samples = len(labels)
    unique_labels = np.unique(labels)

    # Attempt partitioning, ensuring every client has at least min_size samples
    for attempt in range(10):
        client_indices = [[] for _ in range(num_clients)]
        for label in unique_labels:
            label_idx = np.where(labels == label)[0]
            rng.shuffle(label_idx)
            proportions = rng.dirichlet(alpha=[alpha] * num_clients)
            split_points = (np.cumsum(proportions)[:-1] * len(label_idx)).astype(int)
            chunks = np.split(label_idx, split_points)
            for cid, chunk in enumerate(chunks):
                client_indices[cid].extend(chunk.tolist())

        sizes = [len(idx) for idx in client_indices]
        if min(sizes) >= min_size:
            break
        # Re-seed on rare extreme starvation
        rng = np.random.default_rng(seed + attempt + 1)

    return [np.array(sorted(idx)) for idx in client_indices]


def partition_summary(client_indices, labels):
    """Diagnostic: row count and label distribution per client."""
    labels = np.asarray(labels)
    rows = []
    for i, idx in enumerate(client_indices):
        vals, counts = np.unique(labels[idx], return_counts=True)
        rows.append({
            "client": i,
            "n": int(len(idx)),
            "label_counts": {str(k): int(v) for k, v in zip(vals, counts)}
        })
    return rows


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    fake_claimnb = rng.choice([0, 1, 2, 3, 4], size=5000, p=[0.90, 0.06, 0.02, 0.01, 0.01])

    for alpha in [0.1, 0.5, 1.0, 5.0]:
        parts = dirichlet_partition_by_label(fake_claimnb, num_clients=10, alpha=alpha, seed=1)
        sizes = [len(p) for p in parts]
        print(f"alpha={alpha:<4} client sizes: min={min(sizes)} max={max(sizes)}")
        assert sum(sizes) == len(fake_claimnb), "partition must cover every row exactly once"
    print("partition self-check passed")
