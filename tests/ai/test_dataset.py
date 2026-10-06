from ai.dataset import generate_examples


def test_dataset_is_reproducible():
    first = generate_examples(12, seed=7, min_depth=2, max_depth=5)
    second = generate_examples(12, seed=7, min_depth=2, max_depth=5)

    assert first == second
    assert len(first) == 12
    assert all(example.remaining_steps > 0 for example in first)
