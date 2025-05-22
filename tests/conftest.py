import pytest


def pytest_addoption(parser):
    """Add command-line options for transformer tests."""
    parser.addoption(
        "--model-name",
        action="store",
        default="meta-llama/Llama-3.2-1B",
        help="Hugging Face model name to test",
    )
    parser.addoption(
        "--dataset-name",
        action="store",
        default="country-capital",
        help="Dataset name to test with",
    )
    parser.addoption(
        "--n-examples",
        action="store",
        type=int,
        default=20,
        help="Number of examples to test",
    )
    parser.addoption(
        "--start-index",
        action="store",
        type=int,
        default=0,
        help="Start index in the dataset",
    )
    parser.addoption(
        "--prompt",
        action="store",
        default="Rewrite the word in the past tense.",
        help="Prompt to use for testing",
    )
    parser.addoption(
        "--atol",
        action="store",
        type=float,
        default=1e-4,
        help="Absolute tolerance for floating point comparisons",
    )
    parser.addoption(
        "--rtol",
        action="store",
        type=float,
        default=1e-5,
        help="Relative olerance for floating point comparisons",
    )
    parser.addoption(
        "--prompts-file",
        action="store",
        default="country-capital_prompts.json",
        help="JSON file containing prompts to use (from PROMPTS_PATH)",
    )
    parser.addoption(
        "--max-prompts",
        action="store",
        type=int,
        default=8,
        help="Maximum number of prompts to use from the file",
    )
    parser.addoption(
        "--max-tokens",
        action="store",
        type=int,
        default=64,
        help="Maximum token length for prompts",
    )
    parser.addoption(
        "--batch-sizes",
        action="store",
        default="1,2,3,5,7",
        help="Comma-separated list of batch sizes to test with",
    )
    parser.addoption(
        "--last-token-mode",
        action="store",
        default="true",
        choices=["true", "false", "both"],
        help="Test with last_token_only set to True, False, or both"
    )
    parser.addoption(
        "--fake-activation-value",
        action="store",
        default=0,
        type=float,
        help="Fake activation value"
    )
    parser.addoption(
        "--n-trials-per-prompt",
        action="store",
        type=int,
        default=10,
        help="Number of trials per prompt for intervention",
    )
    parser.addoption(
        "--n-eval-prompts",
        action="store",
        type=int,
        default=2,
        help="Number of prompts to use for evaluation",
    )
    parser.addoption(
        "--mean-activation-scale",
        action="store",
        type=float,
        default=1e-1,
        help="Scale factor for random mean activations",
    )
    parser.addoption(
        "--fv-scale",
        action="store",
        type=float,
        default=1e-1,
        help="Scale factor for random function vectors",
    )
    parser.addoption(
        "--test-generation",
        action="store_true",
        default=False,
        help="Whether to test text generation capabilities",
    )
    parser.addoption(
        "--test-nll",
        action="store_true",
        default=True,
        help="Whether to test NLL computation",
    )


@pytest.fixture
def model_params(request):
    """Get model parameters from command line options."""
    batch_sizes_str = request.config.getoption("--batch-sizes", default="1,2,3,5,7")
    batch_sizes = [int(size.strip()) for size in batch_sizes_str.split(",")]

    return {
        "model_name": request.config.getoption(
            "--model-name", default="meta-llama/Llama-3.2-1B"
        ),
        "dataset_name": request.config.getoption(
            "--dataset-name", default="present-past"
        ),
        "n_examples": request.config.getoption("--n-examples", default=4),
        "start_index": request.config.getoption("--start-index", default=0),
        "prompt": request.config.getoption("--prompt"),
        "atol": request.config.getoption("--atol", default=1e-4),
        "rtol": request.config.getoption("--rtol", default=1e-5),
        "prompts_file": request.config.getoption("--prompts-file", default=None),
        "max_prompts": request.config.getoption("--max-prompts", default=8),
        "max_tokens": request.config.getoption("--max-tokens", default=64),
        "batch_sizes": batch_sizes,
        "last_token_mode": request.config.getoption("--last-token-mode", default="both"),
        "n_trials_per_prompt": request.config.getoption("--n-trials-per-prompt", default=10),
        "n_eval_prompts": request.config.getoption("--n-eval-prompts", default=2),
        "mean_activation_scale": request.config.getoption("--mean-activation-scale", default=1e-1),
        "fv_scale": request.config.getoption("--fv-scale", default=1e-1),
        "test_generation": request.config.getoption("--test-generation", default=False),
        "test_nll": request.config.getoption("--test-nll", default=True),
    }
