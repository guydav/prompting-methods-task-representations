#!/usr/bin/env python3
"""
Script to run transformer evaluation tests with different configurations.
"""

import argparse
import subprocess
import sys

import pytest


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run transformer evaluation tests")

    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Hugging Face model name to test",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="country-capital",
        help="Dataset name to test with",
    )
    parser.add_argument(
        "--n-examples", type=int, default=4, help="Number of examples to test"
    )
    parser.add_argument(
        "--start-index", type=int, default=0, help="Start index in the dataset"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Match a country with its capital city:",
        help="Prompt to use for testing",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for floating point comparisons",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for floating point comparisons",
    )
    parser.add_argument(
        "--tests",
        type=str,
        default="all",
        choices=["all", "nll", "generate", "default"],
        help="Which tests to run",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def main():
    """Run the tests with the specified configuration."""
    args = parse_args()

    # Base command with common arguments
    base_cmd = [
        # "python",
        # "-m",
        # "pytest",
        f"--model-name={args.model_name}",
        f"--dataset-name={args.dataset_name}",
        f"--n-examples={args.n_examples}",
        f"--start-index={args.start_index}",
        f"--prompt={args.prompt}",
        f"--atol={args.atol}",
        f"--rtol={args.rtol}",
    ]

    # Add verbosity if requested
    if args.verbose:
        base_cmd.append("-v")

    # Determine which tests to run
    test_paths = []
    if args.tests == "all":
        test_paths = [
            "tests",
            # "test_sentence_eval.py::SentenceEvalEvaluator",
            # "test_prompt_based_eval.py::TestPromptBasedEval",
            # "test_prompt_based_mean_act.py::TestExtractActivations",
            # "test_activation_replacement.py::TestBatchedActivationReplacement",
            # "test_prompt_based_indirect_effect.py::TestComputePromptBasedIndirectEffect",
            # "test_function_vector_intervention.py::TestBatchedFunctionVectorIntervention",
            # "test_prompt_based_intervention_eval.py::TestPromptBasedInterventionEval"
        ]
    elif args.tests == "prompt_eval":
        test_paths = ["test_prompt_based_eval.py::TestPromptBasedEval"]
    elif args.tests == "nll":
        test_paths = ["test_sentence_eval.py::SentenceEvalEvaluator::test_nll_mode"]
    elif args.tests == "generate":
        test_paths = [
            "test_sentence_eval.py::SentenceEvalEvaluator::test_generate_str_mode"
        ]
    elif args.tests == "default":
        test_paths = ["test_sentence_eval.py::SentenceEvalEvaluator::test_default_mode"]

    # Run the tests
    cmd = base_cmd + test_paths
    print(f"Running command: {' '.join(cmd)}")

    sys.exit(pytest.main(cmd))

    # try:
    #     result = subprocess.run(cmd, check=True)
    #     print(f"Tests completed with exit code {result.returncode}")
    #     return result.returncode
    # except subprocess.CalledProcessError as e:
    #     print(f"Tests failed with exit code {e.returncode}")
    #     return e.returncode


if __name__ == "__main__":
    sys.exit(main())
