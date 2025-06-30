# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
from copy import copy

import torch
from scipy.stats import norm
from transformers import AutoModelForCausalLM, AutoTokenizer

from fast_detect_gpt import get_sampling_discrepancy_analytic


SAMPLE_DEVICE = "cuda:0"
SCORE_DEVICE = "cuda:1" if torch.cuda.device_count() >= 2 else SAMPLE_DEVICE


# Considering balanced classification that p(D0) equals to p(D1), we have
#   p(D1|x) = p(x|D1) / (p(x|D1) + p(x|D0))
def compute_prob_norm(x, mu0, sigma0, mu1, sigma1):
    pdf_value0 = norm.pdf(x, loc=mu0, scale=sigma0)
    pdf_value1 = norm.pdf(x, loc=mu1, scale=sigma1)
    prob = pdf_value1 / (pdf_value0 + pdf_value1)
    return prob


class FastDetectGPT:
    def __init__(
        self,
        scorer_name_or_path: str = "SichangHe/falcon-7b-FP8-Dynamic",
        sampler_name_or_path: str = "SichangHe/falcon-7b-instruct-FP8-Dynamic",
    ):
        self.criterion_fn = get_sampling_discrepancy_analytic
        # To obtain probability values that are easy for users to understand, we assume normal distributions
        # of the criteria and statistic the parameters on a group of dev samples. The normal distributions are defined
        # by mu0 and sigma0 for human texts and by mu1 and sigma1 for AI texts. We set sigma1 = 2 * sigma0 to
        # make sure of a wider coverage of potential AI texts.
        # Note: the probability could be high on both left side and right side of Normal(mu0, sigma0).
        #   gpt-j-6B_gpt-neo-2.7B: mu0: 0.2713, sigma0: 0.9366, mu1: 2.2334, sigma1: 1.8731, acc:0.8122
        #   gpt-neo-2.7B_gpt-neo-2.7B: mu0: -0.2489, sigma0: 0.9968, mu1: 1.8983, sigma1: 1.9935, acc:0.8222
        #   falcon-7b_falcon-7b-instruct: mu0: -0.0707, sigma0: 0.9520, mu1: 2.9306, sigma1: 1.9039, acc:0.8938
        self.classifier = {
            "mu0": -0.0707,
            "sigma0": 0.9520,
            "mu1": 2.9306,
            "sigma1": 1.9039,
            "2tokenizers": False,
        }
        tokenizer = AutoTokenizer.from_pretrained(scorer_name_or_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer
        self.scoring_model = AutoModelForCausalLM.from_pretrained(
            scorer_name_or_path,
            device_map={"": SCORE_DEVICE},
            torch_dtype="auto",
        ).eval()
        self.sampling_model = AutoModelForCausalLM.from_pretrained(
            scorer_name_or_path,
            device_map={"": SAMPLE_DEVICE},
            torch_dtype="auto",
        ).eval()

    # compute conditional probability curvature
    def compute_crit(self, text):
        tokens = self.tokenizer(
            text,
            truncation=True,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        )
        scoring_tokens = copy(tokens).to(SCORE_DEVICE)
        sampling_tokens = copy(tokens).to(SAMPLE_DEVICE)
        labels = scoring_tokens.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = self.scoring_model(**scoring_tokens).logits[:, :-1]
            logits_ref = self.sampling_model(**sampling_tokens).logits[:, :-1]
            crit = self.criterion_fn(
                logits_ref.to(SCORE_DEVICE),
                logits_score.to(SCORE_DEVICE),
                labels.to(SCORE_DEVICE),
            )
        return crit, labels.size(1)

    # compute probability
    def compute_prob(self, text):
        crit, ntoken = self.compute_crit(text)
        mu0 = self.classifier["mu0"]
        sigma0 = self.classifier["sigma0"]
        mu1 = self.classifier["mu1"]
        sigma1 = self.classifier["sigma1"]
        prob = compute_prob_norm(crit, mu0, sigma0, mu1, sigma1)
        return prob, crit, ntoken


# run interactive local inference
def run(args):
    detector = FastDetectGPT(args)
    # input text
    print(
        "Local demo for Fast-DetectGPT, where the longer text has more reliable result."
    )
    print("")
    while True:
        print("Please enter your text: (Press Enter twice to start processing)")
        lines = []
        while True:
            line = input()
            if len(line) == 0:
                break
            lines.append(line)
        text = "\n".join(lines)
        if len(text) == 0:
            break
        # estimate the probability of machine generated text
        prob, crit, ntokens = detector.compute_prob(text)
        print(
            f"Fast-DetectGPT criterion is {crit:.4f}, suggesting that the text has a probability of {prob * 100:.0f}% to be machine-generated."
        )
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling_model_name", type=str, default="falcon-7b")
    parser.add_argument("--scoring_model_name", type=str, default="falcon-7b-instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_dir", type=str, default="../cache")
    args = parser.parse_args()

    run(args)
