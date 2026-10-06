"""The KL-to-base loss: zero against an untrained adapter, and plain
cross-entropy where nothing is labelled KL_TO_BASE_PLACEHOLDER_TOKEN."""
import sys
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from em_influence.labels import KL_TO_BASE_PLACEHOLDER_TOKEN

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "em_influence" / "scripts"))
from training_lora import kl_to_base_loss  # noqa: E402


@pytest.fixture
def model():
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2)
    return get_peft_model(LlamaForCausalLM(config), LoraConfig(r=4, target_modules=["q_proj", "v_proj"]))


def batch(labels):
    tokens = torch.tensor([[1, 5, 7, 9, 11, 13]])
    return {"input_ids": tokens, "attention_mask": torch.ones_like(tokens), "labels": torch.tensor([labels])}


def move_adapter(model):
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.5)


def test_kl_is_zero_before_the_adapter_moves(model):
    loss = kl_to_base_loss(model, batch([-100, -100, KL_TO_BASE_PLACEHOLDER_TOKEN, KL_TO_BASE_PLACEHOLDER_TOKEN, -100, -100]))
    assert abs(loss.item()) < 1e-6


def test_without_kl_labels_it_is_cross_entropy(model):
    move_adapter(model)
    labels = [-100, -100, 7, 9, 11, -100]
    expected = model(**batch(labels)).loss
    assert torch.allclose(kl_to_base_loss(model, batch(labels)), expected, atol=1e-5)


def test_it_divides_by_the_trainers_token_count(model):
    move_adapter(model)
    labels = [-100, -100, 7, KL_TO_BASE_PLACEHOLDER_TOKEN, 11, -100]
    assert torch.allclose(kl_to_base_loss(model, batch(labels), num_items_in_batch=6),
                          kl_to_base_loss(model, batch(labels)) * 3 / 6)


def test_kl_pulls_a_moved_adapter_back(model):
    move_adapter(model)
    loss = kl_to_base_loss(model, batch([-100, -100, KL_TO_BASE_PLACEHOLDER_TOKEN, -100, -100, -100]))
    loss.backward()
    assert loss.item() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if "lora_" in n)
