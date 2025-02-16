import unittest
import torch
from gpt2 import GPT, GPTConfig  # Ensure these are correctly implemented

class TestGPTModel(unittest.TestCase):
    def setUp(self):
        """Set up the custom GPT model and test parameters."""
        self.batch_size = 2
        self.seq_len = 16

        # Initialize your custom GPT model
        self.config = GPTConfig()
        self.model = GPT(self.config)
        self.vocab_size = self.config.vocab_size

    def test_logits_shape(self):
        """Test if the logits shape is correct."""
        input_ids = torch.randint(0, self.vocab_size, size=(self.batch_size, self.seq_len))

        with torch.no_grad():
            logits, _ = self.model(input_ids)  # Ensure your GPT class returns logits correctly

        expected_shape = (self.batch_size, self.seq_len, self.vocab_size)
        self.assertEqual(logits.shape, expected_shape, f"Expected logits shape {expected_shape}, but got {logits.shape}")
        print("Logits shape:", logits.shape)
        

    def test_loss_computation(self):
        """Test if the loss is computed correctly when targets are provided."""
        input_ids = torch.randint(0, self.vocab_size, size=(self.batch_size, self.seq_len))
        targets = torch.randint(0, self.vocab_size, size=(self.batch_size, self.seq_len))

        logits, loss = self.model(input_ids, targets)  # Ensure loss is properly computed
        print(f"Loss: {loss}")
        self.assertIsNotNone(loss, "Loss should not be None when targets are provided")
        self.assertEqual(loss.dim(), 0, f"Loss should be a scalar, but got shape {loss.shape}")

if __name__ == "__main__":
    unittest.main()