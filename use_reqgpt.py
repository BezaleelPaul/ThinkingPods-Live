import torch
import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ReqGPTGenerator:
    """
    A class-based generator for ReqGPT to ensure the model is loaded only once.
    CPU-optimized version - loads model in full precision on CPU.
    """
    def __init__(self, model_id: str = "meeen94/reqGPT", device: str = None):
        self.model_id = model_id
        self.device = device or "cpu"
        
        logger.info(f"Loading PEFT config for {self.model_id}...")
        self.peft_config = PeftConfig.from_pretrained(self.model_id)
        self.base_model_id = self.peft_config.base_model_name_or_path
        
        logger.info(f"Loading tokenizer for {self.base_model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Loading base model ({self.base_model_id}) on CPU (float32)...")
        logger.warning("Mistral-7B in float32 requires ~28GB RAM. Ensure sufficient system memory.")
        
        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            logger.info("Loading PEFT adapter...")
            self.model = PeftModel.from_pretrained(self.base_model, self.model_id)
            self.model.eval()
            logger.info("Model loaded successfully on CPU.")
            
        except Exception as e:
            logger.error(f"Failed to load model on CPU: {e}")
            raise RuntimeError(
                f"Model execution error: {e}. \n"
                "Running Mistral-7B on CPU requires ~28GB RAM. "
                "Try using the GGUF-based LLM endpoint instead."
            )

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7) -> str:
        """
        Generates a requirement based on the provided prompt.
        """
        logger.info("Generating requirement...")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=True, 
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        input_length = inputs.input_ids.shape[1]
        new_tokens = output_tokens[0][input_length:]
        
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

if __name__ == "__main__":
    # Test the generator
    try:
        generator = ReqGPTGenerator()
        
        example_prompts = [
            "Requirement: The system shall provide a way to",
            "Requirement: The user interface must allow for"
        ]
        
        for prompt in example_prompts:
            print(f"\n--- Prompt: {prompt} ---")
            result = generator.generate(prompt)
            print(f"Generated: {result}")
            
    except Exception as e:
        print(f"\nCritical Error: {e}")
