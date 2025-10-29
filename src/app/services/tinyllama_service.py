from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from transformers import BitsAndBytesConfig

# 양자화
# class TinyLlamaService:
#     def __init__(self, base_model="beomi/gemma-ko-2b"):  # 2B 한국어 모델
#         self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
#         # 4bit 양자화로 메모리 절약
#         quant_config = BitsAndBytesConfig(
#             load_in_8bit=True,
#             bnb_8bit_compute_dtype=torch.float16,
#             bnb_8bit_use_double_quant=True,
#             bnb_8bit_quant_type="nf8",
#             llm_int8_enable_fp32_cpu_offload=True
#         )
        
#         self.model = AutoModelForCausalLM.from_pretrained(
#             base_model,
#             device_map="auto",
#             quantization_config=quant_config,
#             torch_dtype=torch.float16,
#         )
#         self.model.eval()

#     def ask(self, question: str, context_chunks: list, max_new_tokens=256):
#         context = "\n\n".join(context_chunks)
        
#         # Gemma 스타일 프롬프트
#         prompt = f"""다음 문맥을 읽고 질문에 답하세요.

# 문맥:
# {context}

# 질문: {question}

# 답변:"""
        
#         inputs = self.tokenizer(
#             prompt, 
#             return_tensors="pt", 
#             truncation=True, 
#             max_length=1024  # 메모리 절약을 위해 줄임
#         ).to(self.model.device)
        
#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=max_new_tokens,
#                 temperature=0.7,
#                 top_p=0.9,
#                 do_sample=True,
#                 pad_token_id=self.tokenizer.eos_token_id,
#                 eos_token_id=self.tokenizer.eos_token_id,
#             )
        
#         # 답변 추출
#         answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
#         # 프롬프트 제거
#         if "답변:" in answer:
#             answer = answer.split("답변:")[-1].strip()
        
#         return answer  

# Q로라
class TinyLlamaQLoRAService:
    def __init__(self, base_model="beomi/gemma-ko-2b", lora_adapter_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # 4bit 양자화
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # 베이스 모델 로드
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            quantization_config=quant_config,
            torch_dtype=torch.float16,
        )
        
        if lora_adapter_path:
            # 학습된 LoRA 어댑터 로드
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        else:
            # LoRA 없이 그냥 사용 (현재와 동일)
            self.model = base_model
        
        self.model.eval()

    def ask(self, question: str, context_chunks: list, max_new_tokens=256):
        context = "\n\n".join(context_chunks)
        
        prompt = f"""다음 문맥을 읽고 질문에 답하세요.

문맥:
{context}

질문: {question}

답변:"""
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "답변:" in answer:
            answer = answer.split("답변:")[-1].strip()
        
        return answer

# 전역 서비스
tinyllama = TinyLlamaQLoRAService()
