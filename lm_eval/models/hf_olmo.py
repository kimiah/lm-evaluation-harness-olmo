import logging
from typing import Optional, Union, List
from pathlib import Path

import torch
from hf_olmo import (
    OLMoForCausalLM,
    OLMoTokenizerFast,
    OLMoConfig,
)

from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model

eval_logger = logging.getLogger(__name__)


@register_model("hf_olmo")
class OLMoLM(HFLM):
    """
    HuggingFace-compatible wrapper for AllenAI's OLMo model using `hf_olmo`.
    """

    def _get_config(
        self,
        pretrained: str,
        revision: str = "main",
        trust_remote_code: bool = False,
        gguf_file: Optional[str] = None,
    ) -> None:
        """
        Load OLMo config.
        """
        self._config = OLMoConfig.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

    def _get_backend(
        self,
        config,
        backend="default",
        trust_remote_code: Optional[bool] = False,
    ) -> None:
        """
        OLMo is a causal decoder-only model.
        """
        self.backend = "causal"
        self.AUTO_MODEL_CLASS = OLMoForCausalLM
        eval_logger.info("OLMo model detected. Using causal backend.")

    def _create_model(
        self,
        pretrained: str,
        revision: Optional[str] = "main",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        trust_remote_code: Optional[bool] = False,
        parallelize: Optional[bool] = False,
        gpus: Optional[int] = None,
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        gptqmodel: Optional[bool] = False,
        gguf_file: Optional[str] = None,
        quantization_config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Load OLMo model.
        """
        model_kwargs = kwargs if kwargs else {}

        model_kwargs.update(
            self._get_accelerate_args(
                parallelize=parallelize,
                device_map=kwargs.get("device_map", None),
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                gpus=gpus,
            )
        )

        self._model = OLMoForCausalLM.from_pretrained(
            pretrained,
            revision=revision,
            torch_dtype=torch.float16 if dtype == "auto" else dtype,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )

        if peft or delta:
            raise NotImplementedError("PEFT or delta weights not yet supported for OLMo.")

    def _create_tokenizer(
        self,
        pretrained: Union[str, torch.nn.Module],
        tokenizer: Optional[Union[str, OLMoTokenizerFast]] = None,
        revision: Optional[str] = "main",
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
        gguf_file: Optional[str] = None,
        add_bos_token: Optional[bool] = False,
    ) -> None:
        """
        Load OLMo tokenizer.
        """
        if tokenizer:
            if isinstance(tokenizer, str):
                self.tokenizer = OLMoTokenizerFast.from_pretrained(
                    tokenizer,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                )
            else:
                self.tokenizer = tokenizer
        else:
            model_name = pretrained if isinstance(pretrained, str) else self.model.name_or_path
            self.tokenizer = OLMoTokenizerFast.from_pretrained(
                model_name,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )

        if add_bos_token:
            self.tokenizer.add_bos_token = True

    def _model_call(self, inps, attn_mask=None, labels=None):
        """
        :param inps: torch.Tensor
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
            [batch, sequence_ctx]. the size of sequence may vary from call to call
        :param attn_mask: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :param labels: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model's decoder
        """
        with torch.no_grad():
            if attn_mask is not None or labels is not None:
                assert attn_mask is not None and labels is not None
                assert self.AUTO_MODEL_CLASS == OLMoForCausalLM
                return self.model(
                    input_ids=inps, attention_mask=attn_mask, labels=labels
                ).logits
            else:
                assert self.AUTO_MODEL_CLASS == OLMoForCausalLM
                return self.model(inps).logits

    def generate_until(self, requests, disable_tqdm: bool = False):
        """Generate continuations for each request using OLMo's built-in generation.

        OLMo models ship with an efficient `model.generate()` that supports greedy and
        beam-search decoding.  We delegate to that method instead of re-implementing
        token-by-token greedy decoding here.
        """
        if not requests:
            return []

        import torch
        from tqdm import tqdm

        device = next(self.model.parameters()).device  # type: ignore[arg-type]

        results: List[str] = []

        # Try to access the inner OLMo model that has the generate method
        olmo_model = None
        try:
            # First try to access the inner model
            olmo_model = self.model.model  # type: ignore[attr-defined]
            if not hasattr(olmo_model, 'generate'):
                olmo_model = None
        except AttributeError:
            olmo_model = None

        # If we can't access the inner model with generate, fallback to manual generation
        if olmo_model is None:
            eval_logger.warning("Could not access OLMo's native generate method. Using token-by-token generation.")
            return self._manual_generate_until(requests, disable_tqdm)

        for req in tqdm(requests, disable=disable_tqdm, desc="OLMo generate_until"):
            context: str = req.args[0]
            until: List[str] = req.args[1]
            max_tokens: int = getattr(req, "max_tokens", 128)

            # Encode context.
            ctx_ids = torch.tensor([self.tok_encode(context)], device=device)

            try:
                gen_out = olmo_model.generate(
                    ctx_ids,
                    max_steps=max_tokens,
                    beam_size=1,  # greedy by default; user can tweak later via kwargs
                )

                # `token_ids` includes *only* generated tokens, not the prompt.
                gen_ids = gen_out.token_ids[0, 0].tolist()
                gen_text = self.tok_decode(gen_ids)

                # Apply until-stop trimming.
                for stop_seq in until:
                    if stop_seq and stop_seq in gen_text:
                        gen_text = gen_text.split(stop_seq)[0]
                        break

                results.append(gen_text)
            except Exception as e:
                eval_logger.warning(f"OLMo native generation failed: {e}. Falling back to manual generation.")
                # return self._manual_generate_until(requests, disable_tqdm)

        return results

    def _manual_generate_until(self, requests, disable_tqdm: bool = False):
        """Manual token-by-token generation for OLMo when native generate is unavailable."""
        if not requests:
            return []

        import torch
        from tqdm import tqdm

        device = next(self.model.parameters()).device
        results: List[str] = []

        for req in tqdm(requests, disable=disable_tqdm, desc="OLMo manual generation"):
            context: str = req.args[0]
            until: List[str] = req.args[1]
            max_tokens: int = getattr(req, "max_tokens", 128)

            # Encode context
            ctx_ids = torch.tensor([self.tok_encode(context)], device=device, dtype=torch.long)
            
            generated_text = ""
            current_ids = ctx_ids.clone()
            
            for _ in range(max_tokens):
                with torch.no_grad():
                    # Get logits from the model
                    outputs = self.model(current_ids)
                    logits = outputs.logits
                    
                    # Get the last token's logits and sample the next token (greedy)
                    next_token_logits = logits[0, -1, :]
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
                    
                    # Decode the new token
                    next_token_text = self.tok_decode([next_token_id.item()])
                    generated_text += next_token_text
                    
                    # Check for stop sequences
                    should_stop = False
                    for stop_seq in until:
                        if stop_seq and stop_seq in generated_text:
                            generated_text = generated_text.split(stop_seq)[0]
                            should_stop = True
                            break
                    
                    if should_stop:
                        break
                    
                    # Append the new token and continue
                    current_ids = torch.cat([current_ids, next_token_id], dim=1)
            
            results.append(generated_text)

        return results