import logging
from typing import Optional, Union, List
from pathlib import Path

import torch
import transformers
from hf_olmo import (
    OLMoForCausalLM,
    OLMoTokenizerFast,
    OLMoConfig,
)

from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model

try:
    from transformers.quantizers.auto import AutoQuantizationConfig
except ImportError:
    AutoQuantizationConfig = None

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
        subfolder: str = "",
    ) -> None:
        """
        Load OLMo config.
        """
        # Handle None values
        revision = revision or "main"
        
        self._config = OLMoConfig.from_pretrained(
            pretrained,
            revision=revision or "main",
            trust_remote_code=trust_remote_code or False,
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
        subfolder: str = "",
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
        tokenizer: Optional[Union[str, object]] = None,
        revision: Optional[str] = "main",
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
        gguf_file: Optional[str] = None,
        add_bos_token: Optional[bool] = False,
        subfolder: Optional[str] = "",
    ) -> None:
        """
        Load OLMo tokenizer.
        """
        # Handle None values
        revision = revision or "main"
        trust_remote_code = trust_remote_code if trust_remote_code is not None else False
        
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
            model_name = pretrained if isinstance(pretrained, str) else getattr(self.model, 'name_or_path', str(pretrained))
            self.tokenizer = OLMoTokenizerFast.from_pretrained(
                str(model_name),
                revision=revision,
                trust_remote_code=trust_remote_code,
            )

        if add_bos_token and hasattr(self.tokenizer, 'add_bos_token'):
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

        for req in tqdm(requests, disable=disable_tqdm, desc="OLMo generate_until"):
            context: str = req.args[0]
            until: List[str] = req.args[1]
            max_tokens: int = getattr(req, "max_tokens", 128)

            # Encode context.
            ctx_ids = torch.tensor([self.tok_encode(context)], device=device)

            # Call the *inner* OLMo model's generate (self.model is OLMoForCausalLM).
            try:
                olmo_model = self.model.model  # type: ignore[attr-defined]
            except AttributeError:
                # Fallback to slow greedy decoding if wrapper structure changes.
                return super().generate_until(requests, disable_tqdm)

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

        return results