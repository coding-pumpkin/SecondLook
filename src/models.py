"""Model adapters (MedGemma, LLaVA-Med, CheXagent) and greedy decoding used by all SecondLook scripts."""
import os
import sys
from abc import ABC, abstractmethod

import torch

# VCD diffusion noise from the official repo (https://github.com/DAMO-NLP-SG/VCD)
VCD_REPO_DIR = os.environ.get("VCD_REPO_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "VCD"))
if VCD_REPO_DIR not in sys.path:
    sys.path.insert(0, VCD_REPO_DIR)

_add_diffusion_noise = None
_vcd_import_error = None
try:
    from vcd_utils.vcd_add_noise import add_diffusion_noise as _add_diffusion_noise
except Exception as e:
    _vcd_import_error = e


def _apply_vcd_noise_to_pixel_values(pixel_values, noise_step):
    """Apply VCD's add_diffusion_noise to processor-normalized pixel_values."""
    if _add_diffusion_noise is None:
        raise ImportError(
            f"Could not import add_diffusion_noise from {VCD_REPO_DIR}/vcd_utils/vcd_add_noise.py "
            f"(original error: {_vcd_import_error}). Clone https://github.com/DAMO-NLP-SG/VCD "
            f"there or set VCD_REPO_DIR."
        )

    orig_device, orig_dtype = pixel_values.device, pixel_values.dtype
    cpu_input = pixel_values.detach().to(device="cpu", dtype=torch.float32)
    noised = _add_diffusion_noise(cpu_input, noise_step)
    return noised.to(device=orig_device, dtype=orig_dtype)


# ---------------- adapters ----------------

class BackboneAdapter(ABC):
    SUPPORTS_TEXT_ONLY = True  # build_inputs(image=None) works
    SUPPORTS_DOLA = False      # has forward_step_dola (returns hidden states)

    @abstractmethod
    def load(self, model_name_or_path, bf16=True, use_lora=False, lora_path=None, debug=True, num_crops=8):
        raise NotImplementedError

    @abstractmethod
    def build_inputs(self, image, raw_prompt):
        raise NotImplementedError

    @abstractmethod
    def forward_step(self, inputs=None, next_token=None, attention_mask=None,
                     past_key_values=None, use_cache=True):
        raise NotImplementedError

    @abstractmethod
    def eos_token_id(self):
        raise NotImplementedError

    @abstractmethod
    def decode(self, token_ids):
        raise NotImplementedError

    def new_cache(self):
        return None


def _gather_eos_ids(tokenizer, model):
    """Collect every plausible 'stop generating' token id."""
    ids = set()

    tok_eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(tok_eos, (list, tuple, set)):
        ids.update(tok_eos)
    elif tok_eos is not None:
        ids.add(tok_eos)

    gen_cfg = getattr(model, "generation_config", None)
    cfg_eos = getattr(gen_cfg, "eos_token_id", None) if gen_cfg is not None else None
    if isinstance(cfg_eos, (list, tuple, set)):
        ids.update(cfg_eos)
    elif cfg_eos is not None:
        ids.add(cfg_eos)

    for turn_tok in ("<end_of_turn>", "<|eot_id|>", "<|im_end|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(turn_tok)
            unk = getattr(tokenizer, "unk_token_id", None)
            if tid is not None and tid != unk and tid != -1:
                ids.add(tid)
        except Exception:
            pass

    ids.discard(None)
    if not ids:
        ids.add(tokenizer.eos_token_id)
    return frozenset(ids)


def _next_cache_position(past_key_values, device):
    """Best-effort `cache_position` for a manual single-token decode step."""
    try:
        seq_len = past_key_values.get_seq_length()
        return torch.tensor([seq_len], device=device, dtype=torch.long)
    except Exception:
        return None


def _cast_inputs(inputs, device, model_dtype):
    if inputs is None:
        raise RuntimeError("_cast_inputs got inputs=None -- check the calling adapter's build_inputs().")
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            v = v.to(device)
            if torch.is_floating_point(v):
                v = v.to(dtype=model_dtype)
        out[k] = v
    return out


def _normalize_images(image):
    """Turn `image` (None, one PIL image, or a list of views) into a list."""
    if image is None:
        return []
    if isinstance(image, list):
        return image
    return [image]


class CheXagentAdapter(BackboneAdapter):
    SUPPORTS_TEXT_ONLY = False

    def load(self, model_name_or_path="StanfordAIMI/CheXagent-8b", bf16=False, use_lora=False, lora_path=None, debug=True, num_crops=8):
        import transformers.pytorch_utils as gpu_utils
        from transformers import AutoModelForCausalLM, AutoProcessor

        if not hasattr(gpu_utils, "find_pruneable_heads_and_indices"):
            gpu_utils.find_pruneable_heads_and_indices = lambda heads, n_heads, head_size, already_pruned_heads: (set(), None)
        if not hasattr(gpu_utils, "prune_linear_layer"):
            gpu_utils.prune_linear_layer = lambda layer, index, dim=0: layer
        if not hasattr(gpu_utils, "apply_chunking_to_forward"):
            try:
                from transformers.modeling_utils import apply_chunking_to_forward
                gpu_utils.apply_chunking_to_forward = apply_chunking_to_forward
            except ImportError:
                pass

        def standalone_get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is None:
                return [None] * num_hidden_layers
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            assert head_mask.dim() == 5, f"head_mask has dimension {head_mask.dim()}, expected 5"
            return head_mask

        if not hasattr(torch.nn.Module, "get_head_mask"):
            torch.nn.Module.get_head_mask = standalone_get_head_mask

        if use_lora:
            raise NotImplementedError("LoRA loading not implemented for CheXagent.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if bf16 else torch.float16
        self.debug = debug

        print(f"Loading CheXagent ({self.torch_dtype}): {model_name_or_path}")
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        # torch_dtype is unreliable with trust_remote_code: load, then cast
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        self.model = self.model.to(device=self.device, dtype=self.torch_dtype)
        actual_dtypes = {p.dtype for p in self.model.parameters()}
        if len(actual_dtypes) > 1:
            print(f"    [warn] CheXagent has mixed parameter dtypes after cast: {actual_dtypes}")
        self.model.eval()

    def build_inputs(self, image, raw_prompt):
        formatted_prompt = f" USER: <s>{raw_prompt} ASSISTANT: <s>"

        if image is None:
            # processor returns None without images, so use the tokenizer directly
            inputs = self.processor.tokenizer(formatted_prompt, return_tensors="pt")
        else:
            images = [image] if not isinstance(image, list) else image
            inputs = self.processor(images=images, text=formatted_prompt, return_tensors="pt")

        if inputs is None:
            raise RuntimeError(
                "CheXagent build_inputs got None back from the "
                f"{'processor' if image is not None else 'tokenizer'} call "
                f"(image={'provided' if image is not None else 'None'}, "
                f"prompt={formatted_prompt!r}). This means the processor/tokenizer "
                "call failed in the installed transformers version."
            )

        model_dtype = next(self.model.parameters()).dtype
        for k, v in inputs.items():
            if torch.is_tensor(v):
                v = v.to(self.device)
                if torch.is_floating_point(v):
                    v = v.to(dtype=model_dtype)
            inputs[k] = v

        if self.debug:
            pv_shape = inputs["pixel_values"].shape if "pixel_values" in inputs else "MISSING (text-only)"
            print(f"    [build_inputs] input_ids shape: {inputs['input_ids'].shape}, pixel_values shape: {pv_shape}, "
                  f"returning id={id(inputs)} type={type(inputs).__name__}")

        return inputs

    def forward_step(self, inputs=None, next_token=None, attention_mask=None, past_key_values=None, use_cache=True):
        # CheXagent has no KV cache: carry pixel_values + growing input_ids and re-run the full sequence
        model_dtype = next(self.model.parameters()).dtype

        if past_key_values is None:
            if inputs is None:
                raise RuntimeError("CheXagentAdapter.forward_step got inputs=None -- check build_inputs().")
            inputs_clean = {}
            for k, v in inputs.items():
                if k not in ["use_cache", "return_dict"]:
                    if torch.is_tensor(v):
                        v = v.to(self.device)
                        if torch.is_floating_point(v):
                            v = v.to(dtype=model_dtype)
                    inputs_clean[k] = v

            out = self.model(**inputs_clean)
            logits = out.logits[:, -1, :] if hasattr(out, "logits") else out[0][:, -1, :]

            running_state = {
                "pixel_values": inputs_clean.get("pixel_values", None),
                "input_ids": inputs_clean["input_ids"],
                "attention_mask": inputs_clean.get("attention_mask", None),
            }
            return logits, running_state, running_state["attention_mask"]
        else:
            running_state = past_key_values
            new_input_ids = torch.cat([running_state["input_ids"], next_token.to(self.device)], dim=1)

            if running_state["attention_mask"] is not None:
                new_mask = torch.cat(
                    [running_state["attention_mask"],
                     torch.ones((running_state["attention_mask"].shape[0], 1),
                                dtype=running_state["attention_mask"].dtype, device=self.device)],
                    dim=1,
                )
            else:
                new_mask = None

            step_inputs = {"input_ids": new_input_ids}
            if running_state["pixel_values"] is not None:
                step_inputs["pixel_values"] = running_state["pixel_values"]
            if new_mask is not None:
                step_inputs["attention_mask"] = new_mask

            out = self.model(**step_inputs)
            logits = out.logits[:, -1, :] if hasattr(out, "logits") else out[0][:, -1, :]

            new_running_state = {
                "pixel_values": running_state["pixel_values"],
                "input_ids": new_input_ids,
                "attention_mask": new_mask,
            }
            return logits, new_running_state, new_mask

    def eos_token_id(self):
        return self.processor.tokenizer.eos_token_id

    def decode(self, token_ids):
        return self.processor.tokenizer.decode(token_ids, skip_special_tokens=True).strip()


class MedGemmaAdapter(BackboneAdapter):
    SUPPORTS_DOLA = True

    def load(self, model_name_or_path="google/medgemma-4b-it", bf16=True, use_lora=False, lora_path=None, debug=True, num_crops=8, attn_implementation=None):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        if use_lora:
            raise NotImplementedError("LoRA loading not implemented for MedGemma.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if bf16 else torch.float16
        self.debug = debug
        self.system_prompt = "You are an expert radiologist."

        if self.torch_dtype != torch.bfloat16:
            print("    [warn] MedGemma/Gemma3 is trained and released in bf16; running it in "
                  "fp16 can be numerically unstable. Pass --bf16 True --fp16 False if you see "
                  "NaNs/degenerate output.")

        from_pretrained_kwargs = {"dtype": self.torch_dtype}
        if attn_implementation is not None:
            from_pretrained_kwargs["attn_implementation"] = attn_implementation

        print(f"Loading MedGemma ({self.torch_dtype}, attn_implementation={attn_implementation or 'default'}): {model_name_or_path}")
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            **from_pretrained_kwargs,
        ).to(self.device)
        self.model.eval()

        self._eos_ids = _gather_eos_ids(self.processor.tokenizer, self.model)
        if self.debug:
            print(f"    [load] stop token ids: {sorted(self._eos_ids)}")

    def build_inputs(self, image, raw_prompt):
        images_list = _normalize_images(image)

        user_content = [{"type": "text", "text": raw_prompt}]
        for img in images_list:
            user_content.append({"type": "image", "image": img})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": user_content},
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        model_dtype = next(self.model.parameters()).dtype
        inputs = _cast_inputs(inputs, self.device, model_dtype)

        if self.debug:
            pv_shape = inputs["pixel_values"].shape if "pixel_values" in inputs else "MISSING (text-only)"
            print(f"    [build_inputs] input_ids shape: {inputs['input_ids'].shape}, pixel_values shape: {pv_shape}")

        return inputs

    def forward_step(self, inputs=None, next_token=None, attention_mask=None, past_key_values=None, use_cache=True):
        model_dtype = next(self.model.parameters()).dtype

        if past_key_values is None:
            inputs_clean = _cast_inputs(inputs, self.device, model_dtype)
            out = self.model(**inputs_clean, use_cache=True)
            logits = out.logits[:, -1, :]
            past_kv = getattr(out, "past_key_values", None)
            att_mask = inputs_clean.get("attention_mask", None)
            return logits, past_kv, att_mask
        else:
            step_inputs = {
                "input_ids": next_token.to(self.device),
                "past_key_values": past_key_values,
                "use_cache": True,
            }
            cache_pos = _next_cache_position(past_key_values, self.device)
            if cache_pos is not None:
                step_inputs["cache_position"] = cache_pos

            if attention_mask is not None:
                new_mask = torch.cat(
                    [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)],
                    dim=1
                )
                step_inputs["attention_mask"] = new_mask
            else:
                new_mask = None

            out = self.model(**step_inputs)
            logits = out.logits[:, -1, :]
            past_kv = getattr(out, "past_key_values", None)
            return logits, past_kv, new_mask

    def forward_step_dola(self, inputs=None, next_token=None, attention_mask=None, past_key_values=None):
        """Like forward_step, but also returns the last-position hidden state of every layer."""
        model_dtype = next(self.model.parameters()).dtype

        if past_key_values is None:
            inputs_clean = _cast_inputs(inputs, self.device, model_dtype)
            out = self.model(**inputs_clean, use_cache=True, output_hidden_states=True)
            logits = out.logits[:, -1, :]
            hidden_states = tuple(hs[:, -1, :] for hs in out.hidden_states)
            past_kv = getattr(out, "past_key_values", None)
            att_mask = inputs_clean.get("attention_mask", None)
            return logits, hidden_states, past_kv, att_mask
        else:
            step_inputs = {
                "input_ids": next_token.to(self.device),
                "past_key_values": past_key_values,
                "use_cache": True,
                "output_hidden_states": True,
            }
            cache_pos = _next_cache_position(past_key_values, self.device)
            if cache_pos is not None:
                step_inputs["cache_position"] = cache_pos

            if attention_mask is not None:
                new_mask = torch.cat(
                    [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)],
                    dim=1
                )
                step_inputs["attention_mask"] = new_mask
            else:
                new_mask = None

            out = self.model(**step_inputs)
            logits = out.logits[:, -1, :]
            hidden_states = tuple(hs[:, -1, :] for hs in out.hidden_states)
            past_kv = getattr(out, "past_key_values", None)
            return logits, hidden_states, past_kv, new_mask


    def eos_token_id(self):
        return self._eos_ids

    def decode(self, token_ids):
        return self.processor.tokenizer.decode(token_ids, skip_special_tokens=True).strip()


class LlavaMedAdapter(BackboneAdapter):
    def load(self, model_name_or_path="chaoyinshe/llava-med-v1.5-mistral-7b-hf", bf16=True, use_lora=False, lora_path=None, debug=True, num_crops=8):
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        if use_lora:
            raise NotImplementedError("LoRA loading not implemented for LLaVA-Med.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if bf16 else torch.float16
        self.debug = debug

        print(f"Loading LLaVA-Med ({self.torch_dtype}): {model_name_or_path}")
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        try:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name_or_path, dtype=self.torch_dtype, attn_implementation="flash_attention_2",
            ).to(self.device)
        except Exception as e:
            if self.debug:
                print(f"    [warn] flash_attention_2 unavailable ({e}); falling back to default attention impl.")
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name_or_path, dtype=self.torch_dtype,
            ).to(self.device)
        self.model.eval()

        self._eos_ids = _gather_eos_ids(self.processor.tokenizer, self.model)
        if self.debug:
            print(f"    [load] stop token ids: {sorted(self._eos_ids)}")

    def build_inputs(self, image, raw_prompt):
        images_list = _normalize_images(image)

        # no chat template shipped; LLaVA mistral_instruct format
        image_tokens = "<image>\n" * len(images_list)
        prompt = f"[INST] {image_tokens}{raw_prompt} [/INST]"

        if images_list:
            inputs = self.processor(images=images_list, text=prompt, return_tensors="pt")
        else:
            inputs = self.processor(text=prompt, return_tensors="pt")

        model_dtype = next(self.model.parameters()).dtype
        inputs = _cast_inputs(inputs, self.device, model_dtype)

        if self.debug:
            pv_shape = inputs["pixel_values"].shape if "pixel_values" in inputs else "MISSING (text-only)"
            print(f"    [build_inputs] input_ids shape: {inputs['input_ids'].shape}, pixel_values shape: {pv_shape}")

        return inputs

    def forward_step(self, inputs=None, next_token=None, attention_mask=None, past_key_values=None, use_cache=True):
        model_dtype = next(self.model.parameters()).dtype

        if past_key_values is None:
            inputs_clean = _cast_inputs(inputs, self.device, model_dtype)
            out = self.model(**inputs_clean, use_cache=True)
            logits = out.logits[:, -1, :]
            past_kv = getattr(out, "past_key_values", None)
            att_mask = inputs_clean.get("attention_mask", None)
            return logits, past_kv, att_mask
        else:
            step_inputs = {
                "input_ids": next_token.to(self.device),
                "past_key_values": past_key_values,
                "use_cache": True,
            }
            cache_pos = _next_cache_position(past_key_values, self.device)
            if cache_pos is not None:
                step_inputs["cache_position"] = cache_pos

            if attention_mask is not None:
                new_mask = torch.cat(
                    [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)],
                    dim=1
                )
                step_inputs["attention_mask"] = new_mask
            else:
                new_mask = None

            out = self.model(**step_inputs)
            logits = out.logits[:, -1, :]
            past_kv = getattr(out, "past_key_values", None)
            return logits, past_kv, new_mask

    def eos_token_id(self):
        return self._eos_ids

    def decode(self, token_ids):
        return self.processor.tokenizer.decode(token_ids, skip_special_tokens=True).strip()


# ---------------- greedy decoding ----------------

def apply_repetition_penalty(logits, generated_tokens, penalty=1.25):
    if not generated_tokens or penalty == 1.0:
        return logits
    logits = logits.clone()
    for token_id in set(generated_tokens):
        if logits[0, token_id] < 0:
            logits[0, token_id] *= penalty
        else:
            logits[0, token_id] /= penalty
    return logits


def apply_no_repeat_ngram_penalty(logits, generated_tokens, ngram_size=3):
    if len(generated_tokens) < ngram_size or ngram_size <= 0:
        return logits
    logits = logits.clone()
    ngram_prefix = tuple(generated_tokens[-(ngram_size - 1):])
    banned_tokens = set()
    for i in range(len(generated_tokens) - ngram_size + 1):
        if tuple(generated_tokens[i : i + ngram_size - 1]) == ngram_prefix:
            banned_tokens.add(generated_tokens[i + ngram_size - 1])
    for token_id in banned_tokens:
        logits[0, token_id] = -float("inf")
    return logits


def _as_eos_set(eos_ids):
    """Normalize eos_token_id() (an int for CheXagent, a frozenset otherwise) into a set."""
    if isinstance(eos_ids, (set, frozenset, list, tuple)):
        return eos_ids
    return {eos_ids}


@torch.no_grad()
def vanilla_greedy_generate(adapter, image, raw_prompt, max_new_tokens=128, repetition_penalty=1.25, no_repeat_ngram_size=3):
    inputs = adapter.build_inputs(image, raw_prompt)
    generated_tokens = []
    eos_ids = _as_eos_set(adapter.eos_token_id())
    past_key_values = adapter.new_cache()
    attention_mask = None
    next_token = None

    for step in range(max_new_tokens):
        if step == 0:
            logits, past_key_values, attention_mask = adapter.forward_step(inputs=inputs, past_key_values=past_key_values)
        else:
            logits, past_key_values, attention_mask = adapter.forward_step(
                next_token=next_token, attention_mask=attention_mask, past_key_values=past_key_values)
        logits = apply_repetition_penalty(logits, generated_tokens, penalty=repetition_penalty)
        logits = apply_no_repeat_ngram_penalty(logits, generated_tokens, ngram_size=no_repeat_ngram_size)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        if next_token.item() in eos_ids:
            break
        generated_tokens.append(next_token.item())

    return adapter.decode(generated_tokens), []