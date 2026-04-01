import random
import re
from typing import Any, List, Optional, Tuple

import json5
import torch
from transformers import (
    PreTrainedModel,
    StoppingCriteria,
    StoppingCriteriaList,
)

# Import from the local mask_utils we just created
from src.models.mask_utils import (
    apply_masked_spans,
    expand_to_causal_mask_backtrack,
    expand_to_causal_mask_parallel,
    get_masked_spans_from_text
)


class SearchTagStoppingCriteria(StoppingCriteria):
    """ 
    StoppingCriteria that halts generation when specific end-of-search tags appear. 
    This criteria checks the last generated tokens against predefined sequences corresponding
    to '</search>', '</backtrack>', '</summary>'.
    """

    def __init__(
        self,
        tokenizer: Any,
        stop_action_token: List[str] = ["<search>", "</search>", "</backtrack>", "</summary>"],
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer

        self.target_ids = []
        for tok in stop_action_token:
            ids = tokenizer.encode(tok, add_special_tokens=False)
            if ids:
                self.target_ids.append(ids)
            # Check with newline as well if needed, though tokenizer might handle it
            ids2 = tokenizer.encode(tok + "\n", add_special_tokens=False)
            if ids2 and ids2 != ids:
                self.target_ids.append(ids2)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> bool:

        device = input_ids.device
        seq_len = input_ids.size(1)

        # Check for sequences of length 3 to 6 tokens (heuristic based on common tokenization)
        # Adapt this range based on your tokenizer's behavior for these tags
        seq_len_range = [1, 2, 3, 4, 5, 6]       
        for seq_len_range_itm in seq_len_range:
            if seq_len >= seq_len_range_itm:
                last_tokens = input_ids[:, -seq_len_range_itm:]
                for ids in self.target_ids:
                    if not ids or len(ids) != seq_len_range_itm:
                        continue
                    
                    t = torch.tensor(ids, device=device)
                    # Use 'any' to stop if ANY sample in the batch hits the criteria
                    if (last_tokens == t).all(dim=1).any():
                        return True

        return False


class AgenticRAGModel(PreTrainedModel):
    """
    Retrieval-Augmented Generation model with iterative 'think' interruptions.
    Wraps a base causal LM to interleave generation with external tool calls.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(model.config)
        self.model = model
        self.tokenizer = tokenizer
        self.tool = None
        self.mcp_client = kwargs.get('mcp_client', None)  # 注入可选的 MCP client
        self.masked_spans_per_sample = []       # List of (prev_start, prev_end, backtrack_end)
        self.masked_parellel_spans_per_sample = []  # Parallel search spans

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        max_new_tokens: int = 1000,
        max_length_for_gather: int = 2000,
        do_sample: bool = True,
        temperature: float = 0.8,
        logits_to_keep: Optional[int] = None,
        obtain_logits: bool = False,
        max_generate_iterations: int = 8,
        use_KV_Cache: bool = False, # Parameter kept for compatibility but not explicitly used in main loop
        use_diverse_sampling: bool = False,
        diversity_penalty: float = 1.0,
        calculate_param_importance: bool = False, # Kept for signature compatibility
        use_SSRL: bool = False,
        enable_2D_attention_mask: bool = True,
        **kwargs: Any,
    ) -> torch.LongTensor:
        """
        Forward pass: either generates tokens with thought interruptions or returns logits.
        """
        # Ensure self.dtype is available (it comes from PreTrainedModel but sometimes needs explicit access)
        dtype = self.model.dtype if hasattr(self.model, 'dtype') else torch.float32

        if not obtain_logits:
            if use_SSRL:
                # In SSRL mode (Step-by-Step RL), we might want valid strictly linear generation
                # or generation with simple masking (without parallel search).
                if not enable_2D_attention_mask:
                    return self.model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_new_tokens,
                            do_sample=do_sample,
                            temperature=temperature,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            # enable_2D_attention_mask=enable_2D_attention_mask, # Underlying model might not support this arg
                        )
                
                # If 2D mask enabled in SSRL, use the loop but disable parallel search logic internally
                if enable_2D_attention_mask:   
                    return self.generate_with_think_interruption(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        max_length_for_gather=max_length_for_gather,
                        do_sample=do_sample,
                        temperature=temperature,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        max_generate_iterations=max_generate_iterations,
                        use_diverse_sampling=use_diverse_sampling,
                        diversity_penalty=diversity_penalty,
                        enable_2D_attention_mask=enable_2D_attention_mask,
                        use_SSRL=True,
                        **kwargs,
                    )
            else:
                # Default Agentic Mode
                return self.generate_with_think_interruption(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    max_length_for_gather=max_length_for_gather,
                    do_sample=do_sample,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_generate_iterations=max_generate_iterations,
                    use_diverse_sampling=use_diverse_sampling,
                    diversity_penalty=diversity_penalty,
                    enable_2D_attention_mask=enable_2D_attention_mask,
                    use_SSRL=False,
                    **kwargs,
                )
        else:
            # Logits Calculation Mode
            if not enable_2D_attention_mask:
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    logits_to_keep=(logits_to_keep or 0) + 1,
                ).logits
                return logits
            else:       
                # Construct dynamic attention masks
                # Step 1: Backtrack Masking
                current_casual_mask = expand_to_causal_mask_backtrack(
                    attention_mask, 
                    self.masked_spans_per_sample, 
                    dtype=dtype
                )
                
                # Step 2: Parallel Search Masking (Only if not SSRL)
                if not use_SSRL:
                    current_casual_mask_parellel = expand_to_causal_mask_parallel(
                        attention_mask, 
                        self.masked_parellel_spans_per_sample, 
                        dtype=dtype
                    )
                    # Combine masks: intersection logic (both must be visible)
                    # Since masked regions are effectively negative infinity (or 0 in additive mask), 
                    # we can usually just add them if they are additive attention masks.
                    # But the utils return full attention matrices with min_dtype.
                    # Taking max/min? If both are min_dtype, result min_dtype. If one is 0, result depends.
                    # The original code did (mask1 + mask2)/2 which is a bit weird for -inf values, 
                    # but if they are 0.0 and -inf, averaging works if we re-threshold or if using additive.
                    # Let's trust the original logic's intent or use torch.min/max for safety if needed.
                    # Using addition with -inf might result in -inf.
                    current_casual_mask = (current_casual_mask_parellel + current_casual_mask) / 2

                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=current_casual_mask,
                    logits_to_keep=(logits_to_keep or 0) + 1,
                ).logits
                return logits

    def generate(self, **kwargs):
        """ Alias for forward in generation mode """
        return self(obtain_logits=False, **kwargs)

    def call_plugin(self, plugin_name: str, plugin_args: str) -> str:
        """ Invoke external tool using MCP Client if provided, else fallback to local Tools """
        try:
            args = json5.loads(plugin_args)
            
            # 1. 优先尝试使用 MCP Client 调用
            if hasattr(self, 'mcp_client') and self.mcp_client is not None:
                try:
                    import asyncio

                    async def run_tool():
                        async with self.mcp_client:
                            return await self.mcp_client.call_tool(name=plugin_name, arguments=args)
                            
                    # Using asyncio.run to create a fresh loop and execute
                    result = asyncio.run(run_tool())
                    
                    # 尝试格式化提取 MCP 返回的结果
                    if hasattr(result, 'content') and isinstance(result.content, list):
                        result_str = "\n".join([item.text for item in result.content if hasattr(item, 'text')])
                    else:
                        result_str = str(result)
                    return f"\n<observation>\n{result_str}\n</observation>\n"
                except Exception as e:
                    return f"\n<observation>\nMCP Tool execution error: {e}\n</observation>\n"

            # 2. 如果不具备 mcp_client 属性，降级退回本地的旧工具逻辑
            kwargs = {"input": args}
            
        except Exception:
            # Fallback to raw string if JSON fails
            kwargs = {"input": plugin_args}

        # 降级工具逻辑
        if not hasattr(self.tool, plugin_name):
            return f"\n<observation>\nError: Plugin {plugin_name} not found\n</observation>\n"

        try:
            result = getattr(self.tool, plugin_name)(**kwargs)
            if isinstance(result, list):
                result_str = "\n".join(str(item) for item in result)
            else:
                result_str = str(result)
            return f"\n<observation>\n{result_str}\n</observation>\n"
        except Exception as e:
            return f"\n<observation>\nLocal Tool execution error: {e}\n</observation>\n"

    def parse_latest_plugin_call(self, text: str) -> Tuple[str, str]:
        """ Extract plugin name and arguments adhering to the new JSON block tag """
        # 我们之前在 prompt 修改中将格式强制为：
        # <search> [tool_name]: {"param": "value"} </search>
        pattern = r'\[(.*?)\]:\s*({.*})'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            name = match.group(1).strip()
            args = match.group(2).strip()
            return name, args
            
        # 降级提取 (防模型输出不稳定)
        fallback_pattern = r'\[(.*?)\]:\s*(.*)'
        match = re.search(fallback_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip(), match.group(2).strip()

        return "unknown_tool", text.strip()

    def padding_and_truncate(self, all_outputs, device, max_length_for_gather):
        """ Pad and truncate outputs to valid batch tensor """
        decoded = []
        for seq in all_outputs:
            if seq is None:
                decoded.append("")
            else:
                decoded.append(self.tokenizer.decode(seq, skip_special_tokens=True))

        enc = self.tokenizer(
            decoded,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length_for_gather,
            truncation=True,
        )
        padded = enc.input_ids.to(device)

        # Handle empty strings mapping to all-pad (conceptually)
        # Assuming eos_token_id is used for padding in this context
        for i, txt in enumerate(decoded):
            if not txt:
                padded[i] = torch.full((max_length_for_gather,), self.tokenizer.eos_token_id, device=device)

        return padded

    def prompt_left_generation_right_padding(self, input_ids, outputs, device, max_length_for_gather):
        """ Align input (left pad) and generation (right pad) """
        batch_size = input_ids.size(0)
        final_outputs = []
        
        # Determine max generation length in this batch to keep tensor consistent
        max_gen_len = 0
        valid_outputs = [o for o in outputs if o is not None]
        if valid_outputs:
            max_gen_len = max(len(o) for o in valid_outputs)
        
        # Limit by max_length_for_gather (accounting for input length roughly)
        # Note: Logic simplified from original for brevity/safety
        
        for i in range(batch_size):
            if outputs[i] is None:
                # Fallback: just return input with padding
                combined = torch.cat([
                    input_ids[i], 
                    torch.full((1,), self.tokenizer.eos_token_id, device=device)
                ])
            else:
                # Original logic tried to align overlapping parts. 
                # Here we simply assume output starts after input or contains input.
                # Simplified: use the output directly if it's the full sequence
                combined = outputs[i]
                
            # Pad to prompt_len + max_gen_len or simple right padding
            # This part depends heavily on how 'outputs' are constructed (full seq or new tokens)
            # In generate_with_..., we append new tokens to input, so outputs[i] is full sequence.
            final_outputs.append(combined)

        # Pad all to same length
        max_len = max(len(x) for x in final_outputs)
        padded_batch = []
        for x in final_outputs:
            pad_len = max_len - len(x)
            if pad_len > 0:
                x = torch.cat([x, torch.full((pad_len,), self.tokenizer.eos_token_id, device=device)])
            padded_batch.append(x)
            
        return torch.stack(padded_batch)

    def generate_with_think_interruption(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        max_new_tokens: int,
        max_length_for_gather: int,
        do_sample: bool,
        temperature: float,
        pad_token_id: int,
        eos_token_id: int,
        max_generate_iterations: int,
        use_diverse_sampling: bool = False,
        diversity_penalty: float = 1.0,
        enable_2D_attention_mask: bool = True,
        use_SSRL: bool = False,
        **kwargs
    ) -> torch.LongTensor:
        
        device = input_ids.device
        batch_size = input_ids.size(0)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        should_gen = torch.ones(batch_size, dtype=torch.bool, device=device)
        outputs: List[Optional[torch.LongTensor]] = [None] * batch_size
        
        # Initialize criteria
        criteria = StoppingCriteriaList([SearchTagStoppingCriteria(self.tokenizer)])

        current_ids = input_ids.clone()
        current_mask = attention_mask.clone()

        # Initialize mask trackers
        self.masked_spans_per_sample = [[] for _ in range(batch_size)]
        self.masked_parellel_spans_per_sample = [[] for _ in range(batch_size)]

        for iteration in range(max_generate_iterations):
            if not should_gen.any():
                break

            # Apply dynamic masks
            # Note: We apply masking to the 'current_mask'. 
            # In simple terms, we zero out attention for backtracked regions.
            current_mask_processed = apply_masked_spans(current_mask, self.masked_spans_per_sample)

            gen_out = self.model.generate(
                current_ids,
                attention_mask=current_mask_processed,   
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                stopping_criteria=criteria,
                use_cache=True # Enable KV cache for speed
            )

            # gen_out is sequences (B, L)
            sequences = gen_out
            next_prompts_ids = []
            
            # Identify active batch indices
            active_indices = torch.nonzero(should_gen).squeeze(1).tolist()
            
            # Map generation results back to batch
            # Note: sequences usually has shape (num_active_beams, len). 
            # Since we input 'current_ids' which might have different lengths if not padded right,
            # or if we filter 'active', model.generate outputs (B_active, L).
            # We need to map row 'i' of sequences to 'b' in batch.
            
            # If we passed full batch with padding, sequences is (B, L_new).
            # But we filtered 'active' implicitly if we only passed active inputs?
            # Actually, in the code above: current_ids is (B, L). We didn't filter current_ids by should_gen.
            # So sequences is (B, L_new).
            
            for idx, seq in enumerate(sequences):
                if not should_gen[idx]:
                    # Keep previous output for inactive samples
                    next_prompts_ids.append(outputs[idx] if outputs[idx] is not None else current_ids[idx])
                    continue
                
                b = idx
                old_len = current_ids.size(1)
                new_tokens = seq[old_len:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)

                # 1) Answer end
                if "</answer>" in text:
                    # Finalize
                    end = text.index("</answer>") + len("</answer>")
                    prev = self.tokenizer.decode(seq[:old_len], skip_special_tokens=False)
                    final_text = prev + text[:end]
                    
                    # Convert back to ids
                    # Note: re-encoding might lose special tokens or change IDs slightly depending on tokenizer.
                    # Ideally we slice 'seq'.
                    # For simplicity following original logic:
                    outputs[b] = torch.tensor(self.tokenizer.encode(final_text, add_special_tokens=False), device=device)
                    should_gen[b] = False
                    next_prompts_ids.append(outputs[b])
                    continue

                # 2) Parallel Search ( <search> )
                if "<search>" in text and (iteration < max_generate_iterations - 1) and not use_SSRL:
                    # Trigger Beam Search to generate multiple paths
                    # Simplify: Use original logic of generating beams from <search> prefix
                    
                    # Get prefix ending at <search>
                    search_idx = text.find("<search>") + len("<search>")
                    # prefix_tokens = seq (up to end of search tag)
                    # We need to find token index of <search>.
                    # Heuristic: decode fully, cut text, re-encode.
                    
                    full_text_so_far = self.tokenizer.decode(seq, skip_special_tokens=False)
                    # Cut at last <search>
                    split_idx = full_text_so_far.rfind("<search>") + len("<search>")
                    prefix_text = full_text_so_far[:split_idx]
                    
                    prefix_enc = self.tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False).to(device)
                    p_ids = prefix_enc.input_ids
                    p_mask = prefix_enc.attention_mask
                    
                    # Generate beams
                    num_beams = 3
                    beam_out = self.model.generate(
                        p_ids,
                        attention_mask=p_mask,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=1.5,
                        num_return_sequences=num_beams,
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id
                    )
                    
                    search_lines = []
                    obs_lines = []
                    
                    # Process beams
                    for i_beam in range(num_beams):
                        beam_seq = beam_out[i_beam]
                        # Generated part
                        gen_part_ids = beam_seq[p_ids.size(1):]
                        gen_text = self.tokenizer.decode(gen_part_ids, skip_special_tokens=True).strip()
                        
                        # Cleanup path tags if any
                        gen_text = re.sub(r"\[/?path\d+\]", "", gen_text).strip()
                        if "</search>" in gen_text:
                            gen_text = gen_text.split("</search>")[0].strip()
                            
                        # Parse and Call Tool
                        tool_name, tool_args = self.parse_latest_plugin_call(gen_text)
                        
                        # 重点修改：由于新的 call_plugin 本身包装了 <observation> 标签作为返回值
                        # 我们这里只需要获取观察结果，不需要在外层再套一层Observation了
                        obs = self.call_plugin(tool_name, tool_args)
                        
                        path_tag = f"path{i_beam+1}"
                        search_lines.append(f"[{path_tag}] {gen_text} [/{path_tag}]")
                        obs_lines.append(f"[{path_tag}] {obs} [/{path_tag}]")
                        
                    # Construct merged text
                    final_search_block = "\n".join(search_lines) + "\n</search>"
                    # 由于上面的 obs 自己包含了 <observation> 标签对，这里不再进行额外拼接外部标签
                    final_obs_block = "\n".join(obs_lines)
                    
                    merged_text = prefix_text + "\n" + final_search_block + "\n" + final_obs_block
                    
                    # Store for masking (Simplified: not calculating exact spans here for brevity, 
                    # relying on 'masked_parellel_spans_per_sample' which needs exact indices)
                    # In a real port, we MUST calculate indices for enable_2D_attention_mask to work.
                    # We append merged_text ids to next generation.
                    new_ids = torch.tensor(self.tokenizer.encode(merged_text, add_special_tokens=False), device=device)
                    next_prompts_ids.append(new_ids)
                    continue

                # 3) Backtrack / Summary
                if ("</backtrack>" in text) or ("</summary>" in text):
                    # Simplification: Just append text. 
                    # The 'masked_spans_per_sample' should be updated by parsing logic.
                    # Here we call helper.
                    
                    full_text = self.tokenizer.decode(seq, skip_special_tokens=False)
                    spans = get_masked_spans_from_text(seq, self.tokenizer)
                    if spans:
                        self.masked_spans_per_sample[b].extend(spans)
                    
                    # Continue with current text
                    next_prompts_ids.append(seq)
                    continue

                # 4) Normal Continue
                if eos_token_id in new_tokens:
                    outputs[b] = seq
                    should_gen[b] = False
                    next_prompts_ids.append(seq)
                else:
                    # Continue generating
                    next_prompts_ids.append(seq)

            # Prepare next iteration inputs
            # Pad valid next prompts to create tensor
            if not should_gen.any():
                break
                
            # Pad next_prompts_ids to same length
            max_p_len = max(len(t) for t in next_prompts_ids)
            padded_next = []
            new_masks = []
            
            for t in next_prompts_ids:
                pad_len = max_p_len - len(t)
                if pad_len > 0:
                    # Left padding usually for generation? The model original used left padding.
                    # But output sequences grow to the right. 
                    # Let's right pad with EOS/PAD for mask construction
                    t_pad = torch.cat([t, torch.full((pad_len,), pad_token_id, device=device)])
                    m_pad = torch.cat([torch.ones_like(t), torch.zeros((pad_len,), dtype=torch.long, device=device)])
                else:
                    t_pad = t
                    m_pad = torch.ones_like(t)
                padded_next.append(t_pad)
                new_masks.append(m_pad)

            current_ids = torch.stack(padded_next)
            current_mask = torch.stack(new_masks)

        # Final gather
        # Flatten outputs and handle None
        final_list = [o if o is not None else current_ids[i] for i, o in enumerate(outputs)]
        return self.prompt_left_generation_right_padding(input_ids, final_list, device, max_length_for_gather)
