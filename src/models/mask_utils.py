import torch
from typing import List, Tuple

def apply_masked_spans(
    current_mask: torch.LongTensor,
    masked_spans_per_sample: list[ list[tuple[int,int,int]] ],
) -> torch.LongTensor:
    """
    根据 masked_spans_per_sample 动态屏蔽 attention mask。

    Args:
        current_mask (torch.LongTensor): 当前 attention mask, shape (B, T)
        masked_spans_per_sample (List[List[Tuple[int,int,int]]]): 
            每个 batch 的 span 三元组列表 (prev_action_start, prev_action_end, backtrack_end)

    Returns:
        torch.LongTensor: 更新后的 attention mask
    """
    batch_size, seq_len = current_mask.shape
    new_mask = current_mask.clone()

    for b in range(batch_size):
        for span in masked_spans_per_sample[b]:
            prev_start, prev_end, backtrack_end = span
            # 安全检查，防止越界
            # Note: The original code used prev_end strictly, potentially keeping one token if not careful.
            # Assuming standard python slicing [start:end).
            prev_start = max(0, prev_start)
            prev_end = min(prev_end, seq_len)
            
            if prev_start < prev_end:
                new_mask[b, prev_start:prev_end] = 0  # 屏蔽前一个 action

    return new_mask

def get_masked_spans_from_text(full_seq: torch.Tensor, tokenizer) -> List[Tuple[int, int, int]]:
    """
    解析文本获取需要屏蔽的 span。
    
    Returns:
        List of tuples: (start, end, backtrack_end_pos)
    """
    # 简化的实现，寻找最近一次出现的 <backtrack> 或 <summary>
    # 并假设它屏蔽了上一个 <search>...</search> 或者 <reasoning>...</reasoning>
    # 注意：这是一个简化的逻辑，完全复刻原逻辑需要 src.neuron.action_utils
    
    text = tokenizer.decode(full_seq, skip_special_tokens=False)
    
    # 这里为了保证代码可运行，先返回空列表或者简单的查找逻辑
    # 如果项目中没有 action_utils，我们很难精确知道要屏蔽什么
    # 但为了防止报错，我们返回空列表，或者实现一个简单的基于 tag 的逻辑
    
    # 示例逻辑：查找最后的 </backtrack>
    if "</backtrack>" in text:
        # 假设 <backtrack> 是为了修正前面的某个思考
        # 简单起见，我们不返回实际屏蔽区间（因为这需要复杂的解析），而是依赖调用者逻辑或者留空
        pass

    return [] # Placeholder

def expand_to_causal_mask_backtrack(current_mask: torch.Tensor, masked_spans_per_sample, dtype=torch.float32):
    """
    将 2D padding mask (B,L) 转换为 4D causal attention mask (B,1,L,L)
    并根据 masked_spans_per_sample 动态屏蔽 attention。
    masked_spans_per_sample: List[List[Tuple(prev_start, prev_end, backtrack_end)]]
    """
    B, T = current_mask.shape
    min_dtype = torch.finfo(dtype).min

    # 基础下三角 causal mask
    causal_mask = torch.full((T, T), fill_value=min_dtype, dtype=dtype, device=current_mask.device)
    causal_mask = torch.triu(causal_mask, diagonal=1)  # 对角线及以下为0
    causal_mask = causal_mask[None, None, :, :].expand(B, 1, -1, -1).clone()

    # padding mask
    if current_mask is not None:
        pad_mask_cond = current_mask[:, None, None, :] == 0  # (B,1,1,T)
        causal_mask = causal_mask.masked_fill(pad_mask_cond, min_dtype)

    # 动态屏蔽 span
    for b in range(B):
        if b < len(masked_spans_per_sample):
            for span in masked_spans_per_sample[b]:
                prev_start, prev_end, backtrack_end = span
                # 越界保护
                prev_start = max(0, min(prev_start, T-1))
                prev_end = max(0, min(prev_end, T))
                backtrack_end = max(0, min(backtrack_end, T-1))
                
                # 双向屏蔽：Backtrack 后的内容 <-> 被 Backtrack 的内容
                if prev_start < prev_end and backtrack_end < T:
                     # 将 [prev_start, prev_end] 这一段 与 [backtrack_end+1, end] 这一段 互相屏蔽
                    causal_mask[b, 0, prev_start:prev_end, backtrack_end+1:] = min_dtype
                    causal_mask[b, 0, backtrack_end+1:, prev_start:prev_end] = min_dtype
    return causal_mask

def expand_to_causal_mask_parallel(current_mask: torch.Tensor, masked_parallel_spans_per_sample, dtype=torch.float32):
    """
    将 2D padding mask (B,L) 转换为 4D causal attention mask (B,1,L,L)
    并根据 masked_parallel_spans_per_sample 屏蔽不同 span 之间的 attention。

    masked_parallel_spans_per_sample: List[batch] -> List[rollout] -> List[Tuple(start,end)]
    """
    B, T = current_mask.shape
    min_dtype = torch.finfo(dtype).min

    # 基础 causal mask
    causal_mask = torch.full((T, T), fill_value=min_dtype, dtype=dtype, device=current_mask.device)
    causal_mask = torch.triu(causal_mask, diagonal=1)  # 对角线以下为 0
    causal_mask = causal_mask[None, None, :, :].expand(B, 1, -1, -1).clone()

    # padding mask
    if current_mask is not None:
        pad_mask_cond = current_mask[:, None, None, :] == 0
        causal_mask = causal_mask.masked_fill(pad_mask_cond, min_dtype)

    # 遍历 batch
    for b in range(B):
        if b < len(masked_parallel_spans_per_sample):
            batch_spans = masked_parallel_spans_per_sample[b]  # List of rollout
            for rollout_spans in batch_spans:  # List of (start,end)
                # 两两屏蔽
                for i, span_i in enumerate(rollout_spans):
                    start_i, end_i = span_i
                    for j, span_j in enumerate(rollout_spans):
                        if i == j:
                            continue  # 不屏蔽自己
                        start_j, end_j = span_j
                        
                        start_i = max(0, min(start_i, T))
                        end_i = max(0, min(end_i, T))
                        start_j = max(0, min(start_j, T))
                        end_j = max(0, min(end_j, T))

                        # query in span_i 屏蔽 key in span_j
                        # 注意切片范围
                        causal_mask[b, 0, start_i:end_i, start_j:end_j] = min_dtype
                        # causal_mask[b, 0, start_j:end_j, start_i:end_i] = min_dtype # 对称屏蔽
    return causal_mask
