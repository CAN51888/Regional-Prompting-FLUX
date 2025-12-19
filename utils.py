# utils/attn_recorder.py
import torch


class AttnRecorder:
    """
    记录 FLUX 在某个去噪 step & 某个 Double Stream Block 的 attention。

    target_step:   想记录的 timestep（0-based），比如 5 = 第 6 步
    target_block:  想记录的 Double Stream Block 下标（0-based），比如 16 = 第 17 个 block
    """

    def __init__(self, target_step: int = 5, target_block: int = 16, device: str = "cpu"):
        self.target_step = target_step
        self.target_block = target_block
        self.device = torch.device(device)

        self.current_step = -1
        self.current_block = -1

        # 第 6 步 + 第 17 block 的 attention
        self.cross_attn_list = []  # [B, H, L_text, N_img]
        self.self_attn_list = []   # [B, H, N_img, N_img]

    # ===== 在采样前 / 每次重新采样前调用 =====
    def reset(self):
        self.current_step = -1
        self.current_block = -1
        self.cross_attn_list = []
        self.self_attn_list = []

    # ===== 在去噪 loop 每一步开头设置 =====
    def set_step(self, step_idx: int):
        self.current_step = step_idx

    # ===== 在调用某个 block 前设置 =====
    def set_block(self, block_idx: int):
        self.current_block = block_idx

    # ===== 给 Processor 调用：保存注意力 =====
    @torch.no_grad()
    def maybe_record(self, attn_probs: torch.Tensor, text_len: int, img_len: int):
        """
        attn_probs: [B, H, Q_total, K_total]
        Q_total = text_len + img_len
        K_total = text_len + img_len（这里我们只关心 text->img, img->img）
        """
        if not (
            self.current_step == self.target_step
            and self.current_block == self.target_block
        ):
            return

        B, H, Q_total, K_total = attn_probs.shape
        assert Q_total == text_len + img_len
        assert K_total == text_len + img_len

        # text query 对 image key：cross-attn
        cross = attn_probs[:, :, :text_len, text_len:]  # [B, H, L_text, N_img]

        # image query 对 image key：self-attn (image-only)
        self_img = attn_probs[:, :, text_len:, text_len:]  # [B, H, N_img, N_img]

        self.cross_attn_list.append(cross.to(self.device).detach().cpu())
        self.self_attn_list.append(self_img.to(self.device).detach().cpu())
# utils/recording_double_processor.py
import torch
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb


class RecordingDoubleStreamProcessor:
    """
    用在 FLUX 的 Double Stream Block（pipe.transformer.transformer_blocks[i].attn.processor）

    功能：
      1. 完成原本的 attention 计算（保持行为不变）；
      2. 在指定 step & block 时，从 attn_probs 里抽出：
            - text -> img 的 cross-attn
            - img -> img 的 self-attn
         并交给 AttnRecorder.
    """

    def __init__(self, recorder, block_idx: int):
        self.recorder = recorder
        self.block_idx = block_idx

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "RecordingDoubleStreamProcessor requires PyTorch 2.0+. "
            )

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask = None,
        image_rotary_emb = None,
    ) -> torch.FloatTensor:
        """
        这里 attn 是 FLUX 的 cross/self 复合 attention 模块（带 to_q/to_k/to_v 等）
        hidden_states：image tokens
        encoder_hidden_states：text tokens
        """
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        # ===== sample projections (image branch) =====
        query = attn.to_q(hidden_states)
        key   = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # [B, H, N_img, D]
        key   = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # ===== context projections (text branch) =====
        text_len = encoder_hidden_states.shape[1]
        img_len  = hidden_states.shape[1]

        enc_q = attn.add_q_proj(encoder_hidden_states)
        enc_k = attn.add_k_proj(encoder_hidden_states)
        enc_v = attn.add_v_proj(encoder_hidden_states)

        enc_q = enc_q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        enc_k = enc_k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        enc_v = enc_v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_added_q is not None:
            enc_q = attn.norm_added_q(enc_q)
        if attn.norm_added_k is not None:
            enc_k = attn.norm_added_k(enc_k)

        # ===== concat text + image =====
        # 注意：FLUX 的实现是 [text, image] 的顺序
        q_all = torch.cat([enc_q, query], dim=2)  # [B, H, L_text+N_img, D]
        k_all = torch.cat([enc_k, key],   dim=2)
        v_all = torch.cat([enc_v, value], dim=2)

        if image_rotary_emb is not None:
            q_all = apply_rotary_emb(q_all, image_rotary_emb)
            k_all = apply_rotary_emb(k_all, image_rotary_emb)

        # ===== 手工算 scaled dot-product attention，拿到 attn_probs =====
        # q_all: [B, H, Q_total, D]
        # k_all: [B, H, K_total, D]
        scale = 1.0 / (head_dim ** 0.5)
        attn_scores = torch.matmul(q_all, k_all.transpose(-1, -2)) * scale  # [B, H, Q, K]

        if attention_mask is not None:
            # attention_mask 形状可以是 [B,1,1,K]，这里简单 broadcast
            attn_scores = attn_scores + attention_mask

        attn_probs = torch.softmax(attn_scores, dim=-1)  # [B, H, Q, K]

        # ===== 只在目标 step + 目标 block 时记录 =====
        # 告诉 recorder：当前是哪个 block
        self.recorder.set_block(self.block_idx)
        self.recorder.maybe_record(attn_probs, text_len=text_len, img_len=img_len)

        # ===== 正常输出 =====
        hidden_states_all = torch.matmul(attn_probs, v_all)  # [B, H, Q, D]
        hidden_states_all = hidden_states_all.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_all = hidden_states_all.to(q_all.dtype)

        # 拆回 text / image
        enc_out, img_out = (
            hidden_states_all[:, :text_len],
            hidden_states_all[:, text_len:],
        )

        # linear proj & dropout
        img_out = attn.to_out[0](img_out)
        img_out = attn.to_out[1](img_out)

        enc_out = attn.to_add_out(enc_out)

        if input_ndim == 4:
            img_out = img_out.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            enc_out = enc_out.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return img_out, enc_out


import torch
import numpy as np
from skimage.segmentation import slic
def build_activation_token_ids_from_lora_tags(tokenizer, activation_words):
    """
    只用 <lora:xxx> 作为激活 token。
    activation_words 的 key 视为 lora_name（如 'emma'、'harry'）。
    """
    activation_token_ids = {}

    for name in activation_words.keys():
        # 假设约定 tag 形式为 <lora:name>
        lora_tag = f"<lora:{name}>"

        # 1) 先尝试把 tag 当作一个 token id 直接转换
        tid = tokenizer.convert_tokens_to_ids(lora_tag)

        if tid is not None and tid != tokenizer.unk_token_id:
            # 成功：说明确实注册成了一个 special token
            activation_token_ids[name] = {tid}
            print(f"[LORA TAG] {name}: {lora_tag} -> id {tid}")
            continue

        # 2) 如果上面失败了（可能没注册成 special token），退回用 encode
        ids = tokenizer.encode(lora_tag, add_special_tokens=False)
        if len(ids) == 0:
            print(f"[WARN] No ids found for lora tag {lora_tag}, {name} will have no activation tokens.")
            activation_token_ids[name] = set()
        elif len(ids) == 1:
            activation_token_ids[name] = {ids[0]}
            print(f"[LORA TAG] {name}: {lora_tag} -> id {ids[0]} (via encode)")
        else:
            # 多个 id 的情况：有点怪，但我们可以先全部保留，或者拿第一个
            # 为了安全，这里全部保留，后面 S_dict 里就会用这些位置
            activation_token_ids[name] = set(ids)
            print(f"[WARN] Lora tag {lora_tag} encoded to multiple ids {ids}, using all of them.")

    print("Activation token ids (from lora tags only):", activation_token_ids)
    return activation_token_ids

def find_lora_tag_positions(text_ids_1, tokenizer, activation_words):
        """
        只使用每个 LoRA 对应的 <lora:xxx> tag（或 word_list 里第一个以 '<lora:' 开头的字符串）
        来确定文本里的激活 token 位置。
        返回：
            S_dict: {lora_name: [pos0, pos1, ...]}
        """
        text_ids = text_ids_1.tolist()
        S_dict = {}

        for name, word_list in activation_words.items():
            # 1) 找这个 LoRA 的 tag 文本
            lora_tag = None
            for w in word_list:
                if isinstance(w, str) and w.startswith("<lora:"):
                    lora_tag = w
                    break
            if lora_tag is None:
                # 如果你没在 word_list 里提供 tag，就按约定自己拼一个
                lora_tag = f"<lora:{name}>"

            # 2) encode tag，得到它在 token 空间的 id 序列
            tag_ids = tokenizer.encode(lora_tag, add_special_tokens=False)
            if len(tag_ids) == 0:
                print(f"[WARN] lora_tag '{lora_tag}' for '{name}' encodes to empty ids, skip.")
                S_dict[name] = []
                continue

            # 3) 在整个 text_ids 里找所有等于 tag_ids 的连续子串
            positions = []
            L = len(tag_ids)
            for i in range(0, len(text_ids) - L + 1):
                if text_ids[i : i + L] == tag_ids:
                    # 把整段 [i, i+L) 都算作该 LoRA 的激活位置
                    positions.extend(range(i, i + L))

            S_dict[name] = sorted(set(positions))
            print(f"[LORA TAG] {name}: tag='{lora_tag}', tag_ids={tag_ids}")
            print(f"[DEBUG] {name} positions (from lora tag only) =", S_dict[name])

        return S_dict

def compute_lora_masks_from_attn_single_layer(
    cross_attn: torch.Tensor,
    self_attn: torch.Tensor,
    text_ids: torch.LongTensor,
    tokenizer,
    activation_words: dict,
    image: np.ndarray,
    H_latent: int,
    W_latent: int,
    seed_ratio: float = 0.01,
    num_superpixels: int = 1000,
    bg_threshold: float =  0.03,
    device: torch.device = torch.device("cuda"),
):
    """
    专门用于：单个 step + 单个 Double Stream Block 的 attention。

    Args:
        cross_attn: [B, H_heads, L_text, N_img]  第6步、第17 block 的 cross-attn
        self_attn:  [B, H_heads, N_img, N_img]   同一步、同一层的 self-attn (image-image)
        text_ids:   [B, L_text]
        tokenizer:  文本 tokenizer，需支持 encode(..., add_special_tokens=False)
        activation_words:
            dict, 例如：
            {
                "emma": ["<lora:emma>", "emma"],
                "harry": ["<lora:harry>", "harry potter"],
            }
        image:      粗生成图，np.uint8 [H_img, W_img, 3]
        H_latent, W_latent:
            latent feature map 的高宽（N_img = H_latent * W_latent）
        seed_ratio: 用 top k% cross-attn 作为 seeds
        num_superpixels: SLIC 超像素数量
        bg_threshold:
            超像素在所有 LoRA 上的最大响应若低于该阈值，则判定为背景
        device:    输出 mask 放到哪个 device

    Returns:
        masks_dict: {lora_name: torch.FloatTensor[H_img, W_img]}，0/1 mask
    """
    # 只支持 batch=1，简单粗暴
    B = text_ids.shape[0]

    text_ids_1 = text_ids[0]  # [L_text]

    # ===== 1. 对 heads 做平均，得到单层 mean cross / self =====
    # cross_attn: [1, H, L_text, N_img] -> [L_text, N_img]
    cross_mean = cross_attn.mean(dim=1)[0]  # [L_text, N_img]  torch.Size([512, 1024])
    print("Cross mean shape:", cross_mean.shape)
    # self_attn:  [1, H, N_img, N_img] -> [N_img, N_img]
    self_mean = self_attn.mean(dim=1)[0]    # [N_img, N_img] torch.Size([1024, 1024])
    print("Self mean shape:", self_mean.shape)
    N_img = H_latent * W_latent
    assert cross_mean.shape[1] == N_img, "N_img mismatch with H_latent*W_latent"
    assert self_mean.shape[0] == N_img and self_mean.shape[1] == N_img

    # ===== 2. 预处理 activation words -> 对应的 token id 集合 =====

    S_dict = find_lora_tag_positions(text_ids_1, tokenizer, activation_words)
    print("prompt token ids:", text_ids_1)        
    # # ===== 2. 预处理 activation words -> 对应的 token id 集合 =====
    # activation_token_ids = {}
    # for name, word_list in activation_words.items():
    #     token_id_set = set()
    #     for w in word_list:
    #         # 你可以根据自己的 tokenizer 改成更严谨的 tokenization
    #         ids = tokenizer.encode(w, add_special_tokens=False)
    #         for tid in ids:
    #             token_id_set.add(tid)
    #     activation_token_ids[name] = token_id_set
    # print("Activation token ids:", activation_token_ids)
    # # 在 text_ids 序列中找到每个 LoRA 的 token index 集合 S_i
    # pad_id = tokenizer.pad_token_id
    # S_dict = {}
    # text_ids_list = text_ids_1.tolist()
    # for name, tid_set in activation_token_ids.items():
    #     positions = []
    #     for i, tid in enumerate(text_ids_list):
    #         if tid == pad_id:
    #             continue  # 跳过 padding
    #         if tid in tid_set:
    #             positions.append(i)
    #     print(f"[DEBUG] {name} tid_set =", tid_set)
    #     print(f"[DEBUG] {name} positions =", positions)
    #     if len(positions) == 0:
    #         print(f"[WARN] No activation tokens found for LoRA '{name}' in current prompt.")
    #     S_dict[name] = positions

    # ===== 3. 每个 LoRA: 聚合 cross-attn -> A_cross^(i) =====
    A_cross_dict = {}
    for name, positions in S_dict.items():
        if len(positions) == 0:
            continue
        # cross_mean: [L_text, N_img]
    
        A_cross_i = cross_mean[positions].mean(dim=0)  # [N_img]
        print(f"[DEBUG] {name} attn shape:", A_cross_i.shape)
        print(f"[DEBUG] {name} attn stats: min={A_cross_i.min().item()}, "
        f"max={A_cross_i.max().item()}, mean={A_cross_i.mean().item()}")
        A_cross_i = A_cross_i.reshape(H_latent, W_latent)
        # 归一化到 [0,1]
        A_cross_i = (A_cross_i - A_cross_i.min()) / (A_cross_i.max() - A_cross_i.min() + 1e-6)
        A_cross_dict[name] = A_cross_i

    # ===== 4. self-attn seed 扩散 -> Â^(i) =====
    A_hat_dict = {}
    for name, A_cross_i in A_cross_dict.items():
        flat = A_cross_i.flatten()  # [N_img]
        k = max(1, int(seed_ratio * flat.numel()))
        topk_vals, topk_idx = torch.topk(flat, k)
        seed_mask = torch.zeros(N_img, device=flat.device)
        seed_mask[topk_idx] = 1.0
        # self_mean: [N_img, N_img]
        # 在 A_hat_i 这行前面加：
        seed_mask = seed_mask.to(self_mean.dtype)
        A_hat_i = (seed_mask @ self_mean) / (seed_mask.sum() + 1e-6)  # [N_img]
        A_hat_i = A_hat_i.reshape(H_latent, W_latent)
        # 再归一化
        A_hat_i = (A_hat_i - A_hat_i.min()) / (A_hat_i.max() - A_hat_i.min() + 1e-6)
        A_hat_dict[name] = A_hat_i

    # ===== 5. 上采样到图像分辨率 =====
    H_img, W_img = image.shape[:2]
    A_hat_up_dict = {}
    for name, A_hat_i in A_hat_dict.items():
        A_hat_i_up = torch.nn.functional.interpolate(
            A_hat_i.unsqueeze(0).unsqueeze(0),  # [1,1,H_latent,W_latent]
            size=(H_img, W_img),
            mode="bilinear",
            align_corners=False,
        )[0, 0]  # [H_img, W_img]
        A_hat_up_dict[name] = A_hat_i_up

    # ===== 6. SLIC 超像素划分 =====
    # image: np.uint8 [H_img, W_img, 3]
    segments = slic(
        image,
        n_segments=num_superpixels,
        compactness=10.0,
        start_label=0,
    )
    segments = np.asarray(segments, dtype=np.int32)  # [H_img, W_img]
    num_segments = int(segments.max()) + 1

    lora_names = list(A_hat_up_dict.keys())
    K = len(lora_names)
    if K == 0:
        print("[WARN] No valid LoRA activation maps, return all-zero masks.")
        return {name: torch.zeros(H_img, W_img, device=device) for name in activation_words.keys()}

    # stack LoRA 响应：A_stack: [K, H_img, W_img]
    A_stack = torch.stack([A_hat_up_dict[name] for name in lora_names], dim=0)  # float

    scores = torch.zeros(num_segments, K, device=device)
    segments_t = torch.from_numpy(segments).to(device)

    # ===== 7. 超像素投票：每个 superpixel 属于哪个 LoRA =====
    for sp_id in range(num_segments):
        mask_sp = (segments_t == sp_id)  # [H_img, W_img]
        if mask_sp.sum() == 0:
            continue
        for i in range(K):
            vals = A_stack[i][mask_sp]
            scores[sp_id, i] = vals.mean()
  

    max_vals, owners = scores.max(dim=1)  # [num_segments]
    # owners[max_vals < bg_threshold] = -1  # 得分太低的设为背景
    print("max_vals stats: min", max_vals.min().item(),
      "max", max_vals.max().item(),
      "mean", max_vals.mean().item())
    # ===== 8. 构造每个 LoRA 的二值 mask =====
    masks_dict = {}
    for idx, name in enumerate(lora_names):
        Mi = (owners[segments_t] == idx).float()  # [H_img, W_img]
        masks_dict[name] = Mi.to(dtype=torch.bool,device=device)

    # 对 activation_words 里定义了，但由于没找到 token 之类的没生成的，也补 0 mask
    for name in activation_words.keys():
        if name not in masks_dict:
            masks_dict[name] = torch.zeros(H_img, W_img, device=device)

    return masks_dict
