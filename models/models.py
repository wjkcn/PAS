
import os
import timm
from timm.models.layers import DropPath, trunc_normal_
try:
    from pointnet2_ops import pointnet2_utils
except ImportError:
    from pointnet2_ops_shim import furthest_point_sample, gather_operation
    import types
    pointnet2_utils = types.SimpleNamespace(
        furthest_point_sample=furthest_point_sample,
        gather_operation=gather_operation)
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as checkpoint_seq


class KNN(nn.Module):
    def __init__(self, k, transpose_mode=False):
        super(KNN, self).__init__()
        self.k = k
        self.transpose_mode = transpose_mode

    def forward(self, ref, query):
        if self.transpose_mode:
            ref = ref.transpose(1, 2)
            query = query.transpose(1, 2)
        dist_matrix = torch.cdist(query.transpose(-1, -2), ref.transpose(-1, -2))
        dist, idx = torch.topk(dist_matrix, k=self.k, dim=-1, largest=False)
        return dist, idx


def _interpolate_pos_embed(ckpt_pos_embed, model_pos_embed):
    import torch.nn.functional as F
    if ckpt_pos_embed.shape == model_pos_embed.shape:
        return ckpt_pos_embed
    ckpt_cls = ckpt_pos_embed[:, :1, :]
    ckpt_patch = ckpt_pos_embed[:, 1:, :]
    model_cls = model_pos_embed[:, :1, :]
    model_patch = model_pos_embed[:, 1:, :]
    N_tgt = model_patch.shape[1]
    src_size = int(ckpt_patch.shape[1] ** 0.5)
    tgt_size = int(N_tgt ** 0.5)
    ckpt_patch_grid = ckpt_patch.reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2)
    interp_grid = F.interpolate(ckpt_patch_grid, size=(tgt_size, tgt_size),
                                 mode='bicubic', align_corners=False)
    interp_patch = interp_grid.permute(0, 2, 3, 1).reshape(1, N_tgt, -1)
    return torch.cat([ckpt_cls, interp_patch], dim=1)


class Model(torch.nn.Module):

    def __init__(self, device, rgb_backbone_name='vit_base_patch14_dinov2', out_indices=None, checkpoint_path='',
                 pool_last=False, xyz_backbone_name='Point_MAE', group_size=128, num_group=1024,
                 load_xyz=True):
        super().__init__()
        self.device = device
        kwargs = {'features_only': True if out_indices else False}
        if out_indices:
            kwargs.update({'out_indices': out_indices})

        ## --- 1. RGB Backbone (DINOv2) ---
        print(f"Initializing 2D backbone: {rgb_backbone_name}")
        use_pretrained = 'dinov2' in rgb_backbone_name
        if use_pretrained:
            kwargs['img_size'] = 224

        self.rgb_backbone = timm.create_model(model_name=rgb_backbone_name, pretrained=False, **kwargs)
        loaded = False

        if use_pretrained:
            dino_paths = [
                "checkpoints/dinov2_vitb14_pretrain.safetensors",
                "checkpoints/dinov2_vitb14_pretrain.pth",
            ]
            for dino_local_path in dino_paths:
                if os.path.exists(dino_local_path):
                    print(f"Loading local DINOv2 weights: {dino_local_path}")
                    try:
                        try:
                            from safetensors.torch import load_file
                            state_dict = load_file(dino_local_path)
                        except Exception:
                            state_dict = torch.load(dino_local_path, map_location='cpu', weights_only=False)
                        if 'backbone' in state_dict:
                            state_dict = state_dict['backbone']
                        if 'pos_embed' in state_dict:
                            model_pe = self.rgb_backbone.pos_embed
                            ckpt_pe = state_dict['pos_embed']
                            if model_pe.shape != ckpt_pe.shape:
                                print(f"Interpolating pos_embed: {ckpt_pe.shape} -> {model_pe.shape}")
                                state_dict['pos_embed'] = _interpolate_pos_embed(ckpt_pe, model_pe)
                        self.rgb_backbone.load_state_dict(state_dict, strict=True)
                        del state_dict
                        loaded = True
                        break
                    except Exception as e:
                        print(f"Failed loading {dino_local_path}: {e}")
            if not loaded:
                print("WARNING: DINOv2 weights not found locally, using random init")

        ## --- 2. XYZ Backbone (Point-MAE) ---
        if load_xyz and xyz_backbone_name == 'Point_MAE':
            print(f"Initializing 3D backbone: {xyz_backbone_name}")
            self.xyz_backbone = PointTransformer(
                group_size=group_size,
                num_group=num_group
            )
            mae_path = "checkpoints/pointmae_pretrain.pth"
            if os.path.exists(mae_path):
                print(f"Loading Point-MAE weights: {mae_path}")
                try:
                    mae_state_dict = torch.load(mae_path, map_location='cpu', weights_only=False)
                    if 'base_model' in mae_state_dict:
                        mae_state_dict = mae_state_dict['base_model']
                    elif 'model' in mae_state_dict:
                        mae_state_dict = mae_state_dict['model']
                    self.xyz_backbone.load_state_dict(mae_state_dict, strict=False)
                    del mae_state_dict
                except Exception as e:
                    print(f"Point-MAE loading failed: {e}, using random init")
            else:
                print(f"WARNING: Point-MAE weights not found: {mae_path}")
        elif not load_xyz:
            self.xyz_backbone = None
            print("Point-MAE not loaded (load_xyz=False, using external 3D backbone)")

    def forward_rgb_features(self, x):
        x = self.rgb_backbone.patch_embed(x)
        x = self.rgb_backbone._pos_embed(x)
        x = self.rgb_backbone.norm_pre(x)
        if hasattr(self.rgb_backbone, 'grad_checkpointing') and self.rgb_backbone.grad_checkpointing:
            x = checkpoint_seq(self.rgb_backbone.blocks, x)
        else:
            x = self.rgb_backbone.blocks(x)
        x = self.rgb_backbone.norm(x)
        num_patches = x.shape[1] - 1
        grid_size = int(num_patches ** 0.5)
        feat = x[:, 1:].permute(0, 2, 1).reshape(1, -1, grid_size, grid_size)
        return feat

    def forward(self, rgb, xyz, anomaly_scores=None):
        rgb_features = self.forward_rgb_features(rgb)
        xyz_features, center, ori_idx, center_idx = self.xyz_backbone(xyz, anomaly_scores)
        return rgb_features, xyz_features, center, ori_idx, center_idx


def fps(data, number):
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx.int()).transpose(1, 2).contiguous()
    return fps_data, fps_idx


def pas_fps(xyz, number, anomaly_scores, tau=0.6, pool_mult=3):
    B, N, _ = xyz.shape
    device = xyz.device
    n_feat = int(number * tau)
    n_rand = int(number * 0.1)
    n_fps = number - n_feat - n_rand
    if N <= number:
        idx = torch.arange(N, device=device).unsqueeze(0).expand(B, N).int()
        c = pointnet2_utils.gather_operation(xyz.transpose(1, 2).contiguous(), idx).transpose(1, 2).contiguous()
        return c, idx.long()
    final_indices = []
    for b in range(B):
        available = torch.ones(N, dtype=torch.bool, device=device)
        fps_idx = pointnet2_utils.furthest_point_sample(xyz[b:b + 1], n_fps).squeeze(0)
        available[fps_idx] = False
        scores = anomaly_scores[b].clone()
        scores[~available] = -float('inf')
        if pool_mult > 1:
            pool_size = min(n_feat * pool_mult, int(available.sum().item()))
            if pool_size > n_feat:
                _, candidates = torch.topk(scores, pool_size)
                cand_xyz = xyz[b:b + 1, candidates, :]
                local_fps_idx = pointnet2_utils.furthest_point_sample(cand_xyz, n_feat).squeeze(0)
                feat_idx = candidates[local_fps_idx]
            else:
                _, feat_idx = torch.topk(scores, n_feat)
        else:
            _, feat_idx = torch.topk(scores, n_feat)
        available[feat_idx] = False
        remaining = torch.nonzero(available).squeeze(-1)
        rand_idx = remaining[torch.randperm(len(remaining), device=device)[:n_rand]]
        final_indices.append(torch.cat([fps_idx, feat_idx, rand_idx], dim=0))
    final_idx = torch.stack(final_indices).int()
    centers = pointnet2_utils.gather_operation(xyz.transpose(1, 2).contiguous(), final_idx).transpose(1, 2).contiguous()
    return centers, final_idx.long()


def random_sampling(xyz, number):
    B, N, _ = xyz.shape
    device = xyz.device
    if N <= number:
        idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1).int()
        c = pointnet2_utils.gather_operation(xyz.transpose(1, 2).contiguous(), idx).transpose(1, 2).contiguous()
        return c, idx.long()
    idx = torch.stack([torch.randperm(N, device=device)[:number] for _ in range(B)]).int()
    centers = pointnet2_utils.gather_operation(xyz.transpose(1, 2).contiguous(), idx).transpose(1, 2).contiguous()
    return centers, idx.long()


class Group(nn.Module):
    def __init__(self, num_group, group_size, pas_tau=0.6, pas_pool_mult=3):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.pas_tau = pas_tau
        self.pas_pool_mult = pas_pool_mult
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, anomaly_scores=None):
        batch_size, num_points, _ = xyz.shape
        mode = getattr(self, '_sampling_mode', None)
        if mode is None:
            mode = 'pas' if anomaly_scores is not None else 'fps'
        if mode == 'fps':
            center, center_idx = fps(xyz.contiguous(), self.num_group)
        elif mode == 'pas':
            center, center_idx = pas_fps(xyz.contiguous(), self.num_group, anomaly_scores, tau=self.pas_tau, pool_mult=self.pas_pool_mult)
        else:
            center, center_idx = fps(xyz.contiguous(), self.num_group)
        _, idx = self.knn(xyz, center)
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        ori_idx = idx
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.reshape(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.reshape(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center, ori_idx, center_idx


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x); x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate)
            for i in range(depth)])

    def forward(self, x, pos):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list


class PointTransformer(nn.Module):
    def __init__(self, group_size=128, num_group=1024, encoder_dims=384):
        super().__init__()
        self.trans_dim = 384
        self.depth = 12
        self.drop_path_rate = 0.1
        self.num_heads = 6
        self.group_size = group_size
        self.num_group = num_group
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder_dims = encoder_dims
        if self.encoder_dims != self.trans_dim:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim))
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(embed_dim=self.trans_dim, depth=self.depth, drop_path_rate=dpr, num_heads=self.num_heads)
        self.norm = nn.LayerNorm(self.trans_dim)

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
            for k in list(base_ckpt.keys()):
                if k.stapaswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.stapaswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]
            self.load_state_dict(base_ckpt, strict=False)

    def load_model_from_pb_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.stapaswith('transformer_q') and not k.stapaswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.stapaswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]
        self.load_state_dict(base_ckpt, strict=False)

    def forward(self, pts, anomaly_scores=None):
        if self.encoder_dims != self.trans_dim:
            B, C, N = pts.shape
            pts = pts.transpose(-1, -2)
            neighborhood, center, ori_idx, center_idx = self.group_divider(pts, anomaly_scores)
            group_input_tokens = self.encoder(neighborhood)
            group_input_tokens = self.reduce_dim(group_input_tokens)
            cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
            cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
            pos = self.pos_embed(center)
            x = torch.cat((cls_tokens, group_input_tokens), dim=1)
            pos = torch.cat((cls_pos, pos), dim=1)
            feature_list = self.blocks(x, pos)
            feature_list = [self.norm(x)[:, 1:].transpose(-1, -2).contiguous() for x in feature_list]
            x = torch.cat((feature_list[0], feature_list[1], feature_list[2]), dim=1)
            return x, center, ori_idx, center_idx
        else:
            B, C, N = pts.shape
            pts = pts.transpose(-1, -2)
            neighborhood, center, ori_idx, center_idx = self.group_divider(pts, anomaly_scores)
            group_input_tokens = self.encoder(neighborhood)
            pos = self.pos_embed(center)
            x = group_input_tokens
            feature_list = self.blocks(x, pos)
            feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]
            x = torch.cat((feature_list[0], feature_list[1], feature_list[2]), dim=1)
            return x, center, ori_idx, center_idx
