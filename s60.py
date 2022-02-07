def S60(pretrained=False, **kwargs):
    model = PatchConvnet(
        patch_size=16, embed_dim=384, depth=60, num_heads=1, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        Patch_layer=ConvStem,
        Attention_block = Conv_blocks_se,
        depth_token_only=1,
        mlp_ratio_clstk=3.0,**kwargs)

    return model