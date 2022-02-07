class PatchConvnet(nn.Module):

    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=768, depth=12,
                 num_heads=1, qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers = Layer_scale_init_Block,
                 block_layers_token = Layer_scale_init_Block_only_token,
                 Patch_layer=ConvStem,act_layer=nn.GELU,
                 Attention_block = Conv_blocks_se ,
                dpr_constant=True,init_scale=1e-4,
                Attention_block_token_only=Learned_Aggregation_Layer,
                Mlp_block_token_only= Mlp,
                depth_token_only=1,
                mlp_ratio_clstk = 3.0,
                multiclass=False):
        super().__init__()

        self.multiclass = multiclass
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = Patch_layer(     # convolutional steam
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        
        if not self.multiclass:             # monoclass --> crea un solo token per tutte le classi matrice di 0 di 768dim
            self.cls_token = nn.Parameter(torch.zeros(1, 1, int(embed_dim)))
        
        dpr = [drop_path_rate for i in range(depth)]        # ARRAY DI 0 DI LEN = DEPTH = 12
            
        self.blocks = nn.ModuleList([                       # SINGOLI BLOCCHI DENTRO COLUMN TRUNCK DI NUM = DEPTH
            block_layers(
                dim=embed_dim, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,init_values=init_scale)
            for i in range(depth)])
                    
        
        self.blocks_token_only = nn.ModuleList([            # SINGOLI BLOCCHI DENTRO COLUMN TRUNCK DI NUM = DEPTH + TOKEN
            block_layers_token(
                dim=int(embed_dim), num_heads=num_heads, mlp_ratio=mlp_ratio_clstk,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.0, norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block_token_only,
                Mlp_block=Mlp_block_token_only,init_values=init_scale)
            for i in range(depth_token_only)])
        
        self.norm = norm_layer(int(embed_dim))              # NORMALIZATION PART
        
        self.total_len = depth_token_only+depth             # PROFONDIT TOTALE DELLA RETE --> 1+12 = 13
        
        self.feature_info = [dict(num_chs=int(embed_dim ), reduction=0, module='head')]         # [{'num_chs': 768, 'reduction': 0, 'module': 'head'}]  DICTIONARY
        if not self.multiclass:
            self.head = nn.Linear(int(embed_dim), num_classes) if num_classes > 0 else nn.Identity()        # PAPER DA INFO SUL PERCHè USIAMO QUESTO --> IN PRIMA PAGINA
        


        trunc_normal_(self.cls_token, std=.02)              # FUNCTION IMPORTED FROM TIMM.LAYERS
        self.apply(self._init_weights)
        ################################################ FINE INIT  #########################################################

    def _init_weights(self, m):         ## DA CAPIRE
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head
    
    def get_num_layers(self):
        return len(self.blocks)
    

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        for i , blk in enumerate(self.blocks):
            x  = blk(x)

        for i , blk in enumerate(self.blocks_token_only):
            cls_tokens = blk(x,cls_tokens)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.norm(x)
        
        if not self.multiclass:
            return x[:, 0]
        else:
            return x[:, :self.num_classes].reshape(B,self.num_classes,-1)

    def forward(self, x):
        B = x.shape[0]
        x  = self.forward_features(x)
        if not self.multiclass:
            x = self.head(x)
            return x
        else:
            all_results = []
            for i in range(self.num_classes):
                all_results.append(self.head[i](x[:,i]))
            return torch.cat(all_results,dim=1).reshape(B,self.num_classes)
        