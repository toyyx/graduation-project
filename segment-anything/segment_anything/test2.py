class LightASPP(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(in_ch, in_ch, 1),
            nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, dilation=3, padding=3),
                nn.BatchNorm2d(in_ch)),
            nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, dilation=5, padding=5),
                nn.BatchNorm2d(in_ch)),
            nn.AdaptiveAvgPool2d(1)
        ])
        self.fuse = nn.Conv2d(4*in_ch, in_ch, 1)
    def forward(self, x):
        return self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))

class LightSE(nn.Module):
    def __init__(self, ch, ratio=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch//ratio, 1),
            nn.ReLU(),
            nn.Conv2d(ch//ratio, ch, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.se(x)


import torch
from segment_anything import sam_model_registry

# 加载SAM模型
sam = sam_model_registry["vit_b"]()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sam.to(device)

# 加载预训练权重文件
checkpoint_path = "sam_vit_b_01ec64.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

# 提取提示编码器和掩码解码器的权重
prompt_encoder_weights = {k.replace('prompt_encoder.', ''): v for k, v in checkpoint.items() if k.startswith('prompt_encoder.')}
mask_decoder_weights = {k.replace('mask_decoder.', ''): v for k, v in checkpoint.items() if k.startswith('mask_decoder.')}

# 加载提示编码器和掩码解码器的权重
sam.prompt_encoder.load_state_dict(prompt_encoder_weights)
sam.mask_decoder.load_state_dict(mask_decoder_weights)

# 冻结提示编码器和掩码解码器的参数
for param in sam.prompt_encoder.parameters():
    param.requires_grad = False
for param in sam.mask_decoder.parameters():
    param.requires_grad = False

# 定义优化器，只优化图像编码器的参数
optimizer = torch.optim.Adam(sam.image_encoder.parameters(), lr=1e-4)

# 假设已经有数据加载器 data_loader 和损失函数 criterion
num_epochs = 10
for epoch in range(num_epochs):
    for images, gt_masks, prompts in data_loader:
        images = images.to(device)
        gt_masks = gt_masks.to(device)
        prompts = [prompt.to(device) for prompt in prompts]

        # 前向传播
        image_embeddings = sam.image_encoder(images)
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=None,
            boxes=None,
            masks=None
        )
        low_res_masks, iou_predictions = sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True
        )

        # 计算损失
        loss = criterion(low_res_masks, gt_masks)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')