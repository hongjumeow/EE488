"""
modification made on the basis of link:https://github.com/Xiaoccer/MobileFaceNet_Pytorch
"""
from torch import nn
import torch
import torch.nn.functional as F
import math
from torch.nn import Parameter
import torchaudio
from transformers import WavLMModel, WavLMConfig
from transformers import Wav2Vec2Model, Wav2Vec2Config


class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        #
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)

# original mobilefacenet setting
# Mobilefacenet_bottleneck_setting = [
#     # t, c , n ,s
#     [2, 64, 5, 2],
#     [4, 128, 1, 2],
#     [2, 128, 6, 1],
#     [4, 128, 1, 2],
#     [2, 128, 2, 1]
# ]

# Mobilenetv2_bottleneck_setting = [
#     # t, c, n, s
#     [1, 16, 1, 1],
#     [6, 24, 2, 2],
#     [6, 32, 3, 2],
#     [6, 64, 4, 2],
#     [6, 96, 3, 1],
#     [6, 160, 3, 2],
#     [6, 320, 1, 1],
# ]

# refer to DCASE2022 Task2 Top-1
# https://dcase.community/documents/challenge2022/technical_reports/DCASE2022_Liu_8_t2.pdf
Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 128, 2, 2],
    [4, 128, 2, 2],
    [4, 128, 2, 2],
]


class MobileFaceNet(nn.Module):
    def __init__(self,
                 num_class,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(MobileFaceNet, self).__init__()

        self.conv1 = ConvBlock(2, 64, 3, 2, 1)

        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(bottleneck_setting[-1][1], 512, 1, 1, 0)
        # 20(10), 4(2), 8(4)
        self.linear7 = ConvBlock(512, 512, (8, 20), 1, 0, dw=True, linear=True)
        # self.linear7 = ConvBlock(512, 512, (4, 10), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)

        self.fc_out = nn.Linear(128, num_class)
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x, label):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        feature = x.view(x.size(0), -1)
        out = self.fc_out(feature)
        return out, feature

######################################################################## originial code
# class TgramNet(nn.Module):
#     def __init__(self, num_layer=3, mel_bins=128, win_len=1024, hop_len=512):
#         super(TgramNet, self).__init__()
#         # if "center=True" of stft, padding = win_len / 2
#         self.conv_extrctor = nn.Conv1d(1, mel_bins, win_len, hop_len, win_len // 2, bias=False)
#         self.conv_encoder = nn.Sequential(
#             *[nn.Sequential(
#                 # 313(10) , 63(2), 126(4)
#                 nn.LayerNorm(313),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.Conv1d(mel_bins, mel_bins, 3, 1, 1, bias=False),
#             ) for idx in range(num_layer)])

#     def forward(self, x):
#         out = self.conv_extrctor(x)
#         out = self.conv_encoder(out)
#         return out


# class STgramMFN(nn.Module):
#     def __init__(self, num_classes,
#                  c_dim=128,
#                  win_len=1024,
#                  hop_len=512,
#                  bottleneck_setting=Mobilefacenet_bottleneck_setting,
#                  use_arcface=False, m=0.5, s=30, sub=1):
#         super(STgramMFN, self).__init__()
#         self.arcface = ArcMarginProduct(in_features=128, out_features=num_classes,
#                                         m=m, s=s, sub=sub) if use_arcface else use_arcface
#         self.tgramnet = TgramNet(mel_bins=c_dim, win_len=win_len, hop_len=hop_len)
#         self.mobilefacenet = MobileFaceNet(num_class=num_classes,
#                                            bottleneck_setting=bottleneck_setting)

#     def get_tgram(self, x_wav):
#         return self.tgramnet(x_wav)

#     def forward(self, x_wav, x_mel, label=None):
#         x_wav, x_mel = x_wav.unsqueeze(1), x_mel.unsqueeze(1)
#         x_t = self.tgramnet(x_wav).unsqueeze(1)
#         x = torch.cat((x_mel, x_t), dim=1)
#         out, feature = self.mobilefacenet(x, label)
#         if self.arcface:
#             out = self.arcface(feature, label)
#         return out, feature
######################################################################## originial code


class WavLMWrapper(nn.Module):
    def __init__(self, mel_bins=128, wavlm_model_name='microsoft/wavlm-base'):
        super(WavLMWrapper, self).__init__()
        self.wavlm = WavLMModel.from_pretrained(wavlm_model_name)
        self.proj = nn.Linear(self.wavlm.config.hidden_size, mel_bins)

    def forward(self, x):
        # x: (B, 1, T)
        x = x.squeeze(1)  # -> (B, T)
        outputs = self.wavlm(x, return_dict=True)
        hidden_states = outputs.last_hidden_state  # (B, T', H)
        projected = self.proj(hidden_states)       # -> (B, T', mel_bins)
        out = projected.permute(0, 2, 1)           # -> (B, mel_bins, N)
        out = out.unsqueeze(1)                     # -> (B, 1, mel_bins, N)
        return out

class Wav2Vec2FeatureWrapper(nn.Module):
    def __init__(self, mel_bins=128, model_name="facebook/wav2vec2-base"):
        super(Wav2Vec2FeatureWrapper, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        # Only use the feature extractor (CNN part)
        self.feature_extractor = self.wav2vec2.feature_extractor
        self.proj = nn.Linear(self.feature_extractor.conv_layers[-1].conv.out_channels, mel_bins)

    def forward(self, x, target_time_dim=None):
        # x: (B, 1, T)
        x = x.squeeze(1)  # → (B, T)
        features = self.feature_extractor(x)  # → (B, feature_dim, time')
        features = features.transpose(1, 2)   # → (B, time', feature_dim)
        projected = self.proj(features)       # → (B, time', mel_bins)
        out = projected.permute(0, 2, 1).unsqueeze(1)  # → (B, 1, mel_bins, time')

        return out

class SelfAttentionFrequencyPattern(nn.Module):
    def __init__(self, mel_bins=128, num_heads=4, num_segments=6, segment_len=53):
        super().__init__()
        self.mel_bins = mel_bins
        self.num_heads = num_heads
        self.num_segments = num_segments
        self.segment_len = segment_len

        self.attn = nn.MultiheadAttention(embed_dim=segment_len, num_heads=1, batch_first=True)
        self.norm = nn.LayerNorm(segment_len)

    def forward(self, x):
        # x: (B, 1, M, N)
        x = x.squeeze(1)  # (B, M, N)
        B, M, N = x.shape

        stride = (N - self.segment_len) // (self.num_segments - 1)  # to cover full N with overlap
        segments = []

        for i in range(self.num_segments):
            start = i * stride
            end = start + self.segment_len
            if end > N:
                start = N - self.segment_len
                end = N
            segment = x[:, :, start:end]  # (B, M, segment_len)
            res = segment
            attn_out, _ = self.attn(segment, segment, segment)  # attention over frequency dim
            segment = self.norm(attn_out + res)
            segments.append(segment)

        x_attn = torch.cat(segments, dim=-1)  # (B, M, N')
        return x_attn.unsqueeze(1)  # (B, 1, M, N')



class STgramMFN(nn.Module):
    def __init__(self, num_classes,
                 c_dim=128,
                 use_arcface=False, m=0.5, s=30, sub=1,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(STgramMFN, self).__init__()
        self.arcface = ArcMarginProduct(in_features=128, out_features=num_classes,
                                        m=m, s=s, sub=sub) if use_arcface else use_arcface
        self.feature_wrapper = Wav2Vec2FeatureWrapper(mel_bins=c_dim)
        self.mobilefacenet = MobileFaceNet(num_class=num_classes,
                                           bottleneck_setting=bottleneck_setting)
        self.self_attn = SelfAttentionFrequencyPattern()
    def get_tgram(self, x_wav, target_time_dim):
        return self.feature_wrapper(x_wav, target_time_dim=target_time_dim)

    def forward(self, x_wav, x_mel, label=None):
        x_wav, x_mel = x_wav.unsqueeze(1), x_mel.unsqueeze(1)
        x_t = self.feature_wrapper(x_wav, target_time_dim=x_mel.shape[-1])  # (B, 1, mel_bins, N)
        x_mel = self.self_attn(x_mel)
        x_t = x_t.squeeze(1)
        
        # Interpolate x_t's time axis to match x_mel
        if x_t.shape[-1] != x_mel.shape[-1]:
            x_t =  b(x_t, size=x_mel.shape[-1], mode='linear', align_corners=False)
            # using adpaptive pooling 

        x_t = x_t.unsqueeze(1)
        x = torch.cat((x_mel, x_t), dim=1)  # (B, 2, mel_bins, N)
        out, feature = self.mobilefacenet(x, label)
        if self.arcface:
            out = self.arcface(feature, label)
        return out, feature


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=200, s=32.0, m=0.50, sub=1, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.sub = sub
        self.weight = Parameter(torch.Tensor(out_features * sub, in_features))
        nn.init.xavier_uniform_(self.weight)
        # init.kaiming_uniform_()
        # self.weight.data.normal_(std=0.001)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if self.sub > 1:
            cosine = cosine.view(-1, self.out_features, self.sub)
            cosine, _ = torch.max(cosine, dim=2)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=x.device)
        # print(x.device, label.device, one_hot.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return output


if __name__ == '__main__':
    net = STgramMFN(num_classes=10)
    x_wav = torch.randn((2, 16000*2))

