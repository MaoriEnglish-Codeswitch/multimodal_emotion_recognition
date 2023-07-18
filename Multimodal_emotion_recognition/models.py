import torch
import torch.nn as nn
import torch.nn.functional as F


class TVAModel_Cross(nn.Module):  # trimodal cross-attn model
    def __init__(self, params):
        super(TVAModel_Cross, self).__init__()
        rnn = nn.LSTM
        self.text_conv = nn.Conv1d(in_channels=1, out_channels=50,kernel_size=1)
        self.text_encoder = rnn(input_size=768, hidden_size=params.txt_rnnsize,
                                num_layers=params.txt_rnnnum, dropout=params.txt_rnndp, bidirectional=params.rnndir,
                                batch_first=True)

        self.video_conv = nn.Conv1d(in_channels=32, out_channels=50, kernel_size=1)
        self.video_encoder = rnn(input_size=2208, hidden_size=params.vid_rnnsize,
                                 num_layers=params.vid_rnnnum, dropout=params.vid_rnndp, bidirectional=params.rnndir,
                                 batch_first=True)

        self.audio_conv = nn.Conv1d(in_channels=1, out_channels=50, kernel_size=1)
        self.audio_encoder = rnn(input_size=2048, hidden_size=params.aud_rnnsize,
                                 num_layers=params.aud_rnnnum, dropout=params.aud_rnndp, bidirectional=params.rnndir,
                                 batch_first=True)

        if params.rnndir:
            self.mha_v_t = nn.MultiheadAttention(embed_dim=2 * params.rnnsize, num_heads=params.vt_nh,
                                                 dropout=params.vt_dp, batch_first=True)
            self.mha_a_t = nn.MultiheadAttention(embed_dim=2 * params.rnnsize, num_heads=params.at_nh,
                                                 dropout=params.at_dp, batch_first=True)
            self.mha_t_v = nn.MultiheadAttention(embed_dim=2 * params.rnnsize, num_heads=params.tv_nh,
                                                 dropout=params.tv_dp, batch_first=True)
            self.mha_a_v = nn.MultiheadAttention(embed_dim=2 * params.rnnsize, num_heads=params.av_nh,
                                                 dropout=params.av_dp, batch_first=True)
            self.mha_t_a = nn.MultiheadAttention(embed_dim=2 * params.rnnsize, num_heads=params.ta_nh,
                                                 dropout=params.ta_dp, batch_first=True)
            self.mha_v_a = nn.MultiheadAttention(embed_dim=2 * params.rnnsize, num_heads=params.va_nh,
                                                 dropout=params.va_dp, batch_first=True)


        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.se_fc = nn.Sequential(
            nn.Linear(params.se_block_channels, params.se_block_channels*3 , bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(params.se_block_channels*3, params.se_block_channels, bias=False),
            nn.Sigmoid())


        self.concat_linear = nn.Linear(in_features=2 * 2 * params.rnnsize, out_features=params.rnnsize)
        self.classifier = nn.Linear(in_features=params.rnnsize, out_features=params.output_dim)

    def forward(self, x_txt, x_vid, x_aud):
        # text branch
        x_txt = F.dropout(x_txt, p=0.5, training=self.training)
        a,b=x_txt.size()
        x_txt=torch.reshape(x_txt, (a,1,b))
        x_txt=self.text_conv(x_txt)
        x_txt, h = self.text_encoder(x_txt)

        # video branch
        x_vid = F.dropout(x_vid, p=0.5, training=self.training)
        x_vid = self.video_conv(x_vid)
        x_vid, h = self.video_encoder(x_vid)

        # audio branch
        x_aud = F.dropout(x_aud, p=0.5, training=self.training)
        c,d=x_aud.size()
        x_aud=torch.reshape(x_aud, (c,1,d))
        x_aud = self.audio_conv(x_aud)
        x_aud, h = self.audio_encoder(x_aud)

        ##### V,A -> T
        # video to text
        x_v2t, _ = self.mha_v_t(x_txt, x_vid, x_vid)
        x_v2t = torch.mean(x_v2t, dim=1)
        # audio to text
        x_a2t, _ = self.mha_a_t(x_txt, x_aud, x_aud)
        x_a2t = torch.mean(x_a2t, dim=1)
        # addition
        ####### T,A -> V
        # text to video
        x_t2v, _ = self.mha_t_v(x_vid, x_txt, x_txt)
        x_t2v = torch.mean(x_t2v, dim=1)
        # audio to video
        x_a2v, _ = self.mha_a_v(x_vid, x_aud, x_aud)
        x_a2v = torch.mean(x_a2v, dim=1)
        # addition
        ####### T,V -> A
        # text to audio
        x_t2a, _ = self.mha_t_a(x_aud, x_txt, x_txt)
        x_t2a = torch.mean(x_t2a, dim=1)
        # video to audio
        x_v2a, _ = self.mha_v_a(x_aud, x_vid, x_vid)
        x_v2a = torch.mean(x_v2a, dim=1)

        x_tva2 = torch.stack((x_a2t, x_v2t, x_t2v, x_a2v, x_t2a, x_v2a), dim=1)

        # SE block
        batch_size, channels, features = x_tva2.size()
        # Average along each channel - squeeze
        squeeze = self.avg_pool(x_tva2)
        squeeze_tensor = squeeze.view(batch_size, channels)

        # excitation - 2 fully conneced layers (Relu and Sigmoid) - get the weight vector
        fc_out = self.se_fc(squeeze_tensor)

        # original input multiply the weight vector
        weight_matrix = fc_out.view(batch_size, channels, 1)
        x_tva2 = x_tva2 * weight_matrix.expand_as(x_tva2)


        x_tva2_mean, x_tva2_std = torch.std_mean(x_tva2, dim=1)
        x_tva2 = torch.cat((x_tva2_mean, x_tva2_std), dim=1)
        x_tva = x_tva2
        x_tva = self.concat_linear(x_tva)
        y = self.classifier(x_tva)
        return y, x_tva

