class TorusConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.edge_size = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = torch.cat([x[:, :, :, -self.edge_size[1]:], x, x[:, :, :, :self.edge_size[1]]], dim=3)
        h = torch.cat([h[:, :, -self.edge_size[0]:], h, h[:, :, :self.edge_size[0]]], dim=2)
        h = self.conv(h)
        h = self.bn(h) if self.bn is not None else h
        return h


class SuperConv3d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, bn=True):
        super().__init__()
        self.conv = nn.Conv3d(input_dim, output_dim, kernel_size=(1, kernel_size, kernel_size))
        self.bn = nn.BatchNorm3d(output_dim) if bn else None

    def forward(self, x):
        # TODO
        pass


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        n_chennels = 16
        self.conv0 = SuperConv3d(4, n_chennels, 3, True)
        self.convs = nn.ModuleList([SuperConv3d(n_chennels, n_chennels, 3, True) for _ in range(8)])
        self.head_p = nn.Linear(n_chennels, 4, bias=False)
        self.head_v = nn.Linear(n_chennels * 2, 1, bias=False)

    def forward(self, x):
        # TODO
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)
        h_avg = h.view(h.size(0), h.size(1), -1).mean(-1)
        p = torch.softmax(self.head_p(h_head), 1)
        v = torch.tanh(self.head_v(torch.cat([h_head, h_avg], 1)))

        # 1 位になる確率、2 位以上になる確率、3 位以上になる確率を予測
        # n 位以上になるのは n 人なので、分子を n 倍した softmax を使えば良さそう
        # ... GBDT?
        return p, v


class GeeseNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 32
        self.conv0 = TorusConv2d(17, filters, (3, 3), True)
        self.blocks = nn.ModuleList([TorusConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.head_p = nn.Linear(filters, 4, bias=False)
        self.head_v = nn.Linear(filters * 2, 1, bias=False)

    def forward(self, x):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)
        h_avg = h.view(h.size(0), h.size(1), -1).mean(-1)
        p = torch.softmax(self.head_p(h_head), 1)
        v = torch.tanh(self.head_v(torch.cat([h_head, h_avg], 1)))
        return p, v
