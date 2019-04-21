import torch
import torch.nn as nn


class MatMul(nn.Module):
    def __init__(self,input_features,output_features, bias=True):
        super(MatMul, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        dtype = torch.float
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.weight = nn.Parameter(torch.randn(input_features, output_features, dtype=dtype, requires_grad=True, device=device))
        if bias:
            self.bias = nn.Parameter(torch.randn(output_features, dtype=dtype, requires_grad=True, device=device))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

    def forward(self, input):
        out = torch.mm(input,self.weight)

        if self.bias is not None:
            out += self.bias

        return out


class GCN(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(GCN, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        dtype = torch.float
        #device = torch.device("cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.randn(input_features, output_features, dtype=dtype, requires_grad=True, device=device))
        if bias:
            self.bias = nn.Parameter(torch.randn(output_features, dtype=dtype, requires_grad=True, device=device))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)


    def forward(self, v_feat, in_mat, out_mat):
        out = v_feat
        out = torch.cat((out, torch.mm(in_mat, out), torch.mm(out_mat, out)),
                      dim=1)
        if self.bias is not None:
            out = (torch.mm(out, self.weight) + self.bias).clamp(min=0)
        else:
            out = (torch.mm(out, self.weight)).clamp(min=0)

        return out

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
