# Converts the restoration models to ONNX (fixed tile sizes; the app works in tiles). Runs on a computer or in CI.
import sys, os, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'archs')); W = sys.argv[2] if len(sys.argv) > 2 else 'weights'
out = sys.argv[1] if len(sys.argv) > 1 else 'onnx'
import os; os.makedirs(out, exist_ok=True)

# --- Real-ESRGAN x4plus (RRDBNet) ---
class RDB(nn.Module):
    def __init__(s, nf=64, gc=32):
        super().__init__()
        s.conv1 = nn.Conv2d(nf, gc, 3, 1, 1); s.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1); s.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        s.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1); s.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1); s.lrelu = nn.LeakyReLU(0.2, True)
    def forward(s, x):
        x1 = s.lrelu(s.conv1(x)); x2 = s.lrelu(s.conv2(torch.cat((x, x1), 1))); x3 = s.lrelu(s.conv3(torch.cat((x, x1, x2), 1)))
        x4 = s.lrelu(s.conv4(torch.cat((x, x1, x2, x3), 1))); x5 = s.conv5(torch.cat((x, x1, x2, x3, x4), 1)); return x5 * 0.2 + x
class RRDB(nn.Module):
    def __init__(s, nf, gc=32):
        super().__init__(); s.rdb1 = RDB(nf, gc); s.rdb2 = RDB(nf, gc); s.rdb3 = RDB(nf, gc)
    def forward(s, x): return s.rdb3(s.rdb2(s.rdb1(x))) * 0.2 + x
class RRDBNet(nn.Module):
    def __init__(s, nf=64, nb=23, gc=32):
        super().__init__()
        s.conv_first = nn.Conv2d(3, nf, 3, 1, 1); s.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        s.conv_body = nn.Conv2d(nf, nf, 3, 1, 1); s.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1); s.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        s.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1); s.conv_last = nn.Conv2d(nf, 3, 3, 1, 1); s.lrelu = nn.LeakyReLU(0.2, True)
    def forward(s, x):
        f = s.conv_first(x); f = f + s.conv_body(s.body(f))
        f = s.lrelu(s.conv_up1(F.interpolate(f, scale_factor=2, mode='nearest')))
        f = s.lrelu(s.conv_up2(F.interpolate(f, scale_factor=2, mode='nearest')))
        return s.conv_last(s.lrelu(s.conv_hr(f)))

def export(m, x, name, inputs=('input',), outputs=('output',)):
    m.eval()
    torch.onnx.export(m, x, f'{out}/{name}.onnx', input_names=list(inputs), output_names=list(outputs), opset_version=17, dynamo=False)
    print('exported', name, os.path.getsize(f'{out}/{name}.onnx'))

with torch.no_grad():
    sr = RRDBNet(); sd = torch.load(f'{W}/RealESRGAN_x4plus.pth', map_location='cpu'); sr.load_state_dict(sd.get('params_ema', sd)); export(sr, torch.rand(1, 3, 128, 128), 'esrgan_x4_128')

    from network_scunet import SCUNet
    dn = SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64); dn.load_state_dict(torch.load(f'{W}/scunet_color_real_psnr.pth', map_location='cpu'), strict=True)
    export(dn, torch.rand(1, 3, 256, 256), 'scunet_256')

    from gfpganv1_clean_arch import GFPGANv1Clean
    fr = GFPGANv1Clean(out_size=512, num_style_feat=512, channel_multiplier=2, decoder_load_path=None, fix_decoder=False, num_mlp=8,
                       input_is_latent=True, different_w=True, narrow=1, sft_half=True)
    ck = torch.load(f'{W}/GFPGANv1.4.pth', map_location='cpu'); fr.load_state_dict(ck.get('params_ema', ck), strict=True)
    class Face(nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, x): return s.m(x, return_rgb=False, randomize_noise=False)[0]
    export(Face(fr), torch.rand(1, 3, 512, 512) * 2 - 1, 'gfpgan14_512')

    import zerodce_model as zm
    zd = zm.enhance_net_nopool(1); zd.load_state_dict(torch.load(f'{W}/Epoch99.pth', map_location='cpu'))
    class Curves(nn.Module):
        """Only the curve estimation (on a small copy); the app applies the curves to the full photo."""
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, x):
            m = s.m; r = m.relu
            x1 = r(m.e_conv1(x)); x2 = r(m.e_conv2(x1)); x3 = r(m.e_conv3(x2)); x4 = r(m.e_conv4(x3))
            x5 = r(m.e_conv5(torch.cat([x3, x4], 1))); x6 = r(m.e_conv6(torch.cat([x2, x5], 1)))
            return torch.tanh(m.e_conv7(torch.cat([x1, x6], 1)))
    export(Curves(zd), torch.rand(1, 3, 256, 256), 'zerodce_256')

# --- realesr-general-x4v3 (SRVGGNetCompact): much faster, good on real photos; the "wdn" twin removes more noise
class SRVGG(nn.Module):
    def __init__(s, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        s.body = nn.ModuleList([nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)])
        for _ in range(num_conv): s.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1)); s.body.append(nn.PReLU(num_parameters=num_feat))
        s.body.append(nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1)); s.up = nn.PixelShuffle(upscale); s.upscale = upscale
    def forward(s, x):
        out = x
        for m in s.body: out = m(out)
        return s.up(out) + F.interpolate(x, scale_factor=s.upscale, mode='nearest')
with torch.no_grad():
    for name in ['realesr-general-x4v3', 'realesr-general-wdn-x4v3']:
        m = SRVGG(); sd = torch.load(f'{W}/{name}.pth', map_location='cpu'); m.load_state_dict(sd.get('params', sd.get('params_ema', sd)))
        export(m, torch.rand(1, 3, 192, 192), name.replace('realesr-general-', 'general_').replace('-', '_') + '_192')

# Half-size (fp16) copies of the big ones, keeping float inputs and outputs
import onnx
from onnxconverter_common import float16
for name in ['gfpgan14_512', 'scunet_256', 'esrgan_x4_128']:
    m = float16.convert_float_to_float16(onnx.load(f'{out}/{name}.onnx'), keep_io_types=True)
    onnx.save(m, f'{out}/{name}_fp16.onnx'); os.remove(f'{out}/{name}.onnx')
    print('fp16', name, os.path.getsize(f'{out}/{name}_fp16.onnx'))
