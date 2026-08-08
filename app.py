import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import os
import sys

# Try importing Hugging Face Spaces for ZeroGPU support
try:
    import spaces
    has_spaces = True
except ImportError:
    has_spaces = False

# Add root folder to sys.path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'NST_Code'))

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization

# Load models globally on CPU at startup
vgg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'NST_Code', 'vgg_normalised.pth'))
decoder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'NST_Code', 'experiment', 'final_exp', 'decoder_final.pth'))

encoder = VGGEncoder(vgg_path).to("cpu")
decoder = Decoder().to("cpu")
decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))
encoder.eval()
decoder.eval()

# Implementation function for style transfer
def _transfer_style_impl(content_img, style_img, alpha):
    if content_img is None or style_img is None:
        return None
    
    # Detect running device dynamically (cuda inside @spaces.GPU context)
    run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Lazily move models to the current execution device
    encoder.to(run_device)
    decoder.to(run_device)
    
    # Preprocessing
    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    
    content_tensor = content_transform(content_img).unsqueeze(0).to(run_device)
    style_tensor = style_transform(style_img).unsqueeze(0).to(run_device)
    
    with torch.no_grad():
        content_feats = encoder(content_tensor, is_test=True)
        style_feats = encoder(style_tensor, is_test=True)
        
        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)
        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats
        
        stylized_tensor = decoder(stylized_feats)
        
        # Postprocessing
        output_tensor = stylized_tensor.cpu().clone().squeeze(0).clamp(0, 1)
        output_img = transforms.ToPILImage()(output_tensor)
        
    return output_img

# Wrap with ZeroGPU decorator if spaces is available
if has_spaces:
    @spaces.GPU
    def transfer_style(content_img, style_img, alpha):
        return _transfer_style_impl(content_img, style_img, alpha)
else:
    def transfer_style(content_img, style_img, alpha):
        return _transfer_style_impl(content_img, style_img, alpha)


# Set up clean theme
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="pink",
    neutral_hue="slate",
).set(
    body_background_fill="#080B11",
    body_background_fill_dark="#080B11",
    block_background_fill="#111827",
    block_background_fill_dark="#111827",
    block_border_width="1px",
    block_border_color="rgba(255, 255, 255, 0.06)",
)

with gr.Blocks(theme=theme, title="StyleForge AI") as demo:
    gr.HTML("""
        <div style="text-align: center; margin-bottom: 2rem; margin-top: 2rem;">
            <span style="display: inline-flex; align-items: center; background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.25); color: #A5B4FC; padding: 6px 16px; border-radius: 100px; font-size: 0.85rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">Neural Style Transfer</span>
            <h1 style="font-weight: 700; font-size: 3rem; background: linear-gradient(135deg, #FFFFFF 20%, #A5B4FC 60%, #E879F9 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.02em; margin-top: 0.5rem; margin-bottom: 0.5rem;">StyleForge AI</h1>
            <p style="color: #9CA3AF; max-width: 580px; margin: 0 auto; font-size: 1.1rem;">Arbitrary style transfer in real-time. Upload any photo, pick a style reference, and watch artificial intelligence forge your masterpiece.</p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column():
            content_input = gr.Image(label="Content Image", type="pil")
            style_input = gr.Image(label="Style Reference", type="pil")
            alpha_slider = gr.Slider(label="Style Intensity (Alpha)", minimum=0.0, maximum=1.0, value=1.0, step=0.1)
            submit_btn = gr.Button("Forge Masterpiece", variant="primary")
            
        with gr.Column():
            output_image = gr.Image(label="Stylized Result", type="pil")
            
    submit_btn.click(
        fn=transfer_style,
        inputs=[content_input, style_input, alpha_slider],
        outputs=output_image
    )
    
    # Examples
    examples_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'NST_Code', 'examples'))
    gr.Examples(
        examples=[
            [os.path.join(examples_dir, "brad_pitt.jpg"), os.path.join(examples_dir, "sketch.png"), 1.0],
            [os.path.join(examples_dir, "brad_pitt.jpg"), os.path.join(examples_dir, "picasso_seated_nude_hr.jpg"), 1.0],
        ],
        inputs=[content_input, style_input, alpha_slider],
        outputs=output_image,
        fn=transfer_style,
        cache_examples=False
    )

if __name__ == "__main__":
    demo.launch()
