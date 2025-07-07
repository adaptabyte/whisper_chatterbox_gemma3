import random
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from transformers import Gemma3ForCausalLM
import threading
from chatterbox.src.chatterbox.tts import ChatterboxTTS
import gradio as gr
import spaces

# —– Configuration —–
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

# —– Load ASR Model via Transformers Pipeline —–
asr_device = 0 if DEVICE == "cuda" else -1
asr_model = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
    device=asr_device,
)

# —– Load / Cache TTS Model —–
TTS_MODEL = None

def get_or_load_tts():
    global TTS_MODEL
    if TTS_MODEL is None:
        print("Loading TTS model...")
        TTS_MODEL = ChatterboxTTS.from_pretrained(DEVICE)
        if hasattr(TTS_MODEL, "to"):
            TTS_MODEL.to(DEVICE)
    return TTS_MODEL

# —– Load / Cache Local Chat Model (Gemma 3) —–
CHAT_TOKENIZER = None
CHAT_MODEL = None

def get_or_load_chat_model():
    """Loads Google Gemma 3-1B with 8-bit on GPU or full precision on CPU."""
    global CHAT_TOKENIZER, CHAT_MODEL
    if CHAT_MODEL is None:
        model_id = "google/gemma-3-1b-it"
        if DEVICE == "cuda":
            print("Loading Gemma 3 (1B) in 8-bit...")
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
            CHAT_MODEL = Gemma3ForCausalLM.from_pretrained(
                model_id,
                quantization_config=quant_cfg,
                device_map="auto",
                use_auth_token=True,
            ).eval()
        else:
            print("No CUDA detected; loading Gemma 3 (1B) in full precision (FP32). This may be slow.")
            CHAT_MODEL = Gemma3ForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                use_auth_token=True,
            ).eval()
            CHAT_MODEL.to(DEVICE)
        CHAT_TOKENIZER = AutoTokenizer.from_pretrained(model_id)
    return CHAT_TOKENIZER, CHAT_MODEL

# —– Utilities —–
def transcribe(audio_path: str) -> str:
    """Run HF ASR pipeline on the uploaded audio file."""
    result = asr_model(audio_path)
    return result.get("text", "").strip()

def generate_chat_response(history: list, user_text: str) -> tuple[str, list]:
    """Generate a conversational reply using Gemma 3 and append to history."""
    tokenizer, model = get_or_load_chat_model()
    # 1) append user turn
    history.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
    # 2) assemble messages (system + history)
    system_msg = {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant. limit all responses to 300 characters or less."}]}  
    messages = [system_msg] + history
    # 3) tokenize
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # 4) Stream generation
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = {
        **{k: v for k, v in inputs.items()},
        "max_new_tokens": 128,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
        "streamer": streamer,
    }
    threading.Thread(target=model.generate, kwargs=generate_kwargs).start()
    response = "".join([chunk for chunk in streamer])
    response = response.strip()
    # 5) append assistant turn
    history.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
    return response, history


def set_seed(seed: int):
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

# —– Main Pipeline —–
@spaces.GPU
def respond_to_audio(
    audio_input_path: str,
    history: list,
    exaggeration: float = 0.5,
    temperature: float = 0.8,
    seed_num: int = 0,
    cfgw: float = 0.5,
) -> tuple[tuple[int, np.ndarray], list]:
    # 1) Transcribe user audio → text
    user_text = transcribe(audio_input_path)
    print(f"ASR result: {user_text}")

    # 2) Generate text response locally (with history)
    bot_text, new_history = generate_chat_response(history, user_text)
    print(f"Chat response: {bot_text}")

    # 3) Synthesize response text → speech
    tts = get_or_load_tts()
    if seed_num:
        set_seed(int(seed_num))
    generate_kwargs = {
        "exaggeration": exaggeration,
        "temperature": temperature,
        "cfg_weight": cfgw,
    }
    wav = tts.generate(
        bot_text,
        audio_prompt_path=audio_input_path,
        **generate_kwargs
    )
    return (tts.sr, wav.squeeze(0).numpy()), new_history

# —– Gradio Interface —–
with gr.Blocks() as demo:
    gr.Markdown("# Conversational Audio Demo\nSpeak into your mic and get a spoken reply!")
    with gr.Row():
        with gr.Column():
            audio_in     = gr.Audio(sources="microphone", type="filepath", label="Your Question")
            exaggeration = gr.Slider(0.25, 2, step=0.05, label="Exaggeration", value=0.5)
            cfgw         = gr.Slider(0.2, 1, step=0.05, label="CFG/Pace", value=0.5)
            seed_num     = gr.Number(value=0, label="Seed (0=random)")
            temp         = gr.Slider(0.05, 5, step=0.05, label="Temperature", value=0.8)
            btn          = gr.Button("Generate Response", variant="primary")
            # hidden state to keep conversation history
            history      = gr.State([])
        with gr.Column():
            audio_out = gr.Audio(label="Chatterbox Replies")
    btn.click(
        fn=respond_to_audio,
        inputs=[audio_in, history, exaggeration, temp, seed_num, cfgw],
        outputs=[audio_out, history],
    )

if __name__ == "__main__":
    demo.launch()
