from JOSIEv4o.args import ModelArgs

from JOSIEv4o.JODIOv3 import MultiStreamingJODIO
from JOSIEv4o.JODIOv2 import StreamingJODIO
from JOSIEv4o.JODIOv1 import JODIO
from audio_tokenizer import LiveAudioTokenizer

def handle_tokens(token_data):
    current = token_data['current_tokens']
    history = token_data['token_history']
    context = token_data['full_context']
    print(f"Tokens: {token_data['current_tokens']}")
    print(f"\nProcessed chunk with {len(current)} tokens")
    print(f"Total context length: {len(context)}")

args = ModelArgs()
model = StreamingJODIO(args)

tokenizer = LiveAudioTokenizer(model, args)
tokenizer.stream_tokens(callback_fn=handle_tokens)