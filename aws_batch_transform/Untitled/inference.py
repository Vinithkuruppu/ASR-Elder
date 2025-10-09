import os, json, torch
from transformers import pipeline

print("### LOADED CUSTOM INFERENCE SCRIPT")

try:
    import imageio_ffmpeg
    os.environ["PATH"] = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe()) + os.pathsep + os.environ.get("PATH","")
except Exception as e:
    print("ffmpeg setup warning:", e)

def model_fn(model_dir):
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("automatic-speech-recognition",
                    model=os.environ.get("HF_MODEL_ID","openai/whisper-base"),
                    device=device)

def input_fn(request_body, content_type):
    if content_type in ("audio/mpeg","audio/mp3","audio/wav","application/octet-stream"):
        return request_body
    if content_type in ("application/json","application/jsonlines","text/plain"):
        try: return json.loads(request_body)
        except Exception: return request_body
    return request_body

def predict_fn(input_data, asr):
    if isinstance(input_data, (bytes, bytearray)):
        out = asr(input_data)
        if isinstance(out, dict) and "text" in out: return {"text": out["text"]}
        return {"text": str(out)}
    if isinstance(input_data, dict) and "inputs" in input_data:
        out = asr(input_data["inputs"])
        if isinstance(out, dict) and "text" in out: return {"text": out["text"]}
        return {"text": str(out)}
    return {"text":"", "error":"unsupported_input_type"}

def output_fn(prediction, accept):
    body = json.dumps(prediction)
    if accept in ("application/jsonlines","application/x-jsonlines"):
        return body+"\n", "application/jsonlines"  # per record
    return body, "application/json"

