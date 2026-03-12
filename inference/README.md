# Inference

Download and load `nvidia/Cosmos-Reason2-8B` via HuggingFace Transformers / vLLM. Provides a simple inference wrapper (prompt in, reasoning + answer out) with support for image/video inputs, since Cosmos-Reason2 is a vision-language model for physical reasoning.

## Claude OpenAI-Compatible Endpoint

For using Claude as a teacher VLM via an OpenAI-compatible API, see [cliproxyapi](https://help.router-for.me/introduction/quick-start.html).

```
curl -fsSL https://raw.githubusercontent.com/brokechubb/cliproxyapi-installer/refs/heads/master/cliproxyapi-installer | bash
systemctl --user enable cliproxyapi.service
systemctl --user start cliproxyapi.service
curl -H "Authorization: Bearer $OPENAI_API_KEY" http://127.0.0.1:8317/v1/models 
```
