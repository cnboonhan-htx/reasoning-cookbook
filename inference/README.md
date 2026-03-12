# Inference

Download and load `nvidia/Cosmos-Reason2-8B` via HuggingFace Transformers / vLLM. Provides a simple inference wrapper (prompt in, reasoning + answer out) with support for image/video inputs, since Cosmos-Reason2 is a vision-language model for physical reasoning.

## Claude OpenAI-Compatible Endpoint

For using Claude as a teacher VLM via an OpenAI-compatible API, see [claude-max-api-proxy](https://docs.openclaw.ai/providers/claude-max-api-proxy).

```bash
# Requires Node.js 20+ and Claude Code CLI
npm install -g claude-max-api-proxy

# Verify Claude CLI is authenticated
claude --version
```
