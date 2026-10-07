# jev-local

A small local server that answers decision questions with [Intern-Decision-4B](https://huggingface.co/internlm/Intern-Decision-4B) on Apple silicon. It speaks the JEV request shape (`POST /v1/systemone`) and runs on [MLX](https://github.com/ml-explore/mlx), without PyTorch. The model is downloaded once from Hugging Face and quantised to 4 bits on your Mac. It uses about 2.9 GB steady and 4.0 GB peak, and answers in about 165 ms per request.

[CommissionAI](https://github.com/patrick-lai/commission-ai) downloads and refreshes this runtime on its own. You can also run it by hand.

## Requirements

- A Mac with Apple silicon and at least 16 GB of memory
- Python 3.12 with the packages listed by `jev-local version --json` (`packages`)
- About 8.5 GB of free disk space for the first download

## Install

```sh
npm install -g @patrick-lai/jev-local
jev-local version --json
```

Or download `jev-local-<version>-darwin-arm64.tar.gz` from [Releases](https://github.com/patrick-lai/jev-local/releases) and check it against the signed `manifest.json` (public key in `release-key.pub`).

## Run

```sh
uv venv --python 3.12 venv
uv pip install --python venv/bin/python $(jev-local version --json | jq -r .packages)
mkdir -p state
export HF_HOME=$PWD/state/hf DECISION_HOST=127.0.0.1 DECISION_PORT=8089 DECISION_API_KEY=change-me \
  DECISION_REPO=$(jev-local version --json | jq -r .repo) \
  DECISION_REVISION=$(jev-local version --json | jq -r .revision) \
  DECISION_MODEL_DIR=$PWD/state/model DECISION_MEMORY_LIMIT_MB=4096 DECISION_CACHE_LIMIT_MB=256 \
  DECISION_MAX_PROMPT_TOKENS=8192
touch state/owner.lock
JEV_LOCAL_PYTHON=venv/bin/python jev-local serve state/owner.lock 20
```

The server exits when nothing holds an exclusive lock on the owner file for the grace period (the second argument, in seconds), so run it under a parent that holds the lock, or pass a long grace period.

```sh
curl -s -H 'Authorization: Bearer change-me' -H 'Content-Type: application/json' \
  -d '{"state":"A duplicate charge.","questions":{"t":{"type":"choice","instructions":"Which team?","criteria":{"billing":"Payments","shipping":"Delivery"}}}}' \
  http://127.0.0.1:8089/v1/systemone
```

`GET /health` reports the loaded model and MLX memory use. Input over the model's 8,192-token context is refused with `413`, never truncated, and the refusal does not echo the input.

## Runtime contract

`jev-local version --json` prints the version, a protocol integer (`1`), the model name, the Hugging Face repository and pinned revision, the Python version, the pinned packages and the download size. A host that downloads this runtime checks the protocol before using it. A new model or request shape bumps the protocol.

## Develop

```sh
python3 -m unittest discover -s tests -v
scripts/build-dist.sh
scripts/pack-npm.sh
```

The tests run the real server against stub MLX, tokenizer and Hugging Face modules, so they need no GPU and download nothing.

## Release

Update `version` in `share/jev-local/runtime.json`, commit, and push a `v<version>` tag. The release workflow tests, packages, signs `manifest.json` with the repository's release key, creates the GitHub release and publishes the npm package.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
