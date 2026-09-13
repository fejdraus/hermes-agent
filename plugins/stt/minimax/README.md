# MiniMax speech-to-text

Registers `minimax` as an `stt.provider` backend, served by MiniMax `asr-1.0`.

```yaml
stt:
  provider: minimax
  minimax:
    model: asr-1.0
```

The key is read from `MINIMAX_API_KEY` (`stt.minimax.api_key` / `key_env` override it).
`MINIMAX_STT_BASE_URL` points the call at another MiniMax region; the default is
`https://api.minimax.io`.

## Why it exists

Whisper decides the language from the audio itself, and on a short utterance it
guesses wrong: a Ukrainian sentence came back as Slovak in Latin script. The
`language` hint fixes that only by pinning one language, which then mangles the
other one for a bilingual speaker. `asr-1.0` needs no hint to keep either
language in its own script, so nothing has to be pinned.

A `language` hint is still forwarded when one is configured.
