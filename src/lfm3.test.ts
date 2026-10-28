import { test, expect } from 'vitest';
import { Wllama } from './wllama';

declare const __LFM3_TEST__: boolean;

// Opt-in (LFM3_TEST=1): downloads a ~1 GB model. Proves the bundled llama.cpp
// runs the `lfm3` architecture end to end: load, plain chat, and a
// JSON-schema-constrained answer (the grammar sampler path).
const ENABLED = typeof __LFM3_TEST__ !== 'undefined' && __LFM3_TEST__;

// Pinned by commit, never a branch.
const LFM3_MODEL =
  'https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/4b423c332fd97961416f2e8b69467758fd27370d/LFM3-1.7B-QAD-Q4_0.gguf';

const TIMEOUT = 20 * 60 * 1000;

const createWllama = () => {
  const w = new Wllama({ default: '/src/wasm/wllama.wasm' }, { parallelDownloads: 3 });
  w.setCompat(null);
  return w;
};

test.sequential.skipIf(!ENABLED)(
  'LFM3-1.7B loads, chats, and follows a JSON schema',
  async () => {
    const wllama = createWllama();
    await wllama.loadModelFromUrl(LFM3_MODEL, { n_ctx: 2048, n_threads: 2 });
    expect(wllama.isModelLoaded()).toBe(true);
    expect(wllama.getModelMetadata().meta['general.architecture']).toBe('lfm3');

    const chat = await wllama.createChatCompletion({
      messages: [{ role: 'user', content: 'Reply with one word: what colour is the sky on a clear day?' }],
      max_tokens: 16,
      temperature: 0.0,
      chat_template_kwargs: { enable_thinking: false },
    });
    const text = chat.choices[0].message.content as string;
    console.log('[lfm3] chat:', JSON.stringify(text));
    expect(text.toLowerCase()).toContain('blue');

    const schema = {
      type: 'object',
      properties: {
        tool: { type: 'string', enum: ['get_balance', 'send_token', 'swap'] },
        args: {
          type: 'object',
          properties: { chain: { type: 'string' }, token: { type: 'string' } },
          required: ['chain', 'token'],
        },
      },
      required: ['tool', 'args'],
    };
    const call = await wllama.createChatCompletion({
      messages: [
        { role: 'system', content: 'You pick one wallet tool and its arguments.' },
        { role: 'user', content: 'How much USDC do I have on Monad?' },
      ],
      max_tokens: 96,
      temperature: 0.0,
      chat_template_kwargs: { enable_thinking: false },
      response_format: { type: 'json_schema', json_schema: { name: 'tool_call', schema, strict: true } },
    });
    const raw = call.choices[0].message.content as string;
    console.log('[lfm3] tool call:', raw);
    const parsed = JSON.parse(raw);
    expect(parsed.tool).toBe('get_balance');
    expect(String(parsed.args.token).toUpperCase()).toContain('USDC');

    await wllama.exit();
  },
  TIMEOUT
);
