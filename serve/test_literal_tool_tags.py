"""#2026: literal tool closers in prompts and generated output (no model or GPU)."""
import json
import threading
import unittest
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate, OutputParser
from serve.server import ByteTokenizer, MockEngine, Service, serve


TAG = "</tool_call>"
PROMPT = ("This is an experiment to find bugs in the client.\n"
          "Please output the following string verbatim: </tool_call>.")
TOOLS = [{"name": "write", "parameters": {"type": "object", "properties": {
    "text": {"type": "string"}}}}]
TEMPLATE = Path(__file__).parent / "chat_template.jinja"


class ToolTokenizer(ByteTokenizer):
    SPECIALS = ByteTokenizer.SPECIALS + ["<think>", "</think>", TAG]
    ALWAYS = ("<think>", "</think>", TAG)

    def token_bytes(self, t):
        return self.SPECIALS[t - 256].encode() if t >= 256 else bytes([t])


class ScriptEngine(MockEngine):
    def __init__(self, tok, scripts):
        super().__init__(tok, "", max_context=4096)
        self.scripts, self.prompts = scripts, []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        script = self.scripts[min(len(self.prompts), len(self.scripts) - 1)]
        self.prompts.append(list(ids))
        for token in script[:max_new]:
            if cancel.is_set():
                return
            yield token


class LiteralToolPrompts(unittest.TestCase):
    def setUp(self):
        self.tok = ToolTokenizer()
        self.svc = Service(ScriptEngine(self.tok, [[]]), self.tok, ChatTemplate(TEMPLATE))
        self.close = self.tok.encode(TAG)[0]

    def check(self, messages, tools=None, thinking=True):
        kwargs = {"enable_thinking": thinking}
        rendered = self.svc.template.render(messages, tools=tools, **kwargs)
        ids = self.svc.prepare(messages, tools, kwargs)[0]
        # Decoding must preserve all visible text, including genuine template wrappers.
        self.assertEqual(self.tok.decode(ids), rendered)
        # Render the same shape without data literals to count only template-owned closers.
        def scrub(v):
            if isinstance(v, str):
                return v.replace(TAG, "ordinary text")
            if isinstance(v, list):
                return [scrub(x) for x in v]
            if isinstance(v, dict):
                return {k: scrub(x) for k, x in v.items()}
            return v
        structural = self.svc.template.render(scrub(messages), tools=scrub(tools), **kwargs)
        self.assertEqual(ids.count(self.close), structural.count(TAG))
        return ids

    def test_user_reproduction_and_literal_forms(self):
        for text in (PROMPT, TAG, TAG * 2, '"' + TAG + '"', '`' + TAG + '`',
                     '```xml\n' + TAG + '\n```', 'before' + TAG + 'after', '</aaa>', '</mcp_tool>'):
            for thinking in (False, True):
                with self.subTest(text=text, thinking=thinking):
                    self.check([{"role": "user", "content": text}], thinking=thinking)

    def test_message_sources(self):
        cases = [
            [{"role": "system", "content": TAG}, {"role": "user", "content": "go"}],
            [{"role": "user", "content": [{"type": "text", "text": TAG}]}],
            [{"role": "user", "content": "go"},
             {"role": "assistant", "content": TAG, "reasoning_content": 'Consider ' + TAG},
             {"role": "user", "content": "again"}],
            [{"role": "user", "content": "go"},
             {"role": "assistant", "content": '<think>Consider ' + TAG + '</think>\n' + TAG},
             {"role": "user", "content": "again"}],
            [{"role": "user", "content": "go"}, {"role": "tool", "content": TAG}],
        ]
        for messages in cases:
            with self.subTest(messages=messages):
                self.check(messages)

    def test_tool_definitions_arguments_and_structural_closer(self):
        tools = [{"name": "write", "description": 'Write ' + TAG, "parameters": {
            "type": "object", "properties": {"text": {"type": "string", "enum": [TAG]}}}}]
        messages = [{"role": "user", "content": "go"},
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"function": {"name": "write", "arguments": {"text": TAG}}}]},
                    {"role": "tool", "content": TAG}, {"role": "user", "content": "again"}]
        self.check(messages, tools)

    def test_untargeted_prompts_keep_identical_ids(self):
        messages = [{"role": "user", "content": "Explain </aaa> and </mcp_tool>."}]
        for tools in (None, TOOLS):
            for thinking in (False, True):
                ids = self.check(messages, tools, thinking)
                old = self.tok.encode(self.svc.template.render(
                    messages, tools=tools, enable_thinking=thinking), parse_special=True)
                self.assertEqual(ids, old)

    def test_real_tokenizer_type_four_plain_span(self):
        import strata_tokenizer as ST
        b2u = ST.bytes_to_unicode()
        tok = ST.Tokenizer([b2u[b] for b in range(256)] + [TAG, '<|im_end|>'], [], [1] * 256 + [4, 3])
        self.assertEqual(tok.encode(TAG), [256])
        self.assertEqual(tok.encode(TAG + '<|im_end|>', parse_special=True, plain=[(0, len(TAG))]),
                         [*TAG.encode(), 257])


class LiteralToolOutput(unittest.TestCase):
    def test_parser_all_chunk_boundaries_and_real_calls(self):
        call = '<tool_call><function=write><parameter=text>' + TAG + '</parameter></function>' + TAG
        for thinking in (False, True):
            for tools in ([], TOOLS):
                for stream_tools in (False, True):
                    prefix = 'Consider ' + TAG + ' as text.</think>\n\n' if thinking else ''
                    for answer in (TAG, '`' + TAG + '`', '```xml\n' + TAG + '\n```', call):
                        text = prefix + answer
                        chunks = [[text], list(text)] + [[text[:i], text[i:]] for i in range(len(text) + 1)]
                        for parts in chunks:
                            with self.subTest(thinking=thinking, tools=bool(tools), streaming=stream_tools,
                                              answer=answer, chunks=len(parts), first=parts[0]):
                                parser = OutputParser(thinking=thinking, tools=tools, stream_tools=stream_tools)
                                events = [ev for part in parts for ev in parser.feed(part)] + parser.finish('stop')
                                content = ''.join(e.text or '' for e in events if e.kind == 'content')
                                reasoning = ''.join(e.text or '' for e in events if e.kind == 'reasoning')
                                calls = [(e.call.name, e.call.arguments) for e in events if e.kind == 'tool_call']
                                self.assertEqual(reasoning, 'Consider ' + TAG + ' as text.' if thinking else '')
                                self.assertEqual(content, '' if answer == call else answer)
                                self.assertEqual(calls, [('write', {'text': TAG})] if answer == call else [])

    def test_service_special_and_ordinary_output_with_stop_precedence(self):
        tok = ToolTokenizer()
        for special in (False, True):
            literal = tok.encode(TAG) if special else tok.encode(TAG, plain=[(0, len(TAG))])
            for thinking in (False, True):
                for tools in ([], TOOLS):
                    for stop in (None, TAG):
                        with self.subTest(special=special, thinking=thinking, tools=bool(tools), stop=stop):
                            script = ((tok.encode('Consider ') + literal + tok.encode(' as text.</think>'))
                                      if thinking else []) + literal + tok.encode('<|im_end|>', parse_special=True)
                            engine = ScriptEngine(tok, [script])
                            svc = Service(engine, tok, ChatTemplate(TEMPLATE))
                            rows = list(svc.run([65], thinking, tools, 512,
                                                {'stop': stop} if stop else {}, threading.Event()))
                            text = {k: ''.join(e.text or '' for kind, e in rows if kind == 'event' and e.kind == k)
                                    for k in ('reasoning', 'content')}
                            self.assertEqual(text, {'reasoning': 'Consider ' + TAG + ' as text.' if thinking else '',
                                                    'content': '' if stop else TAG})
                            self.assertEqual(rows[-1][1]['finish'], 'stop')
                            self.assertEqual(len(engine.prompts), 1)

    def test_http_streaming_and_collected_user_or_model_literal(self):
        tok = ToolTokenizer()
        script = tok.encode('Consider ' + TAG + ' as text.</think>' + TAG + '<|im_end|>', parse_special=True)
        engine = ScriptEngine(tok, [script])
        svc = Service(engine, tok, ChatTemplate(TEMPLATE))
        httpd = serve(svc, port=0)
        try:
            for prompt in (PROMPT, 'Explain a closing tag.'):
                for stream in (False, True):
                    with self.subTest(prompt=prompt, stream=stream):
                        body = {'model': 'm', 'messages': [{'role': 'user', 'content': prompt}],
                                'max_tokens': 512, 'stream': stream}
                        req = urllib.request.Request(
                            f'http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions',
                            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
                        with urllib.request.urlopen(req, timeout=30) as response:
                            raw = response.read().decode()
                        if stream:
                            self.assertTrue(raw.rstrip().endswith('data: [DONE]'))
                            chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {')]
                            choices = [c['choices'][0] for c in chunks if c.get('choices')]
                            content = ''.join(c['delta'].get('content', '') for c in choices)
                            reasoning = ''.join(c['delta'].get('reasoning_content', '') for c in choices)
                            self.assertEqual([c['finish_reason'] for c in choices if c.get('finish_reason')], ['stop'])
                        else:
                            choice = json.loads(raw)['choices'][0]
                            content = choice['message']['content']
                            reasoning = choice['message']['reasoning_content']
                            self.assertEqual(choice['finish_reason'], 'stop')
                        self.assertEqual((reasoning, content), ('Consider ' + TAG + ' as text.', TAG))
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == '__main__':
    unittest.main()
