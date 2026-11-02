"""Reuse the installed benchmark loaders; extend inputs to image-bearing demonstrations.

Heavy imports are intentionally delayed until after --dry-run validation.
"""
import importlib
import importlib.util
from copy import copy
from pathlib import Path
from types import SimpleNamespace

def load_legacy(path):
    spec = importlib.util.spec_from_file_location('prompt_eval_legacy', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def content_text(message):
    return '\n'.join(c['text'] for c in message['content'] if c['type'] == 'text')

def images_in(messages):
    return [c['image'] for m in messages for c in m['content'] if c['type'] == 'image']

class LocalBackend:
    def __init__(self, entry, args):
        import torch
        self.torch, self.entry, self.args = torch, entry, args
        self.kind = entry['backend']
        self.legacy = load_legacy(entry['legacy'])
        m = self.legacy
        path = str(args.model_path or entry['model_path'])
        repo = Path(args.repo_root or entry['repo_root'])
        self.info = dict(backend=self.kind, model_path=path, repo_root=str(repo),
                         legacy_script=entry['legacy'], native_thinking=args.enable_thinking,
                         context_policy='full architectural window; no silent visual-token truncation')
        if self.kind in ('qwen', 'glm', 'patho'):
            import transformers
            self.processor = transformers.AutoProcessor.from_pretrained(path, trust_remote_code=True)
            if self.kind == 'qwen':
                loader = getattr(transformers, 'AutoModelForMultimodalLM', None) or transformers.AutoModelForCausalLM
                self.model = loader.from_pretrained(path, torch_dtype=args.torch_dtype,
                    device_map=args.device_map, trust_remote_code=True).eval()
            elif self.kind == 'glm':
                self.model = m.load_model(Path(path), args.torch_dtype, args.device_map).eval()
            else:
                self.model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    path, torch_dtype=args.torch_dtype, device_map=args.device_map).eval()
        elif self.kind == 'health':
            load, self.decode = m.import_healthgpt(repo)
            self.model, self.processor, attn = m.load_healthgpt_model(
                load_model=load, model_path=Path(path), dtype=m.resolve_dtype(args.health_dtype),
                attn_implementation=args.attn_implementation)
            self.info['actual_attn_implementation'] = attn
        elif self.kind in ('quilt', 'pathgen'):
            self.d = m.import_quilt_llava(repo) if self.kind == 'quilt' else m.import_llava(repo)
            if 'disable_torch_init' in self.d:
                self.d['disable_torch_init']()
            name = self.d['get_model_name_from_path'](path)
            kwargs = dict(device=args.device) if self.kind == 'quilt' else dict(device_map=args.device_map)
            self.tokenizer, self.model, self.image_processor, context = self.d['load_pretrained_model'](
                path, None, name, False, False, **kwargs)
            self.model.eval()
            self.loader_context = context
            self.install_multimodal_guard('prepare_inputs_labels_for_multimodal')
        elif self.kind == 'huatuo':
            bot = m.import_huatuo(repo)(model_dir=path, device=args.device)
            self.bot, self.model, self.tokenizer = bot, bot.model.eval(), bot.tokenizer
            m.patch_huatuo_forward_cache_position(self.model)
            m.patch_huatuo_generate_attention_mask(self.model)
            if bot.tokenizer.pad_token_id is None:
                bot.tokenizer.pad_token_id = bot.tokenizer.eos_token_id
            self.install_multimodal_guard('prepare_inputs_labels_for_multimodal_new')
        elif self.kind == 'unimed':
            config = SimpleNamespace(repo_root=repo, model_path=Path(path), checkpoint_weight_path=None,
                cpu_optimized_loading=1, use_fast_chat_api=1, max_mem_per_gpu=args.max_mem_per_gpu,
                loader='multi_gpu_dispatch')
            self.bundle = m.load_unimedvl(config)
            self.model, self.tokenizer = self.bundle['model'], self.bundle['tokenizer']
        else:
            raise ValueError(self.kind)
        self.model.eval()

    def context_limit(self):
        limits = []
        if self.args.max_context_tokens:
            limits.append(self.args.max_context_tokens)
        for config in (self.model.config, getattr(self.model.config, 'text_config', None),
                       getattr(self.model.config, 'llm_config', None)):
            if config:
                for key in ('max_position_embeddings', 'max_sequence_length'):
                    v = getattr(config, key, None)
                    if isinstance(v, int) and 0 < v < 10**8:
                        limits.append(v)
        v = getattr(self, 'loader_context', None)
        if not limits and isinstance(v, int) and v > 0:
            limits.append(v)
        return min(limits) if limits else None

    def check_context(self, n):
        limit = self.context_limit()
        if limit and n + self.args.max_new_tokens > limit:
            raise ValueError(f'Context would overflow: input={n}, generation={self.args.max_new_tokens}, '
                             f'limit={limit}. No example/image is silently dropped. '
                             'Review a common token budget before the formal experiment.')

    def install_multimodal_guard(self, name):
        """Measure expanded visual tokens BEFORE legacy LLaVA can silently truncate."""
        original = getattr(self.model, name)
        self.info['legacy_tokenizer_truncation_cap'] = getattr(self.model.config, 'tokenizer_model_max_length', None)
        def guarded(*a, **kw):
            config = self.model.config
            old = getattr(config, 'tokenizer_model_max_length', None)
            if old is not None:
                config.tokenizer_model_max_length = None
            try:
                result = original(*a, **kw)
            finally:
                if old is not None:
                    config.tokenizer_model_max_length = old
            embeds = result[4]
            if embeds is not None:
                self.check_context(int(embeds.shape[1]))
            return result
        setattr(self.model, name, guarded)

    def generation(self):
        kw = dict(max_new_tokens=self.args.max_new_tokens,
                  do_sample=self.args.temperature > 0, use_cache=True)
        if self.args.temperature > 0:
            kw['temperature'] = self.args.temperature
        return kw

    def decode_result(self, output, input_ids, tokenizer):
        t = self.torch
        if output.ndim == 1:
            output = output.unsqueeze(0)
        # Some installed LLaVA releases return only generated IDs; others return the prefix too.
        n = input_ids.shape[1]
        new = output[:, n:] if output.shape[1] >= n and t.equal(output[:, :n], input_ids) else output
        raw = tokenizer.batch_decode(new, skip_special_tokens=True)[0].strip()
        count = int(new.shape[1])
        eos = getattr(getattr(tokenizer, 'tokenizer', tokenizer), 'eos_token_id', None)
        ended = bool(count and eos is not None and int(new[0, -1]) == eos)
        return dict(raw_output=raw, generated_tokens=count,
                    truncated=count >= self.args.max_new_tokens and not ended)

    def __call__(self, messages):
        with self.torch.inference_mode():
            if self.kind in ('qwen', 'glm', 'patho', 'health'):
                return self.hf(messages)
            if self.kind in ('quilt', 'pathgen'):
                return self.llava(messages)
            if self.kind == 'huatuo':
                return self.huatuo(messages)
            return self.unimed(messages)

    def final_answer(self, messages, max_new_tokens):
        # Same weights and full multimodal context, with a short answer-only
        # budget. Restore the original configuration even if generation fails.
        original = self.args
        self.args = copy(original)
        self.args.max_new_tokens = max_new_tokens
        self.args.enable_thinking = 0
        try:
            return self(messages)
        finally:
            self.args = original

    def hf(self, messages):
        p = self.processor
        if self.kind == 'patho':
            from qwen_vl_utils import process_vision_info
            msgs = [{'role': 'system', 'content': 'You are a pathology expert.'}] + messages
            prompt = p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs, videos = process_vision_info(msgs)
            inputs = p(text=[prompt], images=imgs, videos=videos, padding=True, return_tensors='pt')
        else:
            kw = dict(tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
            if self.kind in ('qwen', 'glm'):
                kw['enable_thinking'] = bool(self.args.enable_thinking)
            inputs = p.apply_chat_template(messages, **kw)
        inputs = inputs.to(self.model.device)
        if self.kind == 'glm':
            inputs.pop('token_type_ids', None)
        self.check_context(int(inputs['input_ids'].shape[1]))
        output = self.model.generate(**inputs, **self.generation())
        return self.decode_result(output, inputs['input_ids'], p)

    def llava(self, messages):
        d, model = self.d, self.model
        conv = d['conv_templates'][self.args.conv_mode].copy()
        paths = []
        image_token = d['DEFAULT_IMAGE_TOKEN']
        if getattr(model.config, 'mm_use_im_start_end', False):
            image_token = d['DEFAULT_IM_START_TOKEN'] + image_token + d['DEFAULT_IM_END_TOKEN']
        for msg in messages:
            content = []
            for c in msg['content']:
                if c['type'] == 'image':
                    paths.append(c['image'])
                    content.append(image_token)
                else:
                    content.append(c['text'])
            conv.append_message(conv.roles[0 if msg['role'] == 'user' else 1], '\n'.join(content))
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        ids = d['tokenizer_image_token'](prompt, self.tokenizer, d['IMAGE_TOKEN_INDEX'],
                                         return_tensors='pt').unsqueeze(0).to(model.device)
        if int((ids == d['IMAGE_TOKEN_INDEX']).sum()) != len(paths):
            raise ValueError('Image placeholder count does not match actual images')
        self.check_context(int(ids.shape[1]))
        kw = self.generation()
        if paths:
            if self.kind == 'quilt':
                tensors = [self.legacy.prepare_image_tensor(Path(x), self.image_processor, model, 'pad', d)
                           for x in paths]
                kw['images'] = self.torch.cat(tensors, dim=0)
            else:
                prepared = [self.legacy.prepare_image_tensor(Path(x), self.image_processor, model, d) for x in paths]
                # Each list entry represents ONE image, including any-resolution crop groups.
                kw['images'] = [x[0].squeeze(0) if x[0].ndim == 5 else x[0] for x in prepared]
                kw['image_sizes'] = [x[1] for x in prepared]
        stop = None
        if self.kind == 'quilt':
            stop = conv.sep if conv.sep_style != d['SeparatorStyle'].TWO else conv.sep2
            kw['stopping_criteria'] = [d['KeywordsStoppingCriteria']([stop], self.tokenizer, ids)]
        output = model.generate(ids, **kw)
        result = self.decode_result(output, ids, self.tokenizer)
        if stop and result['raw_output'].endswith(stop):
            result['raw_output'] = result['raw_output'][:-len(stop)].strip()
            result['truncated'] = False
        return result

    def huatuo(self, messages):
        b = self.bot
        paths, conversation = [], []
        for msg in messages:
            blocks = []
            for c in msg['content']:
                if c['type'] == 'image':
                    paths.append(c['image'])
                    blocks.append(b.insert_image_placeholder('', num_images=1))
                else:
                    blocks.append(b.input_moderation(c['text']))
            conversation.append({'from': 'human' if msg['role'] == 'user' else 'gpt', 'value': '\n'.join(blocks)})
        conversation.append({'from': 'gpt', 'value': None})
        if len(paths) > b.max_image_num:
            raise ValueError('Too many images for HuatuoGPT-Vision')
        ids = b.preprocess(conversation, return_tensors='pt').unsqueeze(0).to(b.device)
        self.check_context(int(ids.shape[1]))
        tensors = self.torch.stack(b.get_image_tensors(paths)).to(dtype=self.torch.bfloat16, device=b.device) if paths else None
        out = b.model.generate(ids, images=tensors, eos_token_id=b.tokenizer.eos_token_id,
                               pad_token_id=b.tokenizer.pad_token_id, **self.generation())
        return self.decode_result(out, ids, b.tokenizer)

    def unimed(self, messages):
        """Native cache-prefill primitives retain text/image order, unlike chat(images,prompt)."""
        from PIL import Image
        t, m, bundle = self.torch, self.model, self.bundle
        ids = bundle['new_token_ids']
        module = importlib.import_module(type(m).__module__)
        cache = module.NaiveCache(m.config.llm_config.num_hidden_layers)
        lens, rope = [0], [0]
        device = next(m.parameters()).device
        def move(data):
            return {k: v.to(device) if t.is_tensor(v) else v for k, v in data.items()}
        # Encode each ordered block; examples are explicitly labelled rather than inventing a chat template.
        target_seen = False
        for msg in messages:
            if msg['role'] == 'user' and any(c.get('text') == 'Target question' for c in msg['content']):
                target_seen = True
            for c in msg['content']:
                if c['type'] == 'image':
                    with Image.open(c['image']) as im:
                        data, lens, rope = m.prepare_vit_images(lens, rope, [im.convert('RGB')],
                            bundle['image_transform'], ids)
                    update = m.forward_cache_update_vit
                else:
                    prefix = ('Previous target response: ' if target_seen else 'Example answer: ') if msg['role'] == 'assistant' else ''
                    text = prefix + c['text']
                    data, lens, rope = m.prepare_prompts(lens, rope, [text], self.tokenizer, ids)
                    update = m.forward_cache_update_text
                self.check_context(lens[0])
                with t.amp.autocast('cuda', enabled=True, dtype=t.bfloat16):
                    cache = update(cache, **move(data))
        start = move(m.prepare_start_tokens(lens, rope, ids))
        with t.amp.autocast('cuda', enabled=True, dtype=t.bfloat16):
            out = m.generate_text(past_key_values=cache, max_length=self.args.max_new_tokens,
                do_sample=self.args.temperature > 0, temperature=max(self.args.temperature, 1e-6),
                end_token_id=ids['eos_token_id'], **start)
        tokens = out[:, 0]
        raw = self.tokenizer.decode(tokens)
        if '<|im_start|>' in raw:
            raw = raw.split('<|im_start|>', 1)[1]
        raw = raw.split('<|im_end|>', 1)[0].strip()
        eos = ids['eos_token_id']
        ended = bool(tokens.numel() and int(tokens[-1]) == int(eos))
        return dict(raw_output=raw, generated_tokens=int(tokens.numel()),
                    truncated=tokens.numel() >= self.args.max_new_tokens and not ended)
