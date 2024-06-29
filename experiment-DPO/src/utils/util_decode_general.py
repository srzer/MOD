import numpy as np
import torch
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import sys
sys.path.append(".")
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.utils import GreedySearchDecoderOnlyOutput, GreedySearchEncoderDecoderOutput, GenerationMixin, SampleDecoderOnlyOutput, SampleEncoderDecoderOutput, BeamSearchDecoderOnlyOutput, BeamSearchEncoderDecoderOutput, BeamSampleDecoderOnlyOutput, BeamSampleEncoderDecoderOutput, ContrastiveSearchDecoderOnlyOutput, ContrastiveSearchEncoderDecoderOutput, BeamSampleDecoderOnlyOutput, BeamSampleEncoderDecoderOutput, GenerationMode, BeamSampleOutput, ContrastiveSearchOutput, GenerateOutput
from transformers.generation.configuration_utils import GenerationConfig
GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput]
SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]
BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]
BeamSampleOutput = Union[BeamSampleEncoderDecoderOutput, BeamSampleDecoderOnlyOutput]
ContrastiveSearchOutput = Union[ContrastiveSearchEncoderDecoderOutput, ContrastiveSearchDecoderOnlyOutput]
GenerateOutput = Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, ContrastiveSearchOutput]
import torch.distributed as dist
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
import copy, inspect
# from multi_object_decoding.utils import (
#     load_hf_lm_and_tokenizer,
# ) 

class FusionModel(nn.Module):
    def __init__(self, models, weights):
        super(FusionModel, self).__init__()
        self.models = models
        self.weights = weights
        self.num_models = len(models)
        assert self.num_models == len(weights)
        assert self.num_models > 0
        self.config = models[0].config
      
    def can_generate(self):
        return True
      
    def __call__(self, past_key_values, **model_inputs):
        outputs = [None]*self.num_models
        for idx in range(self.num_models):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            for key, value in model_inputs.items():
                if isinstance(value, torch.Tensor):
                    model_inputs[key] = value.to(self.models[idx].device)
            with torch.no_grad():
                outputs[idx] = self.models[idx](past_key_values=past_key_value, **model_inputs)
            past_key_value = None
            for key, value in outputs[idx].items():
                if isinstance(value, torch.Tensor):
                    outputs[idx][key] = value.to(self.models[0].device)
        output = outputs[0]
        min_vocab_size = min([model.config.vocab_size for model in self.models])
        output.logits = torch.sum(torch.stack([self.weights[model_idx]*outputs[model_idx].logits[:,:,:min_vocab_size] for model_idx in range(self.num_models)]), dim=0)
        output.past_key_values = [outputs[model_idx].past_key_values for model_idx in range(self.num_models)]
        outputs = None 
        past_key_values = None
        return output

    def forward(self, **model_inputs):
        raise NotImplementedError
    
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
       
        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self.models[0]._validate_model_class()

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
            # two conditions must be met
            # 1) the generation config must have been created from the model config (`_from_model_config` field);
            # 2) the generation config must have seen no modification since its creation (the hash is the same).
            if self.models[0].generation_config._from_model_config and self.models[0].generation_config._original_object_hash == hash(
                self.models[0].generation_config
            ):
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.models[0].generation_config:
                    self.models[0].generation_config = new_generation_config
            generation_config = self.models[0].generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        self.models[0]._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            generation_config.pad_token_id = eos_token_id

        # 3. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self.models[0]._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        # 4. Define other model kwargs
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.models[0].forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self.models[0]._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length
        self.models[0]._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. determine generation mode
        generation_mode = self.models[0]._get_generation_mode(generation_config, assistant_model)

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        # 8. prepare distribution pre_processing samplers
        logits_processor = self.models[0]._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        # 9. prepare stopping criteria
        stopping_criteria = self.models[0]._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        
        if generation_mode == GenerationMode.GREEDY_SEARCH:
            # 11. run greedy search
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        else: raise NotImplementedError
    
    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        print("Hi! Greedy decoding...")
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

        this_peer_finished = False  # used by synced_gpus only
        logp = None
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.models[0].prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)
            vocab_size = next_tokens_scores.shape[-1]
            if logp is not None:
                next_tokens_scores = next_tokens_scores + logp.unsqueeze(1).repeat(1, vocab_size)
            
            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            new_logp = torch.gather(next_tokens_scores, 1, next_tokens.view(-1, 1)).view(-1)
            logp = new_logp

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self.models[0]._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True
        torch.cuda.empty_cache()
        return input_ids
      
# def load_fusion_model(
#         model_names_list, 
#         weights_list, 
#         tokenizer_name_or_path,
#         device_map,
#         use_slow_tokenizer) -> Tuple[FusionModel, Any]:
#     models = []
#     tokenizer = None
#     for idx, model_name_or_path in enumerate(model_names_list):
#         model, tokenizer = load_hf_lm_and_tokenizer(
#                 model_name_or_path=model_name_or_path, 
#                 tokenizer_name_or_path=tokenizer_name_or_path, 
#                 load_in_8bit=False,
#                 device_map=f'cuda:{device_map[idx]}',
#                 gptq_model=False,
#                 use_fast_tokenizer=not use_slow_tokenizer,
#             )
#         print(f"model {model_name_or_path} has vocab size {model.config.vocab_size}")
#         models.append(model)
#     return FusionModel(models, weights_list), tokenizer