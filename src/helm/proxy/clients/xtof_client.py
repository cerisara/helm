
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BloomConfig
import torch

from typing import List, Dict

from helm.common.cache import CacheConfig
from helm.common.request import wrap_request_time, Request, RequestResult, Sequence, Token
from helm.proxy.tokenizers.simple_tokenizer import SimpleTokenizer
from helm.proxy.tokenizers.tokenizer import Tokenizer
from .client import CachingClient


class XtofClient(CachingClient):

    def __init__(self, tokenizer: Tokenizer, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config, tokenizer=tokenizer)
        self.mod = AutoModelForCausalLM.from_pretrained('bigscience/bloomz-560m')
        self.tok = AutoTokenizer.from_pretrained('bigscience/bloomz-560m')

    def make_request(self, request: Request) -> RequestResult:
        print("detson",request)
        raw_request = {
            "engine": request.model_engine,
            "prompt": request.prompt,
            "n": request.num_completions,
        }

        if request.model_engine == "bloomz":

            def do_it():
                return self.invoke_model2(raw_request,request)

            cache_key = CachingClient.make_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            completions = [
                Sequence(
                    text=text,
                    logprob=logprob,
                    tokens=[Token(text=text, logprob=logprob, top_logprobs=response["completions"])],
                )
                for text, logprob in response["completions"].items()
            ]
        else:
            raise ValueError(f"Invalid model: {request.model}")

        return RequestResult(
            success=True,
            cached=False,
            request_time=0,
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )

    def invoke_model2(self, raw_request: Dict, req) -> Dict:
        prompt_tokens = self.tok.encode(raw_request["prompt"])
        print("detsontoks",prompt_tokens)
        rep = self.mod.generate(prompt_tokens, max_new_tokens = req["max_tokens"])
        print("detsonres",rep)
        srep = self.tok.decode(rep)
        print("detsonout",srep)
        response = {"completions": srep}
        return response

    def invoke_model1(self, raw_request: Dict) -> Dict:
        """
        Example: 7 2 4 6
        Completions (num_completions = 3):
        - 6
        - 4
        - 2
        """
        prompt_tokens: List[str] = SimpleTokenizer.tokenize_by_space(raw_request["prompt"])
        choices = reversed(prompt_tokens[-raw_request["n"] :])
        response = {"completions": dict((text, -i) for i, text in enumerate(choices))}
        return response

# convert2myblock(mod)

# Use in HELM
#
# cp this file into /home/xtof/git/github/helm/src/helm/proxy/clients/xtof_client.py
# cd /home/xtof/git/github/helm
# pip install -U .
# come back here... see:
# ./helm.sh

