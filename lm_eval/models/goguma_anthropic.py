import logging
import os
from functools import cached_property
from typing import Any, Dict, List, Tuple, Union
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from dotenv import load_dotenv
import time
import requests

from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import LocalCompletionsAPI
from lm_eval.models.utils import handle_stop_sequences, retry_on_specific_exceptions


eval_logger = logging.getLogger(__name__)
load_dotenv()



@register_model("goguma-anthropic-model", "goguma-anthropic-completions")
class GogumaModelAPI(LocalCompletionsAPI):
    def __init__(
        self,
        base_url=os.getenv('GOGUMA_MODEL_API_ENDPOINT'),  # Replace with your actual endpoint
        tokenizer_backend=None,
        **kwargs,
    ):
        super().__init__(
            base_url=base_url, tokenizer_backend=tokenizer_backend, **kwargs
        )
        eval_logger.warning(
            "Custom model API does not support batching. Defaulting to batch size 1."
        )
        self._batch_size = 1
        eval_logger.warning(
            "Custom model API does not support loglikelihoods."
        )
        self._last_request_time = 0
        self._min_interval = 20  # seconds (3 requests/min = 20s/request)

    @retry(
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(5),
        reraise=True,
    )  
    def model_call(self, messages, *, generate=True, gen_kwargs=None, **kwargs):
      MAX_RETRIES = 5
      BASE_WAIT = 1  # seconds
      MAX_WAIT = 60   # cap the wait time

      retries = 0
      total_wait = 0
      start_time = time.time()

      # Respect rate limit
      now = time.time()
      elapsed = now - self._last_request_time
      if elapsed < self._min_interval:
          sleep_time = self._min_interval - elapsed
          eval_logger.info(f"Sleeping for {sleep_time:.2f}s to respect rate limit")
          time.sleep(sleep_time)
          total_wait_time += sleep_time

      while retries < MAX_RETRIES:
          try:
              payload = self._create_payload(
                  self.create_message(messages),
                  generate=generate,
                  gen_kwargs=gen_kwargs,
                  seed=self._seed,
                  eos=self.eos_string,
                  **kwargs,
              )

              response = requests.post(
                  self.base_url,
                  json=payload,
                  headers=self.header,
                  verify=self.verify_certificate,
                  timeout=self.timeout,
              )

              if not response.ok:
                  logging.warning(f"API request failed ({response.status_code}): {response.text}")
                  response.raise_for_status()

              response.raise_for_status()
              print("RESPONSE: ",)
              self._last_request_time = time.time()
              duration = self._last_request_time - start_time
              resp = response.json()
              query_complexity = resp['metadata']['processing_path']
              return {
                  "response": response.json(),
                  "query_complexity": query_complexity,
              }

          except requests.exceptions.RequestException as e:
              wait_time = min(BASE_WAIT * (2 ** retries), MAX_WAIT)
              logging.warning(f"Request failed: {e}. Retrying in {wait_time}s...")
              time.sleep(wait_time)
              total_wait += wait_time
              retries += 1

      raise RuntimeError(f"Failed after {MAX_RETRIES} retries and {total_wait:.1f}s of wait.")

    @cached_property
    def api_key(self):
        """Override this property to return the API key for your custom API."""
        #key = os.environ.get("CUSTOM_MODEL_API_KEY", None)  # Use your own env var name
        #if key is None:
            #raise ValueError(
                #"API key not found. Please set the CUSTOM_MODEL_API_KEY environment variable."
            #)
        #return key
    
    @cached_property
    def header(self):
        """Customize headers for your API - adjust as needed"""
        return {
            #"Authorization": f"Bearer {self.api_key}",  # Or whatever auth format your API uses
            "Content-Type": "application/json",
        }

    def _create_payload(
        self,
        messages: List[Dict],
        generate=True,
        gen_kwargs: dict = None,
        eos="\n\nHuman:",
        **kwargs,
    ) -> dict:
        """
        Create payload for your /api/chat endpoint.
        """
        if gen_kwargs is None:
            gen_kwargs = {}
        
        # Get the latest user message (last message in the conversation)
        # Your API expects a single "message" field, not a conversation history
        user_message = ""
        print('MESSAGE: ',messages)
        if messages:
            # Find the last user message
            user_message = messages
        
        # Extract generation parameters if your API supports them
        # You may need to pass these in the context or remove them entirely
        # depending on your API's capabilities
        gen_kwargs.pop("do_sample", False)
        max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", ["\n\nHuman:"]), eos=eos)
        
        # Create context from conversation history and generation parameters
        context = {
            "conversation_history": messages,  # Include full conversation for context
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop_sequences": stop if isinstance(stop, list) else [stop] if stop else [],
            **gen_kwargs,  # Any other generation parameters
        }
        
        # Create payload matching your API format
        payload = {
            "message": user_message,
            "session_id": kwargs.get("session_id", None),
            "conversation_id": kwargs.get("conversation_id", None),
            "new_conversation": kwargs.get("new_conversation", False),
            "context": context,
        }
        
        # Remove None values to clean up the payload
        payload = {k: v for k, v in payload.items() if v is not None}
        
        return payload
    

    def parse_generations(
        self, outputs: Union[Dict, List[Dict]], **kwargs
    ) -> List[str]:
        """
        Parse the response from your /api/chat endpoint.
        Adjust this based on your API's actual response format.
        """
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        
        for out in outputs:
            # Common response formats for chat APIs:
            
            if "response" in out:
                # Format: {"response": "AI response text"}
                res.append(out["response"])
            else:
                # Fallback: convert entire response to string
                # You should replace this with your actual response format
                eval_logger.warning(f"Unknown response format: {out}")
                res.append(str(out))
        
        return res
    
    def tok_encode(
        self,
        string: str,
        left_truncate_len=None,
        add_special_tokens=None,
        **kwargs,
    ) -> List[str]:
        """
        Since you don't have access to a tokenizer, return the string as-is.
        This is a simple fallback that treats the entire string as one "token".
        """
        return [string]
    
    def loglikelihood(self, requests, **kwargs):
        """
        Since your API doesn't support loglikelihoods, raise an error.
        """
        raise NotImplementedError(
            "Custom model API does not support the return of loglikelihood"
        )
    
    def loglikelihood_rolling(self, requests, **kwargs):
        """
        Since your API doesn't support loglikelihoods, raise an error.
        """
        raise NotImplementedError(
            "Custom model API does not support rolling loglikelihood"
        )