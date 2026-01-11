import json
import os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler


class TextPairScoreHandler(BaseHandler):
    def initialize(self, context) -> None:
        self.context = context
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = context.system_properties.get("model_dir")
        model_base = os.path.join(model_dir, "model_files")
        if not os.path.isdir(model_base):
            model_base = model_dir

        self.tokenizer = AutoTokenizer.from_pretrained(model_base)
        self.model = AutoModel.from_pretrained(model_base)
        state_path = os.path.join(model_dir, "model.pt")
        if os.path.exists(state_path):
            state_dict = torch.load(state_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=True)

        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, data: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        for item in data:
            payload = item.get("body") if isinstance(item, dict) else item
            if payload is None:
                payload = item.get("data")
            payload = self._parse_payload(payload)
            pairs.extend(self._extract_pairs(payload))
        if not pairs:
            raise ValueError("No valid query/doc pairs provided.")
        return pairs

    def inference(self, pairs: List[Tuple[str, str]], *args, **kwargs) -> List[float]:
        queries = [q for q, _ in pairs]
        docs = [d for _, d in pairs]
        q_inputs = self.tokenizer(
            queries, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        d_inputs = self.tokenizer(
            docs, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            q_out = self.model(**q_inputs)
            d_out = self.model(**d_inputs)
            q_emb = self._mean_pool(q_out, q_inputs["attention_mask"])
            d_emb = self._mean_pool(d_out, d_inputs["attention_mask"])
            q_emb = F.normalize(q_emb, p=2, dim=1)
            d_emb = F.normalize(d_emb, p=2, dim=1)
            scores = (q_emb * d_emb).sum(dim=1)
        return scores.cpu().tolist()

    def postprocess(self, scores: List[float]) -> List[Dict[str, float]]:
        return [{"score": float(score)} for score in scores]

    def _mean_pool(self, model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def _parse_payload(self, payload: Any) -> Any:
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8")
        if isinstance(payload, str):
            return json.loads(payload)
        return payload

    def _extract_pairs(self, payload: Any) -> List[Tuple[str, str]]:
        if payload is None:
            return []
        if isinstance(payload, dict):
            if "instances" in payload:
                return self._extract_pairs(payload["instances"])
            if "query" in payload and "doc" in payload:
                return [(str(payload["query"]), str(payload["doc"]))]
        if isinstance(payload, list):
            pairs: List[Tuple[str, str]] = []
            for item in payload:
                pairs.extend(self._extract_pairs(item))
            return pairs
        return []
