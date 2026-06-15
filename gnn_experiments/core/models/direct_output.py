"""Fixed-depth input-to-output GNN without intermediate steps."""

import torch

from .decoder import Decoder
from .encoder import Encoder
from .processor import Processor
from ..utils import stack_hidden


class DirectOutputModel(torch.nn.Module):
    """Encode inputs, run K message-passing layers, and decode final outputs."""

    def __init__(self, specs, cfg):
        super().__init__()
        if "randomness" in specs:
            raise ValueError(
                "direct_output does not support time-indexed randomness inputs"
            )
        if cfg.MODEL.MSG_PASSING_STEPS < 1:
            raise ValueError("direct_output requires at least one GNN layer")

        self.cfg = cfg
        self.encoder = Encoder(specs, cfg.MODEL.HIDDEN_DIM)
        self.processor = torch.nn.ModuleList(
            [Processor(cfg) for _ in range(cfg.MODEL.MSG_PASSING_STEPS)]
        )

        decoder_input = (
            cfg.MODEL.HIDDEN_DIM * 3
            if cfg.MODEL.DECODER_USE_LAST_HIDDEN
            else cfg.MODEL.HIDDEN_DIM * 2
        )
        self.decoder = Decoder(specs, decoder_input, no_hint=True)

        first_processor = self.processor[0]
        if first_processor.has_edge_weight():
            self.edge_weight_name = "edge_weight"
        elif first_processor.has_edge_attr():
            self.edge_weight_name = "edge_attr"
        else:
            self.edge_weight_name = None

    def _processor_kwargs(self, batch):
        if self.edge_weight_name is None or not hasattr(batch, "weights"):
            return {}
        weights = batch.weights
        if self.edge_weight_name == "edge_attr":
            if weights.dim() == 1:
                weights = weights.unsqueeze(-1)
            weights = weights.float()
        return {self.edge_weight_name: weights}

    def forward(self, batch):
        input_hidden, _ = self.encoder(batch)
        hidden = input_hidden
        last_hidden = input_hidden
        processor_kwargs = self._processor_kwargs(batch)

        for processor in self.processor:
            last_hidden = hidden
            hidden = processor(
                input_hidden,
                hidden,
                last_hidden,
                batch_assignment=batch.batch,
                edge_index=batch.edge_index,
                **processor_kwargs,
            )

        decoder_hidden = stack_hidden(
            input_hidden,
            hidden,
            last_hidden,
            self.cfg.MODEL.DECODER_USE_LAST_HIDDEN,
        )
        output = self.decoder(decoder_hidden, batch, "outputs")
        return output, {}, hidden
