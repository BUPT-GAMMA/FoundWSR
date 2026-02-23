from .MoCo import MoCo
from .. import register_model
from ..base_model import BaseModel

@register_model("MoCo_IQFormer")
class MoCo_IQFormer(MoCo, BaseModel):
    @classmethod
    def build_model_from_args(cls, args):
        
        return cls(args.dim,
                   args.mlp_dim,
                   args.T)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        # print("-----------------------------------")
        # for name, param in self.base_encoder.named_parameters():
        #     print(f"Parameter: {name}, Shape: {param.shape}")
        # print("-----------------------------------")
        if hasattr(self.base_encoder, 'head'):
            hidden_dim = self.base_encoder.head.weight.shape[1]
        else:
            raise ValueError("self.base_encoder.head does not exist")
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        # input_dim: hidden_dim, output_dim: dim
        self.base_encoder.head = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)