from kolmarn.models import KANSequential
from kolmarn.visualize import plot_feature_splines

model = KANSequential(1, [32, 1], num_basis=16)

# model.load_state_dict(torch.load("kan_sin_model.pt"))

plot_feature_splines(model, feature_index=0, layer_index=0)
