import torch
import torch.nn as nn
from copy import deepcopy
from collections import OrderedDict


class DualNetwork(nn.Module):
  def __init__(self, network):
    super(DualNetwork,self).__init__()
    # Add comment
    self.dual_network = deepcopy(network)

  def forward(self,inputs):
    return  self.dual_network(inputs)


class LinearDualNetwork(DualNetwork):
  """
  Outer layer is the layer that gives the features phi(Z),
  by default none, then just takes the layer before the last.
  """
  def __init__(self, network, 
                    input_dim,
                    output_dim,
                    output_layer = None):
    super(LinearDualNetwork,self).__init__(network)
    
    # Get all module names
    module_keys = list(network._modules.keys())
    # If output_layer is specified, use it
    if output_layer:
      layer_to_hook = output_layer
    # If not specified and there are at least 2 layers, use the second-to-last layer
    elif len(module_keys) >= 2:
      layer_to_hook = module_keys[-2]  # Second-to-last layer
    # If only one layer, use it
    elif len(module_keys) == 1:
      layer_to_hook = module_keys[0]
    else:
      # No modules to hook, this is an error condition
      raise ValueError("Model has no modules to hook")

    dummpy_param = next(network.parameters())
    device= dummpy_param.device
    dtype = dummpy_param.dtype

    # Create the hook model
    self.model_with_hook = ModelWithHookAuto(network, layer_to_hook)
    self.out_shape = torch.Size([output_dim])
    self.linear = torch.nn.Linear(input_dim+1, output_dim, bias=False, 
                                device=device, 
                                dtype=dtype)

  def forward(self, inputs,with_features=False):
    self.model_with_hook.eval()
    with torch.no_grad():
      out, selected_out = self.model_with_hook(inputs)
      selected_out = selected_out.detach().flatten(start_dim=1)
      ones = torch.ones(selected_out.shape[0],1, 
                        dtype=selected_out.dtype,
                        device=selected_out.device)
      selected_out = torch.cat((selected_out,ones),dim=1)
    out = self.linear(selected_out)
    out = torch.unflatten(out, dim=1, sizes=self.out_shape)
    if with_features:
      return out, selected_out 
    else:
      return out

  def parameters(self):
      return (self.linear.parameters()) 

class ModelWithHookAuto(nn.Module):
    def __init__(self, network, output_layer=None):
        super(ModelWithHookAuto, self).__init__()
        self.selected_out = None
        # PRETRAINED MODEL
        self.model = network
        self.fhooks = []
        
        # Get all module names
        module_keys = list(self.model._modules.keys())
        
        # If output_layer is specified, use it
        if output_layer:
            layer_to_hook = output_layer
        # If not specified and there are at least 2 layers, use the second-to-last layer
        elif len(module_keys) >= 2:
            layer_to_hook = module_keys[-2]  # Second-to-last layer
        # If only one layer, use it
        elif len(module_keys) == 1:
            layer_to_hook = module_keys[0]
        else:
            # No modules to hook, this is an error condition
            raise ValueError("Model has no modules to hook")
        
        # Register the hook
        if hasattr(self.model, layer_to_hook):
            self.fhooks.append(
                getattr(self.model, layer_to_hook).register_forward_hook(
                    self.forward_hook(layer_to_hook)
                )
            )
            print(f"Successfully hooked layer: {layer_to_hook}", file=sys.stderr)
        else:
            # This shouldn't happen given the logic above, but just in case
            raise ValueError(f"Layer {layer_to_hook} not found in model")
    
    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out = output
        return hook
    
    def forward(self, x):
        out = self.model(x)
        return out, self.selected_out
    
    def __del__(self):
        # Clean up hooks when the model is deleted
        for hook in self.fhooks:
            hook.remove()

class ModelWithHook(nn.Module):
    def __init__(self, output_layers, model):
        super(ModelWithHook, self).__init__()
        self.output_layers = output_layers
        self.selected_out = None
        #PRETRAINED MODEL
        self.model = model
        self.fhooks = []

        for i,l in enumerate(list(self.model._modules.keys())):
            if l==self.output_layers:
                self.fhooks.append(getattr(self.model,l).register_forward_hook(self.forward_hook(l)))
    
    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.selected_out = output
        return hook

    def forward(self, x):
        out = self.model(x)
        return out, self.selected_out

class ModelWithHookAuto(nn.Module):
    def __init__(self, network, output_layers=None):
        super(ModelWithHookAuto, self).__init__()
        self.selected_out = None
        # PRETRAINED MODEL
        self.model = network
        self.fhooks = []
        
        # Get all module names
        module_keys = list(self.model._modules.keys())
        
        # If output_layers is specified, use it
        if output_layers:
            layer_to_hook = output_layers
        # If not specified and there are at least 2 layers, use the second-to-last layer
        elif len(module_keys) >= 2:
            layer_to_hook = module_keys[-2]  # Second-to-last layer
        # If only one layer, use it
        elif len(module_keys) == 1:
            layer_to_hook = module_keys[0]
        else:
            # No modules to hook, this is an error condition
            raise ValueError("Model has no modules to hook")
        
        # Register the hook
        if hasattr(self.model, layer_to_hook):
            self.fhooks.append(
                getattr(self.model, layer_to_hook).register_forward_hook(
                    self.forward_hook(layer_to_hook)
                )
            )
            print(f"Successfully hooked layer: {layer_to_hook}")
        else:
            # This shouldn't happen given the logic above, but just in case
            raise ValueError(f"Layer {layer_to_hook} not found in model")
    
    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out = output
        return hook
    
    def forward(self, x):
        out = self.model(x)
        return out, self.selected_out
    
    def __del__(self):
        # Clean up hooks when the model is deleted
        for hook in self.fhooks:
            hook.remove()