def save_parameters(model):
    """Extract a copy of state dict from a given model

    Parameters
    ----------
        model : nn.Module
            A PyTorch nn model.

    Returns
    -------
        dictionary : dict
            A state dict of the model parameters    
    """
    return model.state_dict().copy()


def restore_original_parameters(original_model_dict, pruned_model_dict):
    """Copies the parameters from a state dict to another model's state
    dict, excluding the pruning mask.

    Parameters
    ----------
        original_model_dict : dict
            A state dictionary of a model

        pruned_model_dict : dict
            A state dict from a pruned model to load the parameters into.
    """
    for module_name in pruned_model_dict:
        if 'mask' not in module_name:            
            pruned_model_dict[module_name] = original_model_dict[module_name.split('_')[0]]