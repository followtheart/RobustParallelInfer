from transformers import pipeline, set_seed
import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D


def conv1D_to_linear(conv1D_layer):
    """
    Convert the custom Conv1D layer to an nn.Linear layer.
    """
    assert isinstance(conv1D_layer, Conv1D), "Input layer should be an instance of Conv1D"

    nx, nf = conv1D_layer.weight.shape
    print (conv1D_layer.weight.shape)
    linear_layer = nn.Linear(nx, nf)

    # Copy the weights and biases
    linear_layer.weight.data = conv1D_layer.weight.transpose(0, 1)
    linear_layer.bias.data = conv1D_layer.bias.data

    return linear_layer


def transform_custom_model(model):
    for name, module in model.named_children():
        if isinstance(module, Conv1D):
            setattr(model, name, conv1D_to_linear(module))
        else:
            # Recurse into child modules
            transform_custom_model(module)
    return model



from transformers import GPT2Tokenizer
#generator = pipeline('text-generation', model='gpt2-large')
generator = pipeline('text-generation', model='gpt2')

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # or 'gpt2-large', etc.
set_seed(42)
# The model is an attribute of the generator
model = generator.model

transformed_model = transform_custom_model(model)

generator.model = transform_custom_model(model)

for name, module in generator.model.named_modules():
    print (f"{name}: {module}")
example_input_str = 'Hello, I am an example input.'
#example_input_tokenized = tokenizer.encode(example_input_str, return_tensors='pt')


# Now you can use this to trace the model
#traced_model = torch.jit.trace(generator.model, example_input_tokenized)

# Save the traced model
#traced_model.save("traced_openai_gpt.pt")

# scripted_model = torch.jit.script(generator.model)
# scripted_model.save("scripted_openai_gpt.pt")
result = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)

print (result)
