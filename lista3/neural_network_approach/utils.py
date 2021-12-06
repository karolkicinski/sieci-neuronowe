

def process_layers(input_data_size, layers, weights_init_class):
    processed_layers = []
    last_input = input_data_size

    for layer in layers:
        layer.input_size = last_input
        last_input = layer.layer_size
        layer.weights_init_approach = weights_init_class(
            sizes=(layer.input_size, layer.layer_size)
        )
        layer.init()
        processed_layers.append(layer)
    return processed_layers
