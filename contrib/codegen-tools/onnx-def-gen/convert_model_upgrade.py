import onnx
from onnx import version_converter, helper, ModelProto


# Referenced from: https://github.com/onnx/onnx/issues/2660#issuecomment-605874784
def add_value_info_for_constants(model : onnx.ModelProto):
    """
    Currently onnx.shape_inference doesn't use the shape of initializers, so add
    that info explicitly as ValueInfoProtos.
    Mutates the model.
    Args:
        model: The ModelProto to update.
    """
    # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
    if model.ir_version < 4:
        return

    def add_const_value_infos_to_graph(graph : onnx.GraphProto):
        inputs = {i.name for i in graph.input}
        existing_info = {vi.name: vi for vi in graph.value_info}
        for init in graph.initializer:
            # Check it really is a constant, not an input
            if init.name in inputs:
                continue

            # The details we want to add
            elem_type = init.data_type
            shape = init.dims

            # Get existing or create new value info for this constant
            vi = existing_info.get(init.name)
            if vi is None:
                vi = graph.value_info.add()
                vi.name = init.name

            # Even though it would be weird, we will not overwrite info even if it doesn't match
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.UNDEFINED:
                tt.elem_type = elem_type
            if not tt.HasField("shape"):
                # Ensure we set an empty list if the const is scalar (zero dims)
                tt.shape.dim.extend([])
                for dim in shape:
                    tt.shape.dim.add().dim_value = dim
            graph_input = graph.input.add()
            graph_input.name = vi.name
            graph_input.type.tensor_type.elem_type = elem_type


        # Handle subgraphs
        for node in graph.node:
            for attr in node.attribute:
                # Ref attrs refer to other attrs, so we don't need to do anything
                if attr.ref_attr_name != "":
                    continue

                if attr.type == onnx.AttributeProto.GRAPH:
                    add_const_value_infos_to_graph(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        add_const_value_infos_to_graph(g)


    return add_const_value_infos_to_graph(model.graph)

def summarize_model(input: ModelProto):
    return f'Inputs {len(input.graph.input)} Nodes {len(input.graph.node)} Initializer {len(input.graph.initializer)} Value info {len(input.graph.value_info)}'

model = onnx.load('C:\\Users\\agibs\\Downloads\\V9\\V9\\best_bracket.onnx')
kotlin_model = onnx.load('C:\\Users\\agibs\\Documents\\GitHub\\dl4j-PR-split\\deeplearning4j\\nd4j\\samediff-import\\samediff-import-onnx\\input-adjusted-model.onnx')
input_names_2 = [node.name for node in kotlin_model.graph.node]
input_init__names_2 = [initializer.name for initializer in kotlin_model.graph.initializer]
#model = onnx.shape_inference.infer_shapes(model)
add_value_info_for_constants(model)
input_names = [node.name for node in model.graph.node]
input_init__names = [initializer.name for initializer in model.graph.initializer]
input_val_info__names = [value_info.name for value_info in model.graph.value_info]
converted_model = version_converter.convert_version(kotlin_model, 13)
converted_input_val_info__names = [value_info.name for value_info in converted_model.graph.value_info]
converted_node_names = [node.name for node in converted_model.graph.node]
onnx.save(converted_model,'output.onnx')
print('Converted model')