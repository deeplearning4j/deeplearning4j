package org.nd4j.imports.descriptors.onnx;

import lombok.Data;

import java.util.List;

@Data
public class OnnxDescriptor {
    private List<OpDescriptor> descriptors;
}
