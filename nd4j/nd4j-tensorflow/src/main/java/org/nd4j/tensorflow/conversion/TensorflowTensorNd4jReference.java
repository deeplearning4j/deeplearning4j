package org.nd4j.tensorflow.conversion;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.tensorflow;
import org.nd4j.linalg.api.ndarray.INDArray;

@Data
@Builder
@AllArgsConstructor
public class TensorflowTensorNd4jReference {
    private INDArray ndarray;
    private tensorflow.TF_Tensor tensor;
    private Pointer nd4jPointer;
    private LongPointer tfShapePointer;

}
