package org.nd4j.linalg.api.ops.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

public class DrawBoundingBoxes extends DynamicCustomOp {
    public DrawBoundingBoxes() {}

    public DrawBoundingBoxes(INDArray images, INDArray boxes, INDArray colors,
                             INDArray output) {
        inputArguments.add(images);
        inputArguments.add(boxes);
        inputArguments.add(colors);
        outputArguments.add(output);
    }

    public DrawBoundingBoxes(SameDiff sameDiff, SDVariable boxes, SDVariable colors) {
        super("", sameDiff, new SDVariable[]{boxes, colors});
    }

    @Override
    public String opName() {
        return "draw_bounding_boxes";
    }

    @Override
    public String[] tensorflowNames() {
        return new String[]{"DrawBoundingBoxes", "DrawBoundingBoxesV2"};
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        int n = args().length;
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == n, "Expected %s input data types for %s, got %s", n, getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}