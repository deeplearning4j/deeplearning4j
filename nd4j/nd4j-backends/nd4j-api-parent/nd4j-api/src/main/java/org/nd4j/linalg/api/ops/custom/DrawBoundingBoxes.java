package org.nd4j.linalg.api.ops.custom;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

public class DrawBoundingBoxes extends DynamicCustomOp {
    public DrawBoundingBoxes() {}

    public DrawBoundingBoxes(INDArray images, INDArray boxes, INDArray colors,
                             INDArray output) {
        inputArguments.add(images);
        inputArguments.add(boxes);
        inputArguments.add(colors);
        outputArguments.add(output);
    }

    @Override
    public String opName() {
        return "draw_bounding_boxes";
    }
}