package org.nd4j.linalg.api.ops.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
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

    public DrawBoundingBoxes(SameDiff sameDiff, SDVariable boxes, SDVariable colors) {
        super("", sameDiff, new SDVariable[]{boxes, colors});
    }

    @Override
    public String opName() {
        return "draw_bounding_boxes";
    }

    @Override
    public String tensorflowName() {
        return "DrawBoundingBoxes";
    }
}