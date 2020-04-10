package org.nd4j.linalg.api.ops.impl.image;

import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.enums.ImageResizeMethods;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

@NoArgsConstructor
public class ImageResize extends DynamicCustomOp {



    @Override
    public String opName() {
        return "image_resize";
    }


    public ImageResize(@NonNull SameDiff sameDiff, @NonNull SDVariable in, @NonNull SDVariable size, boolean preserveAspectRatio, boolean antialias, ImageResizeMethods method) {
        super("image_resize", sameDiff, new SDVariable[]{in, size});
        Preconditions.checkArgument(in.getArr().shape().length==4,"expected input message in NCHW format i.e [batchSize, channels, height, width]");
        addBArgument(preserveAspectRatio, antialias);
        addIArgument(method.ordinal());
    }

    public ImageResize(@NonNull INDArray in, @NonNull INDArray size, INDArray output, boolean preserveAspectRatio, boolean antialias, ImageResizeMethods method) {
        super("image_resize", new INDArray[]{in, size}, new INDArray[]{output});
        Preconditions.checkArgument(in.shape().length==4,"expected input message in NCHW format i.e [batchSize, channels, height, width]");
        addBArgument(preserveAspectRatio, antialias);
        addIArgument(method.ordinal());
    }



    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        Preconditions
                .checkArgument(dataTypes != null && dataTypes.size() == 2, "Expected exactly 2 input datatypes, got %s", dataTypes);
        Preconditions.checkArgument(dataTypes.get(0).isFPType(), "Input datatype must be floating point, got %s", dataTypes);

        return Collections.singletonList(dataTypes.get(0));
    }


}