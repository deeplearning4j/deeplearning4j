package org.nd4j.autodiff.samediff.ops;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.image.CropAndResize;
import org.nd4j.linalg.api.ops.impl.image.ExtractImagePatches;
import org.nd4j.linalg.api.ops.impl.image.NonMaxSuppression;

/**
 * @author Alex Black
 */
public class SDImage extends SDOps {
    public SDImage(SameDiff sameDiff) {
        super(sameDiff);
    }

    /**
     * Given an input image and some crop boxes, extract out the image subsets and resize them to the specified size.
     *
     * @param name               May be null. Name for the output variable.
     * @param image              Input image, with shape [batch, height, width, channels]
     * @param cropBoxes          Float32 crop, shape [numBoxes, 4] with values in range 0 to 1
     * @param boxIndices         Indices: which image (index to dimension 0) the cropBoxes belong to. Rank 1, shape [numBoxes]
     * @param cropOutSize        Output size for the images - int32, rank 1 with values [outHeight, outWidth]
     * @param method             Image resize method
     * @param extrapolationValue Used for extrapolation, when applicable. 0.0 should be used for the default
     * @return Cropped and resized images
     */
    public SDVariable cropAndResize(String name, SDVariable image, SDVariable cropBoxes, SDVariable boxIndices, SDVariable cropOutSize,
                                    CropAndResize.Method method, double extrapolationValue) {
        SDVariable out = new CropAndResize(sd, image, cropBoxes, boxIndices, cropOutSize, method, extrapolationValue).outputVariable();
        return updateVariableNameAndReference(out, name);
    }

    /**
     * Given an input image, extract out image patches (of size kSizes - h x w) and place them in the depth dimension.
     *
     * @param name     Map be null. Name for the output variable
     * @param image    Input image to extract image patches from - shape [batch, height, width, channels]
     * @param kSizes   Kernel size - size of the image patches, [height, width]
     * @param strides  Stride in the input dimension for extracting image patches, [stride_height, stride_width]
     * @param rates    Usually [1,1]. Equivalent to dilation rate in dilated convolutions - how far apart the output pixels
     *                 in the patches should be, in the input. A dilation of [a,b] means every {@code a}th pixel is taken
     *                 along the height/rows dimension, and every {@code b}th pixel is take along the width/columns dimension
     * @param sameMode Padding algorithm. If true: use Same padding
     * @return The extracted image patches
     */
    public SDVariable extractImagePatches(String name, SDVariable image, @NonNull int[] kSizes,
                                          @NonNull int[] strides, @NonNull int[] rates, boolean sameMode) {
        SDVariable out = new ExtractImagePatches(sd, image, kSizes, strides, rates, sameMode).outputVariable();
        return updateVariableNameAndReference(out, name);
    }


    public SDVariable nonMaxSuppression(String name, @NonNull SDVariable boxes, @NonNull SDVariable scores, @NonNull SDVariable maxOutSize,
                                        @NonNull SDVariable iouThreshold, @NonNull SDVariable scoreThreshold){
        SDVariable out = new NonMaxSuppression(sd, boxes, scores, maxOutSize, iouThreshold, scoreThreshold).outputVariable();
        return updateVariableNameAndReference(out, name);
    }
}
