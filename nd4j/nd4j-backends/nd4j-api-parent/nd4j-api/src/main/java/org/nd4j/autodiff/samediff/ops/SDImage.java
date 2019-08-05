package org.nd4j.autodiff.samediff.ops;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.image.CropAndResize;

/**
 *
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
     * @param image              Input image
     * @param cropBoxes          Float32 crop, shape [numBoxes, 4] with values in range 0 to 1
     * @param boxIndices         Indices: which image (index to dimension 0) the cropBoxes belong to. Rank 1, shape [numBoxes]
     * @param cropOutSize        Output size for the images - int32, rank 1 with values [outHeight, outWidth]
     * @param method             Image resize method
     * @param extrapolationValue Used for extrapolation, when applicable. 0.0 should be used for the default
     * @return Cropped and resized images
     */
    public SDVariable cropAndResize(String name, SDVariable image, SDVariable cropBoxes, SDVariable boxIndices, SDVariable cropOutSize,
                                    CropAndResize.Method method, double extrapolationValue){
        SDVariable out = new CropAndResize(sd, image, cropBoxes, boxIndices, cropOutSize, method, extrapolationValue).outputVariable();
        return updateVariableNameAndReference(out, name);
    }


}
