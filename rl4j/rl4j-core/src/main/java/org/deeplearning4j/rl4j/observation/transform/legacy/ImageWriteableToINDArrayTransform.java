package org.deeplearning4j.rl4j.observation.transform.legacy;

import org.datavec.api.transform.Operation;
import org.datavec.image.data.ImageWritable;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

public class ImageWriteableToINDArrayTransform implements Operation<ImageWritable, INDArray> {

    private final int height;
    private final int width;
    private final NativeImageLoader loader;

    public ImageWriteableToINDArrayTransform(int height, int width) {
        this.height = height;
        this.width = width;
        this.loader = new NativeImageLoader(height, width);
    }

    @Override
    public INDArray transform(ImageWritable imageWritable) {
        INDArray out = null;
        try {
            out = loader.asMatrix(imageWritable);
        } catch (IOException e) {
            e.printStackTrace();
        }
        out = out.reshape(1, height, width);
        INDArray compressed = out.castTo(DataType.UINT8);
        return compressed;
    }
}
