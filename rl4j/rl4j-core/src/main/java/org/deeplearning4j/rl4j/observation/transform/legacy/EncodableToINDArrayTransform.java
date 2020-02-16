package org.deeplearning4j.rl4j.observation.transform.legacy;

import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.datavec.api.transform.Operation;
import org.datavec.image.data.ImageWritable;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.bytedeco.opencv.global.opencv_core.CV_32FC;

public class EncodableToINDArrayTransform implements Operation<Encodable, INDArray> {

    private final int[] shape;

    public EncodableToINDArrayTransform(int[] shape) {
        this.shape = shape;
    }

    @Override
    public INDArray transform(Encodable encodable) {
        return Nd4j.create(encodable.toArray()).reshape(shape);
    }

}
