package org.nd4j.linalg.dataset.api.preprocessor.serializer;

import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.*;

/**
 * {@link NormalizerSerializerStrategy}
 * for {@link ImagePreProcessingScaler}
 *
 * Saves the min range, max range, and max pixel value as
 * doubles
 *
 *
 * @author Adam Gibson
 */
public class ImagePreProcessingSerializerStrategy implements NormalizerSerializerStrategy<ImagePreProcessingScaler> {
    @Override
    public void write(ImagePreProcessingScaler normalizer, OutputStream stream) throws IOException {
        try(DataOutputStream dataOutputStream = new DataOutputStream(stream)) {
            dataOutputStream.writeDouble(normalizer.getMinRange());
            dataOutputStream.writeDouble(normalizer.getMaxRange());
            dataOutputStream.writeDouble(normalizer.getMaxPixelVal());
            dataOutputStream.flush();
        }
    }

    @Override
    public ImagePreProcessingScaler restore(InputStream stream) throws IOException {
        DataInputStream dataOutputStream = new DataInputStream(stream);
        double minRange = dataOutputStream.readDouble();
        double maxRange = dataOutputStream.readDouble();
        double maxPixelVal = dataOutputStream.readDouble();
        ImagePreProcessingScaler ret =  new ImagePreProcessingScaler(minRange,maxRange);
        ret.setMaxPixelVal(maxPixelVal);
        return ret;
    }

    @Override
    public NormalizerType getSupportedType() {
        return NormalizerType.IMAGE_MIN_MAX;
    }
}
