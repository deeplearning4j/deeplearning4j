package org.nd4j.linalg.dataset.api.preprocessor.serializer;

import lombok.NonNull;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;

/**
 * Strategy for saving and restoring {@link NormalizerStandardize} instances in single binary files
 *
 * @author Ede Meijer
 */
public class StandardizeSerializerStrategy implements NormalizerSerializerStrategy<NormalizerStandardize> {
    @Override
    public void write(@NonNull NormalizerStandardize normalizer, @NonNull OutputStream stream) throws IOException {
        try (DataOutputStream dos = new DataOutputStream(stream)) {
            dos.writeBoolean(normalizer.isFitLabel());

            Nd4j.write(normalizer.getMean(), dos);
            Nd4j.write(normalizer.getStd(), dos);

            if (normalizer.isFitLabel()) {
                Nd4j.write(normalizer.getLabelMean(), dos);
                Nd4j.write(normalizer.getLabelStd(), dos);
            }
            dos.flush();
        }
    }

    @Override
    public NormalizerStandardize restore(@NonNull InputStream stream) throws IOException {
        DataInputStream dis = new DataInputStream(stream);

        boolean fitLabels = dis.readBoolean();

        NormalizerStandardize result = new NormalizerStandardize(Nd4j.read(dis), Nd4j.read(dis));
        result.fitLabel(fitLabels);
        if (fitLabels) {
            result.setLabelStats(Nd4j.read(dis), Nd4j.read(dis));
        }

        return result;
    }

    @Override
    public NormalizerType getSupportedType() {
        return NormalizerType.STANDARDIZE;
    }
}
