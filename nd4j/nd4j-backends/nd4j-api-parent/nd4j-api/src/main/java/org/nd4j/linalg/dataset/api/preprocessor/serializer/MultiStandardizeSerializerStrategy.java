package org.nd4j.linalg.dataset.api.preprocessor.serializer;

import lombok.NonNull;
import org.nd4j.linalg.dataset.api.preprocessor.MultiNormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.stats.DistributionStats;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Strategy for saving and restoring {@link MultiNormalizerStandardize} instances in single binary files
 *
 * @author Ede Meijer
 */
public class MultiStandardizeSerializerStrategy implements NormalizerSerializerStrategy<MultiNormalizerStandardize> {
    /**
     * Serialize a MultiNormalizerStandardize to a output stream
     *
     * @param normalizer the normalizer
     * @param stream     the output stream to write to
     * @throws IOException
     */
    public void write(@NonNull MultiNormalizerStandardize normalizer, @NonNull OutputStream stream) throws IOException {
        try (DataOutputStream dos = new DataOutputStream(stream)) {
            dos.writeBoolean(normalizer.isFitLabel());
            dos.writeInt(normalizer.numInputs());
            dos.writeInt(normalizer.isFitLabel() ? normalizer.numOutputs() : -1);

            for (int i = 0; i < normalizer.numInputs(); i++) {
                Nd4j.write(normalizer.getFeatureMean(i), dos);
                Nd4j.write(normalizer.getFeatureStd(i), dos);
            }
            if (normalizer.isFitLabel()) {
                for (int i = 0; i < normalizer.numOutputs(); i++) {
                    Nd4j.write(normalizer.getLabelMean(i), dos);
                    Nd4j.write(normalizer.getLabelStd(i), dos);
                }
            }
            dos.flush();
        }
    }

    /**
     * Restore a MultiNormalizerStandardize that was previously serialized by this strategy
     *
     * @param stream the input stream to restore from
     * @return the restored MultiNormalizerStandardize
     * @throws IOException
     */
    public MultiNormalizerStandardize restore(@NonNull InputStream stream) throws IOException {
        DataInputStream dis = new DataInputStream(stream);
        boolean fitLabels = dis.readBoolean();
        int numInputs = dis.readInt();
        int numOutputs = dis.readInt();

        MultiNormalizerStandardize result = new MultiNormalizerStandardize();
        result.fitLabel(fitLabels);

        List<DistributionStats> featureStats = new ArrayList<>();
        for (int i = 0; i < numInputs; i++) {
            featureStats.add(new DistributionStats(Nd4j.read(dis), Nd4j.read(dis)));
        }
        result.setFeatureStats(featureStats);

        if (fitLabels) {
            List<DistributionStats> labelStats = new ArrayList<>();
            for (int i = 0; i < numOutputs; i++) {
                labelStats.add(new DistributionStats(Nd4j.read(dis), Nd4j.read(dis)));
            }
            result.setLabelStats(labelStats);
        }

        return result;
    }

    @Override
    public NormalizerType getSupportedType() {
        return NormalizerType.MULTI_STANDARDIZE;
    }
}
