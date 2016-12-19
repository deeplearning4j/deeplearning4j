package org.nd4j.linalg.dataset;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.dataset.api.preprocessor.MultiNormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.MultiNormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.MultiNormalizerMinMaxScalerSerializer;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.MultiNormalizerStandardizeSerializer;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerMinMaxScalerSerializer;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerStandardizeSerializer;
import org.nd4j.linalg.dataset.api.preprocessor.stats.DistributionStats;
import org.nd4j.linalg.dataset.api.preprocessor.stats.MinMaxStats;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.File;
import java.io.IOException;

import static java.util.Arrays.asList;
import static org.junit.Assert.assertEquals;

/**
 * @author Ede Meijer
 */
@RunWith(Parameterized.class)
public class NormalizerSerializerTest extends BaseNd4jTest {
    private File tmpFile;

    public NormalizerSerializerTest(Nd4jBackend backend) {
        super(backend);
    }

    @Before
    public void setUp() throws IOException {
        tmpFile = File.createTempFile("test", "preProcessor");
        tmpFile.deleteOnExit();
    }

    @Test
    public void testNormalizerStandardizeNotFitLabels() throws IOException {
        NormalizerStandardize original = new NormalizerStandardize(
            Nd4j.create(new double[]{0.5, 1.5}),
            Nd4j.create(new double[]{2.5, 3.5})
        );

        NormalizerStandardizeSerializer.write(original, tmpFile);
        NormalizerStandardize restored = NormalizerStandardizeSerializer.restore(tmpFile);

        assertEquals(original, restored);
    }

    @Test
    public void testNormalizerStandardizeFitLabels() throws IOException {
        NormalizerStandardize original = new NormalizerStandardize(
            Nd4j.create(new double[]{0.5, 1.5}),
            Nd4j.create(new double[]{2.5, 3.5}),
            Nd4j.create(new double[]{4.5, 5.5}),
            Nd4j.create(new double[]{6.5, 7.5})
        );
        original.fitLabel(true);

        NormalizerStandardizeSerializer.write(original, tmpFile);
        NormalizerStandardize restored = NormalizerStandardizeSerializer.restore(tmpFile);

        assertEquals(original, restored);
    }

    @Test
    public void testNormalizerMinMaxScalerNotFitLabels() throws IOException {
        NormalizerMinMaxScaler original = new NormalizerMinMaxScaler(0.1, 0.9);
        original.setFeatureStats(
            Nd4j.create(new double[]{0.5, 1.5}),
            Nd4j.create(new double[]{2.5, 3.5})
        );

        NormalizerMinMaxScalerSerializer.write(original, tmpFile);
        NormalizerMinMaxScaler restored = NormalizerMinMaxScalerSerializer.restore(tmpFile);

        assertEquals(original, restored);
    }

    @Test
    public void testNormalizerMinMaxScalerFitLabels() throws IOException {
        NormalizerMinMaxScaler original = new NormalizerMinMaxScaler(0.1, 0.9);
        original.setFeatureStats(
            Nd4j.create(new double[]{0.5, 1.5}),
            Nd4j.create(new double[]{2.5, 3.5})
        );
        original.setLabelStats(
            Nd4j.create(new double[]{4.5, 5.5}),
            Nd4j.create(new double[]{6.5, 7.5})
        );
        original.fitLabel(true);

        NormalizerMinMaxScalerSerializer.write(original, tmpFile);
        NormalizerMinMaxScaler restored = NormalizerMinMaxScalerSerializer.restore(tmpFile);

        assertEquals(original, restored);
    }

    @Test
    public void testMultiNormalizerStandardizeNotFitLabels() throws IOException {
        MultiNormalizerStandardize original = new MultiNormalizerStandardize();
        original.setFeatureStats(asList(
            new DistributionStats(
                Nd4j.create(new double[]{0.5, 1.5}),
                Nd4j.create(new double[]{2.5, 3.5})
            ),
            new DistributionStats(
                Nd4j.create(new double[]{4.5, 5.5, 6.5}),
                Nd4j.create(new double[]{7.5, 8.5, 9.5})
            )
        ));

        MultiNormalizerStandardizeSerializer.write(original, tmpFile);
        MultiNormalizerStandardize restored = MultiNormalizerStandardizeSerializer.restore(tmpFile);

        assertEquals(original, restored);
    }

    @Test
    public void testMultiNormalizerStandardizeFitLabels() throws IOException {
        MultiNormalizerStandardize original = new MultiNormalizerStandardize();
        original.setFeatureStats(asList(
            new DistributionStats(
                Nd4j.create(new double[]{0.5, 1.5}),
                Nd4j.create(new double[]{2.5, 3.5})
            ),
            new DistributionStats(
                Nd4j.create(new double[]{4.5, 5.5, 6.5}),
                Nd4j.create(new double[]{7.5, 8.5, 9.5})
            )
        ));
        original.setLabelStats(asList(
            new DistributionStats(
                Nd4j.create(new double[]{0.5, 1.5}),
                Nd4j.create(new double[]{2.5, 3.5})
            ),
            new DistributionStats(
                Nd4j.create(new double[]{4.5}),
                Nd4j.create(new double[]{7.5})
            ),
            new DistributionStats(
                Nd4j.create(new double[]{4.5, 5.5, 6.5}),
                Nd4j.create(new double[]{7.5, 8.5, 9.5})
            )
        ));
        original.fitLabel(true);

        MultiNormalizerStandardizeSerializer.write(original, tmpFile);
        MultiNormalizerStandardize restored = MultiNormalizerStandardizeSerializer.restore(tmpFile);

        assertEquals(original, restored);
    }

    @Test
    public void testMultiNormalizerMinMaxScalerNotFitLabels() throws IOException {
        MultiNormalizerMinMaxScaler original = new MultiNormalizerMinMaxScaler(0.1, 0.9);
        original.setFeatureStats(asList(
            new MinMaxStats(
                Nd4j.create(new double[]{0.5, 1.5}),
                Nd4j.create(new double[]{2.5, 3.5})
            ),
            new MinMaxStats(
                Nd4j.create(new double[]{4.5, 5.5, 6.5}),
                Nd4j.create(new double[]{7.5, 8.5, 9.5})
            )
        ));

        MultiNormalizerMinMaxScalerSerializer.write(original, tmpFile);
        MultiNormalizerMinMaxScaler restored = MultiNormalizerMinMaxScalerSerializer.restore(tmpFile);

        assertEquals(original, restored);
    }

    @Test
    public void testMultiNormalizerMinMaxScalerFitLabels() throws IOException {
        MultiNormalizerMinMaxScaler original = new MultiNormalizerMinMaxScaler(0.1, 0.9);
        original.setFeatureStats(asList(
            new MinMaxStats(
                Nd4j.create(new double[]{0.5, 1.5}),
                Nd4j.create(new double[]{2.5, 3.5})
            ),
            new MinMaxStats(
                Nd4j.create(new double[]{4.5, 5.5, 6.5}),
                Nd4j.create(new double[]{7.5, 8.5, 9.5})
            )
        ));
        original.setLabelStats(asList(
            new MinMaxStats(
                Nd4j.create(new double[]{0.5, 1.5}),
                Nd4j.create(new double[]{2.5, 3.5})
            ),
            new MinMaxStats(
                Nd4j.create(new double[]{4.5}),
                Nd4j.create(new double[]{7.5})
            ),
            new MinMaxStats(
                Nd4j.create(new double[]{4.5, 5.5, 6.5}),
                Nd4j.create(new double[]{7.5, 8.5, 9.5})
            )
        ));
        original.fitLabel(true);

        MultiNormalizerMinMaxScalerSerializer.write(original, tmpFile);
        MultiNormalizerMinMaxScaler restored = MultiNormalizerMinMaxScalerSerializer.restore(tmpFile);

        assertEquals(original, restored);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
