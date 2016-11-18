package org.nd4j.linalg.dataset;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.dataset.api.preprocessor.MultiNormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerSerializer;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.File;
import java.io.IOException;

import static java.util.Arrays.asList;
import static org.junit.Assert.assertEquals;

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

        NormalizerSerializer.write(original, tmpFile);
        NormalizerStandardize restored = NormalizerSerializer.restoreNormalizerStandardize(tmpFile);

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

        NormalizerSerializer.write(original, tmpFile);
        NormalizerStandardize restored = NormalizerSerializer.restoreNormalizerStandardize(tmpFile);

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

        NormalizerSerializer.write(original, tmpFile);
        MultiNormalizerStandardize restored = NormalizerSerializer.restoreMultiNormalizerStandardize(tmpFile);

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

        NormalizerSerializer.write(original, tmpFile);
        MultiNormalizerStandardize restored = NormalizerSerializer.restoreMultiNormalizerStandardize(tmpFile);

        assertEquals(original, restored);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
