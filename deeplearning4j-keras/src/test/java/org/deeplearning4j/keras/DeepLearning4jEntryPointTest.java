package org.deeplearning4j.keras;

import org.apache.commons.io.FileUtils;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import java.io.File;
import java.io.IOException;

public class DeepLearning4jEntryPointTest {

    private DeepLearning4jEntryPoint deepLearning4jEntryPoint = new DeepLearning4jEntryPoint();

    public @Rule ExpectedException thrown = ExpectedException.none();

    @Test
    public void shouldFitTheSampleSequentialModel() throws Exception {
        // Given
        File model = prepareResource("theano_mnist/model.h5");
        File features = prepareResource("theano_mnist/x_train.h5");
        File labels = prepareResource("theano_mnist/y_train.h5");

        EntryPointFitParameters entryPointParameters = EntryPointFitParameters
                .builder()
                .modelFilePath(model.getAbsolutePath())
                .trainFeaturesFile(features.getAbsolutePath())
                .trainLabelsFile(labels.getAbsolutePath())
                .batchSize(128)
                .nbEpoch(2)
                .type("sequential")
                .build();

        // When
        deepLearning4jEntryPoint.fit(entryPointParameters);

        // Then
        // fall through - the rule will fail the test execution if an exception is thrown
    }

    private File prepareResource(String resourceName) throws IOException {
        File file = File.createTempFile("dl4j", "test");
        file.delete();
        file.deleteOnExit();
        FileUtils.copyInputStreamToFile(this.getClass().getClassLoader().getResourceAsStream(resourceName), file);
        return file;
    }

}