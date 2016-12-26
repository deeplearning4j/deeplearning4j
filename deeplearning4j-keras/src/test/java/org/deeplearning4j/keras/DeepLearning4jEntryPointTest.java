package org.deeplearning4j.keras;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

public class DeepLearning4jEntryPointTest {

    private DeepLearning4jEntryPoint deepLearning4jEntryPoint = new DeepLearning4jEntryPoint();

    public @Rule ExpectedException thrown = ExpectedException.none();

    @Test
    public void shouldFitTheSampleSequentialModel() throws Exception {
        // Given
        EntryPointFitParameters entryPointParameters = EntryPointFitParameters
                .builder()
                .batchSize(128)
                .nbEpoch(2)
                .build();

        // When
        deepLearning4jEntryPoint.fit(entryPointParameters);

        // Then
        // fall through - the rule will fail the test execution if an exception is thrown
    }

}