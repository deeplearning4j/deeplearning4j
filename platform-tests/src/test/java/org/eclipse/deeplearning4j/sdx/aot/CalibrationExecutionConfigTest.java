package org.eclipse.deeplearning4j.sdx.aot;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CalibrationExecutionConfigTest {
    @Test
    void calibrationUsesExplicitSlotBySlotExecution() {
        assertEquals(GraphExecutionMode.SLOT_BY_SLOT,
                SdxGgufModelPreparer.calibrationExecutionConfig().getExecutionMode());
    }
}
