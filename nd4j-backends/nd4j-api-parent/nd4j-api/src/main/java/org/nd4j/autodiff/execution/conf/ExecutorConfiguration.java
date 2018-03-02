package org.nd4j.autodiff.execution.conf;

import com.google.flatbuffers.FlatBufferBuilder;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.graph.Direction;
import org.nd4j.graph.FlatConfiguration;
import org.nd4j.graph.ProfilingMode;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;

/**
 *
 * @author raver119@gmail.com
 */
@Data
@Slf4j
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ExecutorConfiguration {
    @Builder.Default private OpExecutioner.ProfilingMode profilingMode = OpExecutioner.ProfilingMode.DISABLED;
    @Builder.Default private ExecutionMode executionMode = ExecutionMode.SEQUENTIAL;
    @Builder.Default private OutputMode outputMode = OutputMode.IMPLICIT;
    @Builder.Default boolean gatherTimings = true;
    @Builder.Default private long footprintForward = 0L;
    @Builder.Default private long footprintBackward = 0L;


    /**
     * This method
     * @param builder
     * @return
     */
    public int getFlatConfiguration(FlatBufferBuilder builder) {

        byte prof = profilingMode == OpExecutioner.ProfilingMode.INF_PANIC ? ProfilingMode.INF_PANIC :
                    profilingMode == OpExecutioner.ProfilingMode.NAN_PANIC ? ProfilingMode.NAN_PANIC :
                    profilingMode == OpExecutioner.ProfilingMode.ANY_PANIC ? ProfilingMode.ANY_PANIC : ProfilingMode.NONE;

        byte exec = executionMode == ExecutionMode.SEQUENTIAL ? org.nd4j.graph.ExecutionMode.SEQUENTIAL :
                    executionMode == ExecutionMode.AUTO ? org.nd4j.graph.ExecutionMode.AUTO :
                    executionMode == ExecutionMode.STRICT ? org.nd4j.graph.ExecutionMode.STRICT : -1;

        byte outp = outputMode == OutputMode.IMPLICIT ? org.nd4j.graph.OutputMode.IMPLICIT :
                    outputMode == OutputMode.EXPLICIT ? org.nd4j.graph.OutputMode.EXPLICIT :
                    outputMode == OutputMode.EXPLICIT_AND_IMPLICIT ? org.nd4j.graph.OutputMode.EXPLICIT_AND_IMPLICIT :
                    outputMode == OutputMode.VARIABLE_SPACE ? org.nd4j.graph.OutputMode.VARIABLE_SPACE : -1;

        if (exec == -1)
            throw new UnsupportedOperationException("Unknown values were passed into configuration as ExecutionMode: [" + executionMode + "]");

        if (outp == -1)
            throw new UnsupportedOperationException("Unknown values were passed into configuration as OutputMode: [" + outputMode + "]");

        return FlatConfiguration.createFlatConfiguration(builder, -1, prof, exec, outp, gatherTimings, footprintForward, footprintBackward, Direction.FORWARD_ONLY);
    }
}
