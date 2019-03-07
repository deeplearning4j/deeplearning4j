package org.nd4j.linalg.profiler;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ProfilerConfig {

    private boolean notOptimalArguments;
    private boolean notOptimalTAD;
    private boolean nativeStatistics;
    private boolean checkForNAN;
    private boolean checkForINF;
    private boolean stackTrace;
    private boolean checkElapsedTime;
    private boolean checkWorkspaces;
}
