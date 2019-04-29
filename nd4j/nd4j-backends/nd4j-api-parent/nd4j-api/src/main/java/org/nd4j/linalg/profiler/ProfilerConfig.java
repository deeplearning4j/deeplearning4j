package org.nd4j.linalg.profiler;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ProfilerConfig {

    @Builder.Default private boolean notOptimalArguments = false;
    @Builder.Default private boolean notOptimalTAD = false;
    @Builder.Default private boolean nativeStatistics = false;
    @Builder.Default private boolean checkForNAN = false;
    @Builder.Default private boolean checkForINF = false;
    @Builder.Default private boolean stackTrace = false;
    @Builder.Default private boolean checkElapsedTime = false;

    /**
     * If enabled, each pointer will be workspace validation will be performed one each call
     */
    @Builder.Default private boolean checkWorkspaces = true;

    /**
     * If enabled, thread<->device affinity will be checked on each call
     *
     * PLEASE NOTE: everything will gets slower
     */
    @Builder.Default private boolean checkLocality = false;
}
