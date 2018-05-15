package org.deeplearning4j.perf.listener;

import lombok.Builder;
import lombok.Data;

import java.io.Serializable;

@Data
@Builder
public class DiskInfo implements Serializable {
    private long bytesRead,bytesWritten,transferTime;
    private String name,modelName;
}
