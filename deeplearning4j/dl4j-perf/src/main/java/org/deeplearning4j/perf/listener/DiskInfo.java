package org.deeplearning4j.perf.listener;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

import java.io.Serializable;

@Data
@Builder
@AllArgsConstructor
public class DiskInfo implements Serializable {
    private long bytesRead,bytesWritten,transferTime;
    private String name,modelName;

    private DiskInfo(){
        //No-arg for JSON/YAML
    }
}
