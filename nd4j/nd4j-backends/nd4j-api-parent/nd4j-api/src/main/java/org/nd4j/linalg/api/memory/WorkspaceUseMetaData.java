package org.nd4j.linalg.api.memory;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class WorkspaceUseMetaData {

    private String stackTrace;
    private String workspaceName;
    private long eventTime;
    private EventTypes eventType;
    private String threadName;

    public static enum EventTypes {
        ENTER,
        CLOSE,
        BORROW

    }

}
