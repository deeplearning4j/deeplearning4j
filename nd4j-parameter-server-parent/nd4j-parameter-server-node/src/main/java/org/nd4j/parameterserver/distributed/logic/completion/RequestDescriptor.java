package org.nd4j.parameterserver.distributed.logic.completion;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/**
 * @author raver119@gmail.com
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class RequestDescriptor {
    private long originatorId;
    private long taskId;


    public static RequestDescriptor createDescriptor(long originatorId, long taskId) {
        return new RequestDescriptor(originatorId, taskId);
    }
}
