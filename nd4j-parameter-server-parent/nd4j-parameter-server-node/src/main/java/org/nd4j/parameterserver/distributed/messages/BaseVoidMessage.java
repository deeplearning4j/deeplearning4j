package org.nd4j.parameterserver.distributed.messages;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

/**
 * @author raver119@gmail.com
 */
@NoArgsConstructor
@Data
public class BaseVoidMessage implements Serializable {
    protected long nodeId;
    protected long batchId;
}
