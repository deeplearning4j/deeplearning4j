package org.nd4j.parameterserver.distributed.messages;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.apache.commons.lang3.SerializationUtils;
import org.nd4j.linalg.api.ops.aggregates.impl.HierarchicSoftmax;

/**
 * @author raver119@gmail.com
 */
@Data
public class HierarchicSoftmaxBatchMessage extends BaseVoidMessage {

    public HierarchicSoftmaxBatchMessage() {
        super(1);
    }

}
