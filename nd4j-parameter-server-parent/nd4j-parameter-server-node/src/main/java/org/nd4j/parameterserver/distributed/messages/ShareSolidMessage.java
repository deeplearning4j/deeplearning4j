package org.nd4j.parameterserver.distributed.messages;

import lombok.Data;
import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Array passed here will be shared & available on all shards.
 *
 * @author raver119@gmail.com
 */
@Data
public class ShareSolidMessage extends BaseVoidMessage {
    /**
     * The only use of this message is negTable sharing.
     */

    private INDArray payload;

    public ShareSolidMessage(@NonNull INDArray array) {
        super(5);
        this.payload = array;
    }
}
