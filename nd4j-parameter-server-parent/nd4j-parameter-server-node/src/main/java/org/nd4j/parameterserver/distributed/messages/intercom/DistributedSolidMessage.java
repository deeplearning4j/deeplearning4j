package org.nd4j.parameterserver.distributed.messages.intercom;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.DistributedMessage;

/**
 * Array passed here will be shared & available on all shards.
 *
 * @author raver119@gmail.com
 */
@Data
@NoArgsConstructor
public class DistributedSolidMessage extends BaseVoidMessage implements DistributedMessage {
    /**
     * The only use of this message is negTable sharing.
     */

    private Integer key;
    private INDArray payload;
    private boolean overwrite;

    public DistributedSolidMessage(@NonNull Integer key, @NonNull INDArray array, boolean overwrite) {
        super(5);
        this.payload = array;
        this.key = key;
        this.overwrite = overwrite;
    }

    /**
     * This method will be started in context of executor, either Shard, Client or Backup node
     */
    @Override
    public void processMessage() {
        if (overwrite)
            storage.setArray(key, payload);
        else if (!storage.arrayExists(key))
            storage.setArray(key, payload);
    }
}
