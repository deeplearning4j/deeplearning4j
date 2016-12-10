package org.nd4j.parameterserver.distributed.messages;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.apache.commons.lang3.SerializationUtils;

/**
 * This message contains information about finished computations for specific batch, being sent earlier
 *
 * @author raver119@gmail.com
 */
@Data
public class CompletionMessage extends BaseVoidMessage {
    public CompletionMessage() {
        super(10);
    }
}
