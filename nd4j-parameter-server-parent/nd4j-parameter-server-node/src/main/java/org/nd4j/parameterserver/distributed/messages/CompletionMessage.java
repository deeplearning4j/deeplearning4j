package org.nd4j.parameterserver.distributed.messages;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * This message contains information about finished computations for specific batch, being sent earlier
 *
 * @author raver119@gmail.com
 */
@Data
@NoArgsConstructor
public class CompletionMessage extends BaseVoidMessage {

}
