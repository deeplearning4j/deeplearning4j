package org.nd4j.parameterserver.distributed.messages;

/**
 * Array passed here will be shared & available on all shards.
 *
 * @author raver119@gmail.com
 */
public class ShareSolidMessage extends BaseVoidMessage {
    /**
     * FIXME: We don't really need this thing for a poc, but we'll need it for generalized DistributedOpExecutioner implementation
     */
}
