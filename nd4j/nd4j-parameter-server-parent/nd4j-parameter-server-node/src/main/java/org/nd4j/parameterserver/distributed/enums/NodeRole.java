package org.nd4j.parameterserver.distributed.enums;

/**
 * @author raver119@gmail.com
 */
public enum NodeRole {
    NONE, // just undefined role
    SHARD, // basic processing node
    BACKUP, // acts are backup node, replicating one of the shards
    CLIENT, // just a client node,
    MASTER, // acts as shard + some additional functionality
}
