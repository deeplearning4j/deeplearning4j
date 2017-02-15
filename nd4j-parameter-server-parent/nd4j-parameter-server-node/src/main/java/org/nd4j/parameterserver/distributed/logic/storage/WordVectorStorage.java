package org.nd4j.parameterserver.distributed.logic.storage;

import org.nd4j.parameterserver.distributed.logic.storage.BaseStorage;

/**
 * @author raver119@gmail.com
 */
public class WordVectorStorage extends BaseStorage {
    public static final Integer SYN_0 = "syn0".hashCode();
    public static final Integer SYN_1 = "syn1".hashCode();
    public static final Integer SYN_1_NEGATIVE = "syn1Neg".hashCode();
    public static final Integer EXP_TABLE = "expTable".hashCode();
    public static final Integer NEGATIVE_TABLE = "negTable".hashCode();
}
