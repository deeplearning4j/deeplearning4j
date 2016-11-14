package org.deeplearning4j.ui.storage;

import org.agrona.DirectBuffer;
import org.agrona.MutableDirectBuffer;
import org.deeplearning4j.api.storage.Persistable;

/**
 * Created by Alex on 07/10/2016.
 */
public interface AgronaPersistable extends Persistable {

    void encode(MutableDirectBuffer buffer);

    void decode(DirectBuffer buffer);

}
