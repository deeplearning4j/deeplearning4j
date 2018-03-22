package org.datavec.api.split.partition;

import org.datavec.api.conf.Configuration;
import org.datavec.api.split.InputSplit;

public interface Partitioner {

    int numPartitions();

    void init(InputSplit inputSplit);

    void init(Configuration configuration,InputSplit split);


}
