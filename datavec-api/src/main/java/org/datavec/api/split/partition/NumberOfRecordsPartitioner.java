package org.datavec.api.split.partition;

import org.datavec.api.conf.Configuration;
import org.datavec.api.split.InputSplit;

import java.net.URI;

public class NumberOfRecordsPartitioner implements Partitioner {

    private URI[] locations;
    private int recordsPerFile = DEFAULT_RECORDS_PER_FILE;
    //all records in to 1 file
    public final static int DEFAULT_RECORDS_PER_FILE = -1;

    public final static String RECORDS_PER_FILE_CONFIG = "org.datavec.api.split.partition.numrecordsperfile";

    @Override
    public int numPartitions() {
        //possible it's a directory
        if(locations.length < 2) {

            if(locations.length > 0 && locations[0].isAbsolute()) {
                return recordsPerFile;
            }
            //append all results to 1 file when -1
            else {
                return 1;
            }
        }

        //otherwise it's a series of specified files.
        return locations.length / recordsPerFile;
    }

    @Override
    public void init(InputSplit inputSplit) {
        this.locations = inputSplit.locations();
    }

    @Override
    public void init(Configuration configuration, InputSplit split) {
        init(split);
        this.recordsPerFile = configuration.getInt(RECORDS_PER_FILE_CONFIG,DEFAULT_RECORDS_PER_FILE);
    }
}
