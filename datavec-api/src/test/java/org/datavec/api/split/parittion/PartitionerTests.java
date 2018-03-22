package org.datavec.api.split.parittion;

import com.google.common.io.Files;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.split.partition.Partitioner;
import org.junit.Test;

import java.io.File;

import static org.junit.Assert.assertEquals;

public class PartitionerTests {
    @Test
    public void testRecordsPerFilePartition() {
        Partitioner partitioner = new NumberOfRecordsPartitioner();
        File tmpDir = Files.createTempDir();
        FileSplit fileSplit = new FileSplit(tmpDir);
        partitioner.init(fileSplit);
        assertEquals(NumberOfRecordsPartitioner.DEFAULT_RECORDS_PER_FILE,partitioner.numPartitions());
    }

}
