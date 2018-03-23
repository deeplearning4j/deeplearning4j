package org.datavec.api.split.parittion;

import com.google.common.io.Files;
import org.datavec.api.conf.Configuration;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.split.partition.PartitionMetaData;
import org.datavec.api.split.partition.Partitioner;
import org.junit.Test;

import java.io.File;
import java.io.OutputStream;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class PartitionerTests {
    @Test
    public void testRecordsPerFilePartition() {
        Partitioner partitioner = new NumberOfRecordsPartitioner();
        File tmpDir = Files.createTempDir();
        FileSplit fileSplit = new FileSplit(tmpDir);
        assertTrue(fileSplit.needsBootstrapForWrite());
        fileSplit.bootStrapForWrite();
        partitioner.init(fileSplit);
        assertEquals(1,partitioner.numPartitions());
    }

    @Test
    public void testInputAddFile() throws Exception {
        Partitioner partitioner = new NumberOfRecordsPartitioner();
        File tmpDir = Files.createTempDir();
        FileSplit fileSplit = new FileSplit(tmpDir);
        assertTrue(fileSplit.needsBootstrapForWrite());
        fileSplit.bootStrapForWrite();
        Configuration configuration = new Configuration();
        configuration.set(NumberOfRecordsPartitioner.RECORDS_PER_FILE_CONFIG,String.valueOf(5));
        partitioner.init(configuration,fileSplit);
        partitioner.updatePartitionInfo(PartitionMetaData.builder().numRecordsUpdated(5).build());
        assertTrue(partitioner.needsNewPartition());
        OutputStream os = partitioner.openNewStream();
        os.close();
        assertNotNull(os);
        //run more than once to ensure output stream creation works properly
        partitioner.updatePartitionInfo(PartitionMetaData.builder().numRecordsUpdated(5).build());
        os = partitioner.openNewStream();
        os.close();
        assertNotNull(os);


    }

}
