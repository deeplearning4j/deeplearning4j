package org.datavec.api.split.partition;

import org.datavec.api.conf.Configuration;
import org.datavec.api.split.InputSplit;

import java.io.OutputStream;
import java.net.URI;

/**
 * Partition relative to number of records written per file.
 * This partitioner will ensure that no more than
 * {@link #recordsPerFile} number of records is written per
 * file when outputting to the various locations of the
 * {@link InputSplit} locations.
 *
 * @author Adam Gibson
 */
public class NumberOfRecordsPartitioner implements Partitioner {

    private URI[] locations;
    private int recordsPerFile = DEFAULT_RECORDS_PER_FILE;
    //all records in to 1 file
    public final static int DEFAULT_RECORDS_PER_FILE = -1;

    public final static String RECORDS_PER_FILE_CONFIG = "org.datavec.api.split.partition.numrecordsperfile";
    private int numRecordsSoFar = 0;
    private int currLocation;
    private InputSplit inputSplit;
    private OutputStream current;
    private boolean doneWithCurrentLocation = false;
    private int totalRecordsWritten;

    @Override
    public int totalRecordsWritten() {
        return totalRecordsWritten;
    }

    @Override
    public int numRecordsWritten() {
        return numRecordsSoFar;
    }

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
        this.inputSplit = inputSplit;

    }

    @Override
    public void init(Configuration configuration, InputSplit split) {
        init(split);
        this.recordsPerFile = configuration.getInt(RECORDS_PER_FILE_CONFIG,DEFAULT_RECORDS_PER_FILE);
    }

    @Override
    public void updatePartitionInfo(PartitionMetaData metadata) {
        this.numRecordsSoFar += metadata.getNumRecordsUpdated();
        this.totalRecordsWritten += metadata.getNumRecordsUpdated();
        if(numRecordsSoFar >= recordsPerFile && recordsPerFile > 0)  {
            doneWithCurrentLocation = true;
        }
    }

    @Override
    public boolean needsNewPartition() {
        doneWithCurrentLocation = numRecordsSoFar >= recordsPerFile && recordsPerFile > 0;
        return recordsPerFile > 0 && numRecordsSoFar >= recordsPerFile ||  doneWithCurrentLocation;
    }

    @Override
    public OutputStream openNewStream() {
        //reset status of location
        doneWithCurrentLocation = false;
        //ensure count is 0 for records so far for current record
        numRecordsSoFar = 0;

        //only append when directory, also ensure we can bootstrap and we can write to the current location
        if(currLocation >= locations.length - 1 && locations.length >= 1 && needsNewPartition() || inputSplit.needsBootstrapForWrite() ||
                locations.length < 1 ||
                currLocation >= locations.length || !inputSplit.canWriteToLocation(locations[currLocation])
                && needsNewPartition()) {

            String newInput = inputSplit.addNewLocation();
            try {
                OutputStream ret =  inputSplit.openOutputStreamFor(newInput);
                this.current = ret;
                return ret;
            } catch (Exception e) {
                throw new IllegalStateException(e);
            }
        }

        else {
            try {
                OutputStream ret =  inputSplit.openOutputStreamFor(locations[currLocation].toString());
                currLocation++;
                this.current = ret;
                return ret;
            } catch (Exception e) {
                throw new IllegalStateException(e);
            }
        }

    }

    @Override
    public OutputStream currentOutputStream() {
        if(current == null) {
            current = openNewStream();
        }
        return current;
    }
}
