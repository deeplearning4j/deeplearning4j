package org.datavec.api.records.mapper;

import lombok.Builder;
import lombok.Getter;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.partition.Partitioner;
import org.datavec.api.writable.Writable;

import java.util.List;

/**
 * This takes data from a specified {@link RecordReader}
 * and writes the data out with the specified {@link RecordWriter}.
 *
 * The setup is as follows:
 *
 * Specify a {@link RecordReader} as the data source
 * Specify a {@link RecordWriter} as the destination.
 *
 * When setting up the locations, use 2 different {@link InputSplit}
 * calling {@link RecordWriter#initialize(InputSplit, Partitioner)}
 * and {@link RecordReader#initialize(InputSplit)}
 * respectively to configure the locations of where the data will be
 * read from and written to.
 *
 * When writing the data, you need to specify a link {@link Partitioner} to
 * determine how to slice up the data being written (say in to number of lines per record per file
 * per {@link org.datavec.api.split.partition.NumberOfRecordsPartitioner} among other implementations.
 *
 * Finally, you may specify a batch size for batch read and write if the record reader and writer support it.
 *
 * See {@link #copy()} for more information here.
 */
@Builder
public class RecordMapper {

    private RecordReader recordReader;
    private RecordWriter recordWriter;
    private InputSplit inputUrl;
    private InputSplit outputUrl;
    @Builder.Default
    private Configuration configuration = new Configuration();
    @Getter
    private Partitioner partitioner;
    private int batchSize;

    /**
     * Copy the {@link RecordReader}
     * data using the {@link RecordWriter}.
     * Note that unless batch is supported by
     * both the {@link RecordReader} and {@link RecordWriter}
     * then writes will happen one at a time.
     * You can see if batch is enabled via {@link RecordReader#batchesSupported()}
     * and {@link RecordWriter#supportsBatch()} respectively.
     * @throws Exception
     */
    public void copy() throws Exception {
        recordReader.initialize(configuration,inputUrl);
        partitioner.init(configuration,outputUrl);
        recordWriter.initialize(configuration,outputUrl,partitioner);


        if(batchSize > 0 && recordReader.batchesSupported() && recordWriter.supportsBatch()) {
            while(recordReader.hasNext()) {
                List<List<Writable>> next = recordReader.next(batchSize);
                //ensure we can write a file for either the current or next iterations
                if(partitioner.needsNewPartition()) {
                    partitioner.currentOutputStream().flush();
                    partitioner.currentOutputStream().close();
                    partitioner.openNewStream();
                }
                //update records written
                partitioner.updatePartitionInfo(recordWriter.writeBatch(next));

            }

            partitioner.currentOutputStream().flush();
            partitioner.currentOutputStream().close();
            recordReader.close();
            recordWriter.close();
        }

        else {
            while(recordReader.hasNext()) {
                List<Writable> next = recordReader.next();
                //update records written
                partitioner.updatePartitionInfo(recordWriter.write(next));
                if(partitioner.needsNewPartition()) {
                    partitioner.openNewStream();
                }
            }
        }

    }


}
