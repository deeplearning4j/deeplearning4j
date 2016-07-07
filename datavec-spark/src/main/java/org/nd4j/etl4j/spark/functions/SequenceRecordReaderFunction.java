package org.nd4j.etl4j.spark.functions;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.input.PortableDataStream;
import org.nd4j.etl4j.api.records.reader.SequenceRecordReader;
import org.nd4j.etl4j.api.writable.Writable;
import scala.Tuple2;

import java.io.DataInputStream;
import java.net.URI;
import java.util.Collection;

/**RecordReaderFunction: Given a SequenceRecordReader and a file (via Spark PortableDataStream), load and parse the
 * sequence data into a Collection<Collection<Writable>>
 * @author Alex Black
 */
public class SequenceRecordReaderFunction implements Function<Tuple2<String,PortableDataStream>,Collection<Collection<Writable>>> {
    protected SequenceRecordReader sequenceRecordReader;

    public SequenceRecordReaderFunction(SequenceRecordReader sequenceRecordReader){
        this.sequenceRecordReader = sequenceRecordReader;
    }

    @Override
    public Collection<Collection<Writable>> call(Tuple2<String, PortableDataStream> value) throws Exception {
        URI uri = new URI(value._1());
        PortableDataStream ds = value._2();
        try(DataInputStream dis = ds.open()){
            return sequenceRecordReader.sequenceRecord(uri,dis);
        }
    }
}
