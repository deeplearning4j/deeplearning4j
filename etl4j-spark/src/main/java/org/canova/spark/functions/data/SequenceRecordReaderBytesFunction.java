package org.canova.spark.functions.data;

import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.function.Function;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.SequenceRecordReader;
import org.canova.api.writable.Writable;
import scala.Tuple2;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.net.URI;
import java.util.Collection;

/**SequenceRecordReaderBytesFunction: Converts binary data (in the form of a BytesWritable) to Canova format data
 * ({@code Collection<Collection<<Writable>>}) using a SequenceRecordReader
 * @author Alex Black
 */
public class SequenceRecordReaderBytesFunction implements Function<Tuple2<Text, BytesWritable>, Collection<Collection<Writable>>> {

    private final SequenceRecordReader recordReader;

    public SequenceRecordReaderBytesFunction(SequenceRecordReader recordReader){
        this.recordReader = recordReader;
    }

    @Override
    public Collection<Collection<Writable>> call(Tuple2<Text, BytesWritable> v1) throws Exception {
        URI uri = new URI(v1._1().toString());
        DataInputStream dis = new DataInputStream(new ByteArrayInputStream(v1._2().getBytes()));
        return recordReader.sequenceRecord(uri, dis);
    }
}
