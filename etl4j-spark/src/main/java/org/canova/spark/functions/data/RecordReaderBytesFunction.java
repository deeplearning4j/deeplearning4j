package org.canova.spark.functions.data;

import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.function.Function;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.writable.Writable;
import scala.Tuple2;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.net.URI;
import java.util.Collection;

/**RecordReaderBytesFunction: Converts binary data (in the form of a BytesWritable) to Canova format data
 * ({@code Collection<Writable>}) using a RecordReader
 * @author Alex Black
 */
public class RecordReaderBytesFunction implements Function<Tuple2<Text, BytesWritable>, Collection<Writable>> {

    private final RecordReader recordReader;

    public RecordReaderBytesFunction(RecordReader recordReader){
        this.recordReader = recordReader;
    }

    @Override
    public Collection<Writable> call(Tuple2<Text, BytesWritable> v1) throws Exception {
        URI uri = new URI(v1._1().toString());
        DataInputStream dis = new DataInputStream(new ByteArrayInputStream(v1._2().getBytes()));
        return recordReader.record(uri, dis);
    }


}
