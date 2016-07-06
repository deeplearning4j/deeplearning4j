package org.canova.spark.functions;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.input.PortableDataStream;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.writable.Writable;
import scala.Tuple2;

import java.io.DataInputStream;
import java.net.URI;
import java.util.Collection;

/**RecordReaderFunction: Given a RecordReader and a file (via Spark PortableDataStream), load and parse the
 * data into a Collection<Writable>.
 * NOTE: This is only useful for "one record per file" type situations (ImageRecordReader, etc)
 * @author Alex Black
 */
public class RecordReaderFunction implements Function<Tuple2<String,PortableDataStream>,Collection<Writable>> {
    protected RecordReader recordReader;

    public RecordReaderFunction(RecordReader recordReader){
        this.recordReader = recordReader;
    }

    @Override
    public Collection<Writable> call(Tuple2<String, PortableDataStream> value) throws Exception {
        URI uri = new URI(value._1());
        PortableDataStream ds = value._2();
        try( DataInputStream dis = ds.open() ){
            return recordReader.record(uri,dis);
        }
    }
}
