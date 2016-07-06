package org.canova.spark.functions.pairdata;

import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.function.Function;
import org.canova.api.records.reader.SequenceRecordReader;
import org.canova.api.writable.Writable;
import scala.Tuple2;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.net.URI;
import java.util.Collection;

/**SequenceRecordReaderBytesFunction: Converts two sets of binary data (in the form of a BytesPairWritable) to Canova format data
 * ({@code Tuple2<Collection<Collection<<Writable>>,Collection<Collection<Writable>>}) using two SequenceRecordReaders.
 * Used for example when network input and output data comes from different files
 * @author Alex Black
 */
public class PairSequenceRecordReaderBytesFunction implements Function<Tuple2<Text, BytesPairWritable>, Tuple2<Collection<Collection<Writable>>,Collection<Collection<Writable>>>> {
    private final SequenceRecordReader recordReaderFirst;
    private final SequenceRecordReader recordReaderSecond;

    public PairSequenceRecordReaderBytesFunction(SequenceRecordReader recordReaderFirst, SequenceRecordReader recordReaderSecond){
        this.recordReaderFirst = recordReaderFirst;
        this.recordReaderSecond = recordReaderSecond;
    }

    @Override
    public Tuple2<Collection<Collection<Writable>>,Collection<Collection<Writable>>> call(Tuple2<Text, BytesPairWritable> v1) throws Exception {
        BytesPairWritable bpw = v1._2();
        DataInputStream dis1 = new DataInputStream(new ByteArrayInputStream(bpw.getFirst()));
        DataInputStream dis2 = new DataInputStream(new ByteArrayInputStream(bpw.getSecond()));
        URI u1 = (bpw.getUriFirst() != null ? new URI(bpw.getUriFirst()) : null);
        URI u2 = (bpw.getUriSecond() != null ? new URI(bpw.getUriSecond()) : null);
        Collection<Collection<Writable>> first = recordReaderFirst.sequenceRecord(u1, dis1);
        Collection<Collection<Writable>> second = recordReaderSecond.sequenceRecord(u2, dis2);
        return new Tuple2<>(first,second);
    }
}
