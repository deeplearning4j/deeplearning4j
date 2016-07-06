package org.canova.spark.functions.data;

import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.input.PortableDataStream;
import scala.Tuple2;

/**A PairFunction that simply loads bytes[] from a a PortableDataStream, and wraps it (and the String key)
 * in Text and BytesWritable respectively.
 * @author Alex Black
 */
public class FilesAsBytesFunction implements PairFunction<Tuple2<String,PortableDataStream>, Text, BytesWritable> {
    @Override
    public Tuple2<Text, BytesWritable> call(Tuple2<String, PortableDataStream> in) throws Exception {
        return new Tuple2<>(new Text(in._1()), new BytesWritable(in._2().toArray()));
    }
}
