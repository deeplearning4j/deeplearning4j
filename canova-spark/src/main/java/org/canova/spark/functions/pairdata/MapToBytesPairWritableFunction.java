package org.canova.spark.functions.pairdata;

import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.input.PortableDataStream;
import scala.Tuple2;
import scala.Tuple3;

/** A function to read files (assuming exactly 2 per input) from a PortableDataStream and combine the contents into a BytesPairWritable
 * @see org.canova.spark.util.CanovaSparkUtil#combineFilesForSequenceFile(JavaSparkContext, String, String, PathToKeyConverter, PathToKeyConverter)
 */
public class MapToBytesPairWritableFunction implements PairFunction<Tuple2<String, Iterable<Tuple3<String, Integer, PortableDataStream>>>, Text, BytesPairWritable> {
    @Override
    public Tuple2<Text, BytesPairWritable> call(Tuple2<String, Iterable<Tuple3<String, Integer, PortableDataStream>>> in) throws Exception {
        byte[] first = null;
        byte[] second = null;
        String firstOrigPath = null;
        String secondOrigPath = null;
        Iterable<Tuple3<String, Integer, PortableDataStream>> iterable = in._2();
        for (Tuple3<String, Integer, PortableDataStream> tuple : iterable) {
            if (tuple._2() == 0) {
                first = tuple._3().toArray();
                firstOrigPath = tuple._1();
            } else if (tuple._2() == 1) {
                second = tuple._3().toArray();
                secondOrigPath = tuple._1();
            }
        }
        return new Tuple2<>(new Text(in._1()), new BytesPairWritable(first, second, firstOrigPath, secondOrigPath));
    }
}
