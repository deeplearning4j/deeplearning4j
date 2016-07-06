package io.skymind.echidna.spark.misc;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function;
import org.canova.api.writable.Writable;

import java.util.List;

/**
 * Simple function to map an example to a String format (such as CSV)
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class WritablesToStringFunction implements Function<List<Writable>,String> {

    private final String delim;

    @Override
    public String call(List<Writable> c) throws Exception {

        StringBuilder sb = new StringBuilder();
        boolean first = true;
        for(Writable w : c){
            if(!first) sb.append(delim);
            sb.append(w.toString());
            first = false;
        }

        return sb.toString();
    }

}
