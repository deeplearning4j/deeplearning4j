package org.datavec.api.transform.ops;

import lombok.Getter;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;

/**
 * Groups useful {@link IAggregableReduceOp} utilities on Strings
 *
 * Created by huitseeker on 5/18/17.
 */
public class StringAggregatorImpls {

    private static abstract class AggregableStringReduce implements IAggregableReduceOp<String, Writable> {
        @Getter
        protected StringBuilder sb = new StringBuilder();
    }

    public static class AggregableStringAppend extends AggregableStringReduce {

        @Override
        public <W extends IAggregableReduceOp<String, Writable>> void combine(W accu) {
            if (accu instanceof AggregableStringAppend)
                sb.append(((AggregableStringAppend) accu).getSb());
            else
                throw new UnsupportedOperationException("Tried to combine() incompatible " + accu.getClass().getName()
                                + " operator where" + this.getClass().getName() + " expected");
        }

        @Override
        public void accept(String s) {
            sb.append(s);
        }

        @Override
        public Writable get() {
            return new Text(sb.toString());
        }
    }

    public static class AggregableStringPrepend extends AggregableStringReduce {

        @Override
        public <W extends IAggregableReduceOp<String, Writable>> void combine(W accu) {
            if (accu instanceof AggregableStringPrepend)
                sb.append(((AggregableStringPrepend) accu).getSb());
            else
                throw new UnsupportedOperationException("Tried to combine() incompatible " + accu.getClass().getName()
                                + " operator where" + this.getClass().getName() + " expected");
        }

        @Override
        public void accept(String s) {
            String rev = new StringBuilder(s).reverse().toString();
            sb.append(rev);
        }

        @Override
        public Writable get() {
            return new Text(sb.reverse().toString());
        }
    }

}
