package org.datavec.api.transform.ops;

import org.datavec.api.writable.Writable;
import org.junit.Test;

import java.io.Serializable;
import java.util.*;

import static org.junit.Assert.assertTrue;

/**
 * Created by huitseeker on 5/14/17.
 */
public class AggregableMultiOpTest {

    private List<Integer> intList = new ArrayList<>(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9));

    @Test
    public void testMulti() throws Exception {
        AggregatorImpls.AggregableFirst<Integer> af = new AggregatorImpls.AggregableFirst<>();
        AggregatorImpls.AggregableSum<Integer> as = new AggregatorImpls.AggregableSum<>();
        AggregableMultiOp<Integer> multi = new AggregableMultiOp<>(Arrays.asList(af, as));

        assertTrue(multi.getOperations().size() == 2);
        for (int i = 0; i < intList.size(); i++) {
            multi.accept(intList.get(i));
        }

        // mutablility
        assertTrue(as.get().toDouble() == 45D);
        assertTrue(af.get().toInt() == 1);

        List<Writable> res = multi.get();
        assertTrue(res.get(1).toDouble() == 45D);
        assertTrue(res.get(0).toInt() == 1);

        AggregatorImpls.AggregableFirst<Integer> rf = new AggregatorImpls.AggregableFirst<>();
        AggregatorImpls.AggregableSum<Integer> rs = new AggregatorImpls.AggregableSum<>();
        AggregableMultiOp<Integer> reverse = new AggregableMultiOp<>(Arrays.asList(rf, rs));

        for (int i = 0; i < intList.size(); i++) {
            reverse.accept(intList.get(intList.size() - i - 1));
        }

        List<Writable> revRes = reverse.get();
        assertTrue(revRes.get(1).toDouble() == 45D);
        assertTrue(revRes.get(0).toInt() == 9);

        multi.combine(reverse);
        List<Writable> combinedRes = multi.get();
        assertTrue(combinedRes.get(1).toDouble() == 90D);
        assertTrue(combinedRes.get(0).toInt() == 1);
    }

    @Test
    public void testAllAggregateOpsAreSerializable() throws Exception {
        Set<String> allTypes = new HashSet<>();
        allTypes.add("org.datavec.api.transform.ops.LongWritableOp");
        allTypes.add("org.datavec.api.transform.ops.IntWritableOp");
        allTypes.add("org.datavec.api.transform.ops.AggregatorImpls$AggregableMean");
        allTypes.add("org.datavec.api.transform.ops.StringAggregatorImpls$AggregableStringReduce");
        allTypes.add("org.datavec.api.transform.ops.AggregatorImpls$AggregableRange");
        allTypes.add("org.datavec.api.transform.ops.AggregatorImplsTest");
        allTypes.add("org.datavec.api.transform.ops.DispatchWithConditionOp");
        allTypes.add("org.datavec.api.transform.ops.AggregatorImpls$AggregableVariance");
        allTypes.add("org.datavec.api.transform.ops.AggregatorImpls");
        allTypes.add("org.datavec.api.transform.ops.FloatWritableOp");
        allTypes.add("org.datavec.api.transform.ops.AggregatorImpls$AggregableProd");
        allTypes.add("org.datavec.api.transform.ops.AggregatorImpls$AggregableLast");
        allTypes.add("org.datavec.api.transform.ops.StringAggregatorImpls$AggregableStringPrepend");
        allTypes.add("org.datavec.api.transform.ops.ByteWritableOp");
        allTypes.add("org.datavec.api.transform.ops.AggregableMultiOpTest");
        allTypes.add("org.datavec.api.transform.ops.AggregatorImpls$AggregableStdDev");
        allTypes.add("org.datavec.api.transform.ops.StringAggregatorImpls$1");
        allTypes.add("org.datavec.api.transform.ops.DispatchOp");
        allTypes.add("org.datavec.api.transform.ops.AggregatorImpls$AggregableMin");
        allTypes.add("org.datavec.api.transform.ops.StringAggregatorImpls$AggregableStringAppend");
        allTypes.add("org.datavec.api.transform.ops.AggregatorImpls$AggregableCount");
        allTypes.add("org.datavec.api.transform.ops.AggregatorImpls$AggregableSum");
        allTypes.add("org.datavec.api.transform.ops.AggregatorImpls$AggregablePopulationVariance");
        allTypes.add("org.datavec.api.transform.ops.AggregableCheckingOp");
        allTypes.add("org.datavec.api.transform.ops.AggregatorImpls$AggregableMax");
        allTypes.add("org.datavec.api.transform.ops.AggregableMultiOp");
        allTypes.add("org.datavec.api.transform.ops.IAggregableReduceOp");
        allTypes.add("org.datavec.api.transform.ops.DispatchOpTest");
        allTypes.add("org.datavec.api.transform.ops.AggregatorImpls$AggregableCountUnique");
        allTypes.add("org.datavec.api.transform.ops.AggregatorImpls$AggregableUncorrectedStdDev");
        allTypes.add("org.datavec.api.transform.ops.StringWritableOp");
        allTypes.add("org.datavec.api.transform.ops.StringAggregatorImpls");
        allTypes.add("org.datavec.api.transform.ops.DoubleWritableOp");
        allTypes.add("org.datavec.api.transform.ops.AggregatorImpls$AggregableFirst");

        Set<String> ops = new HashSet<>();

        for (String type : allTypes) {
            if (type.startsWith("org.datavec.api.transform.ops")) {
                if (type.endsWith("Op")) {
                    ops.add(type);
                }

                if (type.contains("Aggregable") && !type.endsWith("Test")) {
                    ops.add(type);
                }
            }
        }

        for (String op : ops) {
            Class<?> cls = Class.forName(op);
            assertTrue(op + " should implement Serializable", implementsSerializable(cls));
        }
    }

    private boolean implementsSerializable(Class<?> cls) {
        if (cls == null) {
            return false;
        }
        if (cls == Serializable.class) {
            return true;
        }

        Class<?>[] interfaces = cls.getInterfaces();
        Set<Class<?>> parents = new HashSet<>();
        parents.add(cls.getSuperclass());

        for (Class<?> anInterface : interfaces) {
            Collections.addAll(parents, anInterface.getInterfaces());

            if (anInterface.equals(Serializable.class)) {
                return true;
            }
        }

        for (Class<?> parent : parents) {
            if (implementsSerializable(parent)) {
                return true;
            }
        }

        return false;
    }
}
