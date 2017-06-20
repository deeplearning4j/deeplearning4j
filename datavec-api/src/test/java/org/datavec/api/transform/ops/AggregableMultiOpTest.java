package org.datavec.api.transform.ops;

import org.datavec.api.writable.Writable;
import org.junit.Test;
import org.reflections.Reflections;
import org.reflections.scanners.SubTypesScanner;
import org.reflections.util.ClasspathHelper;
import org.reflections.util.ConfigurationBuilder;
import org.reflections.util.FilterBuilder;

import java.io.Serializable;
import java.util.*;

import static org.junit.Assert.*;

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
        for(int i = 0; i < intList.size(); i++){
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

        for(int i = 0; i < intList.size(); i++){
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
        List<ClassLoader> classLoadersList = new LinkedList<ClassLoader>();
        classLoadersList.add(ClasspathHelper.contextClassLoader());
        classLoadersList.add(ClasspathHelper.staticClassLoader());

        Reflections reflections = new Reflections(new ConfigurationBuilder()
                .setScanners(new SubTypesScanner(false /* don't exclude Object.class */))
                .setUrls(ClasspathHelper.forClassLoader(classLoadersList.toArray(new ClassLoader[0])))
                .filterInputsBy(new FilterBuilder().include(FilterBuilder.prefix("org.datavec.api.transform.ops"))));

        Set<String> allTypes = reflections.getAllTypes();
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
        if (cls == null) { return false; }
        if (cls == Serializable.class) { return true; }

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