package org.deeplearning4j.linalg.factory;

import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.springframework.core.io.ClassPathResource;

import java.lang.reflect.Constructor;
import java.util.Properties;

/**
 *
 * Creation of ndarrays via classpath discovery.
 *
 *
 * @author Adam Gibson
 */
public class NDArrays {

    private static Class<? extends INDArray> clazz;
    public final static String LINALG_PROPS = "/dl4j-linalg.properties";


    static {
        try {
            ClassPathResource c = new ClassPathResource(LINALG_PROPS);
            Properties props = new Properties();
            props.load(c.getInputStream());

        }catch(Exception e) {
            throw new RuntimeException(e);
        }
    }


    /**
     * Creates an ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @return the instance
     */
    public static INDArray create(int[] shape) {
        try {
            Constructor c = clazz.getConstructor(int[].class);
            return (INDArray) c.newInstance(shape);
        }catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


}
