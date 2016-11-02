package org.nd4j.linalg.factory;

import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

/**
 * This class acts as factory for new Random objects and thread-isolated holder for previously created Random instances
 *
 * @author raver119@gmail.com
 */
public class RandomFactory {
    private ThreadLocal<Random> threadRandom = new ThreadLocal<>();
    private Class randomClass;

    public RandomFactory(Class randomClass) {
        this.randomClass = randomClass;
    }


    public Random getRandom() {
        try {
            if (threadRandom.get() == null) {
                Random t = (Random) randomClass.newInstance();
                if (t.getStatePointer() != null) {
                    // TODO: attach this thing to deallocator
                    // if it's stateless random - we just don't care then
                }
                threadRandom.set(t);
            }


            return threadRandom.get();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
