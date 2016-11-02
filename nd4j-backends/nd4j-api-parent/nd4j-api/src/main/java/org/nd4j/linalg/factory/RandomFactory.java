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

    /**
     * This method returns Random implementation instance associated with calling thread
     *
     * @return object implementing Random interface
     */
    public Random getRandom() {
        try {
            if (threadRandom.get() == null) {
                Random t = (Random) randomClass.newInstance();
                if (t.getStatePointer() != null) {
                    // TODO: attach this thing to deallocator
                    // if it's stateless random - we just don't care then
                }
                threadRandom.set(t);
                return t;
            }


            return threadRandom.get();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * This method returns new onject implementing Random interface, initialized with System.currentTimeMillis() as seed
     *
     * @return object implementing Random interface
     */
    public Random getNewRandomInstance() {
        return getNewRandomInstance(System.currentTimeMillis());
    }


    /**
     * This method returns new onject implementing Random interface, initialized with seed value
     *
     * @return object implementing Random interface
     */
    public Random getNewRandomInstance(long seed) {
        try {
            Random t = (Random) randomClass.newInstance();
            if (t.getStatePointer() != null) {
                // TODO: attach this thing to deallocator
                // if it's stateless random - we just don't care then
            }
            t.setSeed(seed);
            return t;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
