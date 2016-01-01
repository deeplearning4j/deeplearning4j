package org.arbiter.optimize.parameter;

/**ModelParameterSpace: defines the acceptable ranges of values a given parameter may take
 */
public interface ParameterSpace<P> {

    P randomValue();

}
