package org.arbiter.optimize.api;

/**Model(Hyper)ParameterSpace: defines the set of valid ranges
 */
public interface ModelParameterSpace<T> {

    T randomCandidate();


}
