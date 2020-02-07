package org.nd4j.autodiff.samediff.optimize;

import java.util.List;

/**
 *
 * @author Alex Black
 */
public interface OptimizerSet {

    List<Optimizer> getOptimizers();

}
