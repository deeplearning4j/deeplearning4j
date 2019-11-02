package org.nd4j.autodiff.samediff.internal;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.linalg.primitives.Pair;

import java.util.List;

/**
 * A list of dependencies, used in {@link AbstractDependencyTracker}
 *
 * @author Alex Black
 */
@Data
@AllArgsConstructor
public class DependencyList<T, D> {
    private T dependencyFor;
    private List<D> dependencies;
    private List<Pair<D, D>> orDependencies;
}
