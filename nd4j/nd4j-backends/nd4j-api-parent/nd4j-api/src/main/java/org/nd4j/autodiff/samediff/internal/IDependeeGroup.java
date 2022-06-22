package org.nd4j.autodiff.samediff.internal;

import java.util.Collection;

public interface IDependeeGroup<T> {
    long getId();

    Collection<T> getCollection();

}