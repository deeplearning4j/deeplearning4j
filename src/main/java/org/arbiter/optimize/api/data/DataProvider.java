package org.arbiter.optimize.api.data;

public interface DataProvider<D> {

    D trainData();

    D testData();

}
