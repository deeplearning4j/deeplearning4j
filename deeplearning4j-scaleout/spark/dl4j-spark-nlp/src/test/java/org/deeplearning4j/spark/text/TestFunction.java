package org.deeplearning4j.spark.text;

import org.apache.spark.api.java.function.Function;

import java.util.List;

/**
 * Created by jeffreytang on 8/14/15.
 */
public class TestFunction implements Function<Integer, Integer> {
    public TestFunction(List<Integer> lst) {
        this.lst = lst;
    }

    public List<Integer> getLst() {
        return lst;
    }

    public int getA() {
        return a;
    }

    private List<Integer> lst;
    private int a;


    @Override
    public Integer call(Integer i) {
        lst.add(i);
        a = 1000;
        return i + 1;
    }
}

