package org.deeplearning4j.ui.play;

import play.mvc.Result;

import java.util.function.Supplier;

import static play.mvc.Results.ok;

/**
 * Created by Alex on 08/10/2016.
 */
public class Index implements Supplier<Result> {
    @Override
    public Result get() {
        return ok("Index page goes here");
    }
}
