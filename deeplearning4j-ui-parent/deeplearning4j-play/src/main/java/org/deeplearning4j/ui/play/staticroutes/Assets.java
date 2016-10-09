package org.deeplearning4j.ui.play.staticroutes;

import play.mvc.Result;

import java.io.File;
import java.util.function.Function;

import static play.mvc.Results.ok;

/**
 * Created by Alex on 09/10/2016.
 */
public class Assets implements Function<String,Result> {

    @Override
    public Result apply(String s) {
        return ok("Received asset request: " + s);
    }
}
