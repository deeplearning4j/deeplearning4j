package org.deeplearning4j.nearestneighbor.server;

import play.libs.F;
import play.mvc.Result;

import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Utility methods for Routing
 *
 * @author Alex Black
 */
public class FunctionUtil {


    public static F.Function0<Result> function0(Supplier<Result> supplier) {
        return supplier::get;
    }

    public static <T> F.Function<T, Result> function(Function<T, Result> function) {
        return function::apply;
    }

}
