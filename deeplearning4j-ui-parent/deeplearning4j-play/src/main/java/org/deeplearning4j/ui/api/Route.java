package org.deeplearning4j.ui.api;

import lombok.AllArgsConstructor;
import lombok.Data;
import play.libs.F;
import play.mvc.Result;

import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Created by Alex on 08/10/2016.
 */
@Data
@AllArgsConstructor
public class Route {
    private final String route;
    private final HttpMethod httpMethod;
    private final FunctionType functionType;
    private final Supplier<Result> supplier;
    private final Function<String, Result> function;
    private final BiFunction<String, String, Result> function2;

    public Route(String route, HttpMethod method, FunctionType functionType, Supplier<Result> supplier) {
        this(route, method, functionType, supplier, null, null);
    }

    public Route(String route, HttpMethod method, FunctionType functionType, Function<String, Result> function) {
        this(route, method, functionType, null, function, null);
    }

    public Route(String route, HttpMethod method, FunctionType functionType, BiFunction<String,String,Result> function) {
        this(route, method, functionType, null, null, function);
    }
}
