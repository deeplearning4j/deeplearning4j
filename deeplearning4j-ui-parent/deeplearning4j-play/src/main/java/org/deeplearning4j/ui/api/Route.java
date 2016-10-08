package org.deeplearning4j.ui.api;

import lombok.AllArgsConstructor;
import lombok.Data;
import play.mvc.Result;

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

}
