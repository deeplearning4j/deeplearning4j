package org.deeplearning4j.ui.play.misc;

import lombok.AllArgsConstructor;
import play.libs.F;
import play.mvc.Result;

import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Created by Alex on 10/10/2016.
 */
public class FunctionUtil {

    public static F.Function0<Result> function0(Supplier<Result> supplier){
        return supplier::get;
    }

    public static <T> F.Function<T,Result> function(Function<T,Result> function){
        return function::apply;
    }

    @AllArgsConstructor
    public static class Function0 implements F.Function0<Result>{
        private final Supplier<Result> supplier;
        @Override
        public Result apply() throws Throwable {
            return supplier.get();
        }
    }

}
