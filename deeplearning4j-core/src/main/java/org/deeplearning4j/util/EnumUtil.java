package org.deeplearning4j.util;

/**
 * Created by agibsonccc on 9/3/14.
 */
public class EnumUtil {

    public static <E extends Enum> E parse(String value,Class<E> clazz) {
        int i = Integer.parseInt(value);
        Enum[] constants = clazz.getEnumConstants();
        for(Enum constant : constants) {
            if(constant.ordinal() == i)
                return (E) constant;
        }

        return null;

    }


}
