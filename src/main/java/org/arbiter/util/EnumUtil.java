/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.arbiter.util;

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
