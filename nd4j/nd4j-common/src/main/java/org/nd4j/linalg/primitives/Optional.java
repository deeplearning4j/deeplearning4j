/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.primitives;

import lombok.EqualsAndHashCode;
import lombok.NonNull;

import java.util.NoSuchElementException;

/**
 * Simple Optional class, based loosely on Java 8's optional class
 *
 * @param <T> Type for optional
 * @author Alex Black
 */
@EqualsAndHashCode
public class Optional<T> {
    private static final Optional EMPTY = new Optional();

    private final T value;

    private Optional(){
        this(null);
    }

    private Optional(T value){
        this.value = value;
    }

    /**
     * Returns an empty Optional instance. No value is present for this Optional.
     *
     */
    public static <T> Optional<T> empty(){
        return (Optional<T>)EMPTY;
    }

    /**
     * Returns an Optional with the specified present non-null value.
     *
     * @param value the value to be present, which must be non-null
     * @return an Optional with the value present
     */
    public static <T> Optional<T> of(@NonNull T value){
        return new Optional<>(value);
    }

    /**
     * Returns an Optional describing the specified value, if non-null, otherwise returns an empty Optional.
     *
     * @param value the possibly-null value to describe
     * @return an Optional with a present value if the specified value is non-null, otherwise an empty Optional
     */
    public static <T> Optional<T> ofNullable(T value){
        if(value == null){
            return empty();
        }
        return new Optional<>(value);
    }

    /**
     * If a value is present in this Optional, returns the value, otherwise throws NoSuchElementException.
     *
     * @return the non-null value held by this Optional
     * @throws NoSuchElementException - if there is no value present
     */
    public T get(){
        if (!isPresent()) {
            throw new NoSuchElementException("Optional is empty");
        }
        return value;
    }

    /**
     * Return true if there is a value present, otherwise false.
     *
     * @return true if there is a value present, otherwise false
     */
    public boolean isPresent(){
        return value != null;
    }

    /**
     * Return the value if present, otherwise return other.
     *
     * @param other  the value to be returned if there is no value present, may be null
     * @return
     */
    public T orElse(T other){
        if(isPresent()){
            return get();
        }
        return other;
    }

    public String toString(){
        if(isPresent()){
            return "Optional(" + value.toString() + ")";
        }
        return "Optional()";
    }
}
