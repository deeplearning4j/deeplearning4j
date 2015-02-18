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

package org.nd4j.linalg.util;

import org.nd4j.linalg.api.complex.IComplexNumber;

import java.io.Serializable;

/**
 * ComplexIterationResult
 *
 * @author Adam Gibson
 */
public class ComplexIterationResult implements Serializable {

    private boolean nextIteration;
    private IComplexNumber number;

    public ComplexIterationResult(boolean nextIteration, IComplexNumber number) {
        this.nextIteration = nextIteration;
        this.number = number;
    }


    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ComplexIterationResult)) return false;

        ComplexIterationResult that = (ComplexIterationResult) o;

        if (nextIteration != that.nextIteration) return false;
        if (number != null ? !number.equals(that.number) : that.number != null) return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = (nextIteration ? 1 : 0);
        result = 31 * result + (number != null ? number.hashCode() : 0);
        return result;
    }

    public boolean isNextIteration() {
        return nextIteration;
    }

    public void setNextIteration(boolean nextIteration) {
        this.nextIteration = nextIteration;
    }

    public IComplexNumber getNumber() {
        return number;
    }

    public void setNumber(IComplexNumber number) {
        this.number = number;
    }
}
