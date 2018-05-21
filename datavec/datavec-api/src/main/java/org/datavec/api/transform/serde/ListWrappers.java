/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.api.transform.serde;

import lombok.Getter;
import org.datavec.api.transform.DataAction;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.filter.Filter;
import org.datavec.api.transform.reduce.IAssociativeReducer;
import org.datavec.api.transform.sequence.SequenceComparator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * A collection of list wrappers to avoid issues with Jackson losing generic type information and hence
 * ignoring the json configuration annotations.<br>
 *
 * These are used internally in {@link BaseSerializer} and should not be used elsewhere
 *
 * @author Alex Black
 */
public class ListWrappers {

    private ListWrappers() {}

    @Getter
    public static class TransformList {
        private List<Transform> list;

        public TransformList(@JsonProperty("list") List<Transform> list) {
            this.list = list;
        }
    }

    @Getter
    public static class FilterList {
        private List<Filter> list;

        public FilterList(@JsonProperty("list") List<Filter> list) {
            this.list = list;
        }
    }

    @Getter
    public static class ConditionList {
        private List<Condition> list;

        public ConditionList(@JsonProperty("list") List<Condition> list) {
            this.list = list;
        }
    }

    @Getter
    public static class ReducerList {
        private List<IAssociativeReducer> list;

        public ReducerList(@JsonProperty("list") List<IAssociativeReducer> list) {
            this.list = list;
        }
    }

    @Getter
    public static class SequenceComparatorList {
        private List<SequenceComparator> list;

        public SequenceComparatorList(@JsonProperty("list") List<SequenceComparator> list) {
            this.list = list;
        }
    }

    @Getter
    public static class DataActionList {
        private List<DataAction> list;

        public DataActionList(@JsonProperty("list") List<DataAction> list) {
            this.list = list;
        }
    }
}
