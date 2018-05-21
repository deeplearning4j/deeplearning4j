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

package org.datavec.api.transform.reduce;

import org.datavec.api.transform.ops.IAggregableReduceOp;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;
import java.util.List;

/**
 * A reducer aggregates or combines
 * a set of examples into
 * a single List<Writable>
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
@JsonSubTypes(value = {@JsonSubTypes.Type(value = Reducer.class, name = "Reducer")})
public interface IAssociativeReducer extends Serializable {

    /**
     *
     * @param schema
     */
    void setInputSchema(Schema schema);

    /**
     *
     * @return
     */
    Schema getInputSchema();

    /**
     *
     * @param schema
     * @return
     */
    Schema transform(Schema schema);

    /**
     * An aggregation that has the property that
     * reduce(List(reduce(List(l1, l2)), l3)) = reduce(List(l1, reduce(List(l2, l3)))
     * @return
     */
    IAggregableReduceOp<List<Writable>, List<Writable>> aggregableReducer();

    /**
     *
     * @return
     */
    List<String> getKeyColumns();

}
