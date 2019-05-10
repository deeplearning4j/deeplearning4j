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

package org.datavec.api.transform.serde;

import org.datavec.api.transform.DataAction;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.filter.Filter;
import org.datavec.api.transform.rank.CalculateSortedRank;
import org.datavec.api.transform.reduce.IAssociativeReducer;
import org.datavec.api.transform.sequence.SequenceComparator;
import org.datavec.api.transform.sequence.SequenceSplit;
import org.nd4j.shade.jackson.annotation.JsonAutoDetect;
import org.nd4j.shade.jackson.annotation.PropertyAccessor;
import org.nd4j.shade.jackson.core.JsonFactory;
import org.nd4j.shade.jackson.core.type.TypeReference;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;
import org.nd4j.shade.jackson.datatype.joda.JodaModule;

import java.util.Arrays;
import java.util.List;

/**
 * Abstract serializer for mapping Transforms, Conditions, Filters, DataActions etc to/from JSON.<br>
 * Also: lists and arrays of these.
 *
 * @author Alex Black
 */
public abstract class BaseSerializer {

    public abstract ObjectMapper getObjectMapper();

    private <T> T load(String str, Class<T> clazz) {
        ObjectMapper om = getObjectMapper();
        try {
            return om.readValue(str, clazz);
        } catch (Exception e) {
            //TODO better exception
            throw new RuntimeException(e);
        }
    }

    private <T> T load(String str, TypeReference<T> typeReference) {
        ObjectMapper om = getObjectMapper();
        try {
            return om.readValue(str, typeReference);
        } catch (Exception e) {
            //TODO better exception
            throw new RuntimeException(e);
        }
    }

    /**
     * Serialize the specified object, such as a {@link Transform}, {@link Condition}, {@link Filter}, etc<br>
     * <b>NOTE:</b> For lists use the list methods, such as {@link #serializeTransformList(List)}<br>
     * <p>
     * To deserialize, use the appropriate method: {@link #deserializeTransform(String)} for example.
     *
     * @param o Object to serialize
     * @return String (json/yaml) representation of the object
     */
    public String serialize(Object o) {
        ObjectMapper om = getObjectMapper();
        try {
            return om.writeValueAsString(o);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    //===================================================================
    //Wrappers for arrays and lists

    public String serialize(Transform[] transforms) {
        return serializeTransformList(Arrays.asList(transforms));
    }

    /**
     * Serialize a list of Transforms
     */
    public String serializeTransformList(List<Transform> list) {
        ObjectMapper om = getObjectMapper();
        try {
            return om.writeValueAsString(new ListWrappers.TransformList(list));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    public String serialize(Filter[] filters) {
        return serializeFilterList(Arrays.asList(filters));
    }

    /**
     * Serialize a list of Filters
     */
    public String serializeFilterList(List<Filter> list) {
        ObjectMapper om = getObjectMapper();
        try {
            return om.writeValueAsString(new ListWrappers.FilterList(list));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public String serialize(Condition[] conditions) {
        return serializeConditionList(Arrays.asList(conditions));
    }

    /**
     * Serialize a list of Conditions
     */
    public String serializeConditionList(List<Condition> list) {
        ObjectMapper om = getObjectMapper();
        try {
            return om.writeValueAsString(new ListWrappers.ConditionList(list));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public String serialize(IAssociativeReducer[] reducers) {
        return serializeReducerList(Arrays.asList(reducers));
    }

    /**
     * Serialize a list of IReducers
     */
    public String serializeReducerList(List<IAssociativeReducer> list) {
        ObjectMapper om = getObjectMapper();
        try {
            return om.writeValueAsString(new ListWrappers.ReducerList(list));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public String serialize(SequenceComparator[] seqComparators) {
        return serializeSequenceComparatorList(Arrays.asList(seqComparators));
    }

    /**
     * Serialize a list of SequenceComparators
     */
    public String serializeSequenceComparatorList(List<SequenceComparator> list) {
        ObjectMapper om = getObjectMapper();
        try {
            return om.writeValueAsString(new ListWrappers.SequenceComparatorList(list));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public String serialize(DataAction[] dataActions) {
        return serializeDataActionList(Arrays.asList(dataActions));
    }

    /**
     * Serialize a list of DataActions
     */
    public String serializeDataActionList(List<DataAction> list) {
        ObjectMapper om = getObjectMapper();
        try {
            return om.writeValueAsString(new ListWrappers.DataActionList(list));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    //======================================================================
    // Deserialization methods

    /**
     * Deserialize a Transform serialized using {@link #serialize(Object)}
     *
     * @param str String representation (YAML/JSON) of the Transform
     * @return Transform
     */
    public Transform deserializeTransform(String str) {
        return load(str, Transform.class);
    }

    /**
     * Deserialize a Filter serialized using {@link #serialize(Object)}
     *
     * @param str String representation (YAML/JSON) of the Filter
     * @return Filter
     */
    public Filter deserializeFilter(String str) {
        return load(str, Filter.class);
    }

    /**
     * Deserialize a Condition serialized using {@link #serialize(Object)}
     *
     * @param str String representation (YAML/JSON) of the Condition
     * @return Condition
     */
    public Condition deserializeCondition(String str) {
        return load(str, Condition.class);
    }

    /**
     * Deserialize an IStringReducer serialized using {@link #serialize(Object)}
     *
     * @param str String representation (YAML/JSON) of the IStringReducer
     * @return IStringReducer
     */
    public IAssociativeReducer deserializeReducer(String str) {
        return load(str, IAssociativeReducer.class);
    }

    /**
     * Deserialize a SequenceComparator serialized using {@link #serialize(Object)}
     *
     * @param str String representation (YAML/JSON) of the SequenceComparator
     * @return SequenceComparator
     */
    public SequenceComparator deserializeSequenceComparator(String str) {
        return load(str, SequenceComparator.class);
    }

    /**
     * Deserialize a CalculateSortedRank serialized using {@link #serialize(Object)}
     *
     * @param str String representation (YAML/JSON) of the CalculateSortedRank
     * @return CalculateSortedRank
     */
    public CalculateSortedRank deserializeSortedRank(String str) {
        return load(str, CalculateSortedRank.class);
    }

    /**
     * Deserialize a SequenceSplit serialized using {@link #serialize(Object)}
     *
     * @param str String representation (YAML/JSON) of the SequenceSplit
     * @return SequenceSplit
     */
    public SequenceSplit deserializeSequenceSplit(String str) {
        return load(str, SequenceSplit.class);
    }

    /**
     * Deserialize a DataAction serialized using {@link #serialize(Object)}
     *
     * @param str String representation (YAML/JSON) of the DataAction
     * @return DataAction
     */
    public DataAction deserializeDataAction(String str) {
        return load(str, DataAction.class);
    }

    /**
     * Deserialize a Transform List serialized using {@link #serializeTransformList(List)}, or
     * an array serialized using {@link #serialize(Transform[])}
     *
     * @param str String representation (YAML/JSON) of the Transform list
     * @return {@code List<Transform>}
     */
    public List<Transform> deserializeTransformList(String str) {
        return load(str, ListWrappers.TransformList.class).getList();
    }

    /**
     * Deserialize a Filter List serialized using {@link #serializeFilterList(List)}, or
     * an array serialized using {@link #serialize(Filter[])}
     *
     * @param str String representation (YAML/JSON) of the Filter list
     * @return {@code List<Filter>}
     */
    public List<Filter> deserializeFilterList(String str) {
        return load(str, ListWrappers.FilterList.class).getList();
    }

    /**
     * Deserialize a Condition List serialized using {@link #serializeConditionList(List)}, or
     * an array serialized using {@link #serialize(Condition[])}
     *
     * @param str String representation (YAML/JSON) of the Condition list
     * @return {@code List<Condition>}
     */
    public List<Condition> deserializeConditionList(String str) {
        return load(str, ListWrappers.ConditionList.class).getList();
    }

    /**
     * Deserialize an IStringReducer List serialized using {@link #serializeReducerList(List)}, or
     * an array serialized using {@link #serialize(IReducer[])}
     *
     * @param str String representation (YAML/JSON) of the IStringReducer list
     * @return {@code List<IStringReducer>}
     */
    public List<IAssociativeReducer> deserializeReducerList(String str) {
        return load(str, ListWrappers.ReducerList.class).getList();
    }

    /**
     * Deserialize a SequenceComparator List serialized using {@link #serializeSequenceComparatorList(List)}, or
     * an array serialized using {@link #serialize(SequenceComparator[])}
     *
     * @param str String representation (YAML/JSON) of the SequenceComparator list
     * @return {@code List<SequenceComparator>}
     */
    public List<SequenceComparator> deserializeSequenceComparatorList(String str) {
        return load(str, ListWrappers.SequenceComparatorList.class).getList();
    }

    /**
     * Deserialize a DataAction List serialized using {@link #serializeDataActionList(List)}, or
     * an array serialized using {@link #serialize(DataAction[])}
     *
     * @param str String representation (YAML/JSON) of the DataAction list
     * @return {@code List<DataAction>}
     */
    public List<DataAction> deserializeDataActionList(String str) {
        return load(str, ListWrappers.DataActionList.class).getList();
    }
}
