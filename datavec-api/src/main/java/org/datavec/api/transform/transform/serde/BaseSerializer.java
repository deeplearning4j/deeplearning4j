/*
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

package org.datavec.api.transform.transform.serde;

import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.datatype.joda.JodaModule;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.filter.Filter;
import org.datavec.api.transform.reduce.IReducer;
import org.datavec.api.transform.sequence.SequenceComparator;

import java.util.Arrays;
import java.util.List;

/**
 * Created by Alex on 20/07/2016.
 */
public abstract class BaseSerializer {

    public abstract ObjectMapper getObjectMapper();

    protected ObjectMapper getObjectMapper(JsonFactory factory){
        ObjectMapper om = new ObjectMapper(factory);
        om.registerModule(new JodaModule());
        om.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        om.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        om.enable(SerializationFeature.INDENT_OUTPUT);
        om.setVisibility(PropertyAccessor.ALL, JsonAutoDetect.Visibility.NONE);
        om.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);
        return om;
    }

    private <T> T load(String str, Class<T> clazz){
        ObjectMapper om = getObjectMapper();
        try {
            return om.readValue(str, clazz);
        }catch(Exception e){
            throw new RuntimeException(e);
        }
    }

    private <T> T load(String str, TypeReference<T> typeReference){
        ObjectMapper om = getObjectMapper();
        try {
            return om.readValue(str, typeReference);
        }catch(Exception e){
            throw new RuntimeException(e);
        }
    }

    public String serialize(Object o){
        ObjectMapper om = getObjectMapper();
        try{
            return om.writeValueAsString(o);
        }catch(Exception e){
            throw new RuntimeException(e);
        }
    }

    //===================================================================
    //Wrappers for arrays and lists

    public String serialize(Transform[] transforms){
        return serializeTransformList(Arrays.asList(transforms));
    }

    public String serializeTransformList(List<Transform> list){
        ObjectMapper om = getObjectMapper();
        try{
            return om.writeValueAsString(new ListWrappers.TransformList(list));
        }catch(Exception e){
            throw new RuntimeException(e);
        }
    }


    public String serialize(Filter[] filters){
        return serializeFilterList(Arrays.asList(filters));
    }

    public String serializeFilterList(List<Filter> list){
        ObjectMapper om = getObjectMapper();
        try{
            return om.writeValueAsString(new ListWrappers.FilterList(list));
        }catch(Exception e){
            throw new RuntimeException(e);
        }
    }

    public String serialize(Condition[] conditions){
        return serializeConditionList(Arrays.asList(conditions));
    }

    public String serializeConditionList(List<Condition> list){
        ObjectMapper om = getObjectMapper();
        try{
            return om.writeValueAsString(new ListWrappers.ConditionList(list));
        }catch(Exception e){
            throw new RuntimeException(e);
        }
    }

    public String serialize(IReducer[] reducers){
        return serializeReducerList(Arrays.asList(reducers));
    }

    public String serializeReducerList(List<IReducer> list){
        ObjectMapper om = getObjectMapper();
        try{
            return om.writeValueAsString(new ListWrappers.ReducerList(list));
        }catch(Exception e){
            throw new RuntimeException(e);
        }
    }

    public String serialize(SequenceComparator[] seqComparators){
        return serializeSequenceComparatorList(Arrays.asList(seqComparators));
    }

    public String serializeSequenceComparatorList(List<SequenceComparator> list){
        ObjectMapper om = getObjectMapper();
        try{
            return om.writeValueAsString(new ListWrappers.SequenceComparatorList(list));
        }catch(Exception e){
            throw new RuntimeException(e);
        }
    }


    //======================================================================
    // Deserialization methods

    public Transform deserializeTransform(String str){
        return load(str, Transform.class);
    }

    public Filter deserializeFilter(String str){
        return load(str, Filter.class);
    }

    public Condition deserializeCondition(String str){
        return load(str, Condition.class);
    }

    public IReducer deserializeReducer(String str){
        return load(str, IReducer.class);
    }

    public SequenceComparator deserializeSequenceComparator(String str){
        return load(str, SequenceComparator.class);
    }

    public List<Transform> deserializeTransformList(String str){
        return load(str, ListWrappers.TransformList.class).getList();
    }

    public List<Filter> deserializeFilterList(String str){
        return load(str, ListWrappers.FilterList.class).getList();
    }

    public List<Condition> deserializeConditionList(String str){
        return load(str, ListWrappers.ConditionList.class).getList();
    }

    public List<IReducer> deserializeReducerList(String str){
        return load(str, ListWrappers.ReducerList.class).getList();
    }

    public List<SequenceComparator> deserializeSequenceComparatorList(String str){
        return load(str, ListWrappers.SequenceComparatorList.class).getList();
    }
}
