package org.nd4j.linalg.profiler.data.array.event.dict;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.profiler.data.array.event.NDArrayEvent;
import org.nd4j.linalg.profiler.data.array.event.NDArrayEventType;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class BreakDownComparison implements Serializable {

    private List<NDArrayEvent> first;
    private List<NDArrayEvent> second;


    public Pair<StackTraceElement,StackTraceElement> stackTracesAt(int i) {
        return Pair.of(first.get(i).getStackTrace()[0], second.get(i).getStackTrace()[0]);
    }

    public Pair<NDArrayEventType,NDArrayEventType> eventTypesAt(int i) {
        return Pair.of(first.get(i).getNdArrayEventType(), second.get(i).getNdArrayEventType());
    }


    public Pair<NDArrayEvent,NDArrayEvent> eventsAt(int i) {
        return Pair.of(first.get(i), second.get(i));
    }

    public Pair<String,String> displayFirstDifference() {
        Pair<NDArrayEvent, NDArrayEvent> diff = firstDifference();
        if(diff != null) {
            return Pair.of(diff.getFirst().getDataAtEvent().getData().toString(), diff.getSecond().getDataAtEvent().getData().toString());
        }
        return null;
    }

    public Pair<NDArrayEvent, NDArrayEvent> firstDifference() {
        for(int i = 0; i < first.size(); i++) {
            if(!first.get(i).equals(second.get(i))) {
                return Pair.of(first.get(i), second.get(i));
            }
        }
        return null;
    }

    public int firstIndexDifference() {
        int ret = -1;
        for(int i = 0; i < first.size(); i++) {
            if(!first.get(i).equals(second.get(i))) {
                ret = i;
                break;
            }
        }
        return ret;
    }

    public static BreakDownComparison empty() {
        return BreakDownComparison.builder()
                .first(new ArrayList<>())
                .second(new ArrayList<>())
                .build();
    }

}
