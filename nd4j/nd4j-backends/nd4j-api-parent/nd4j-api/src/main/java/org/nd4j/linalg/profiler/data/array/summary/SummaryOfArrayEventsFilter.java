package org.nd4j.linalg.profiler.data.array.summary;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.memory.WorkspaceUseMetaData;
import org.nd4j.linalg.profiler.data.array.event.NDArrayEvent;
import org.nd4j.linalg.profiler.data.array.event.NDArrayEventType;
import org.nd4j.linalg.profiler.data.stacktrace.StackTraceQuery;

import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Builder
@Data
@NoArgsConstructor
@AllArgsConstructor
public class SummaryOfArrayEventsFilter {

    private WorkspaceUseMetaData.EventTypes targetType;
    private String workspaceName;
    private String threadName;
    private StackTraceQuery stackTraceQuery;
    private Enum associatedWorkspaceEnum;
    private String dataAtEventRegex;
    private NDArrayEventType ndArrayEventType;
    private boolean andModeFilter;



    public boolean meetsCriteria(SummaryOfArrayEvents summaryOfArrayEvents) {
        if(summaryOfArrayEvents.getWorkspaceUseMetaData() != null) {
            for(WorkspaceUseMetaData workspaceUseMetaData : summaryOfArrayEvents.getWorkspaceUseMetaData()) {
                if(targetType != null && workspaceUseMetaData.getEventType() == targetType) {
                    return true;
                }
                if(workspaceName != null && workspaceName.equals(workspaceUseMetaData.getWorkspaceName())) {
                    return true;
                }
                if(threadName != null && threadName.equals(workspaceUseMetaData.getThreadName())) {
                    return true;
                }
                if(stackTraceQuery != null &&
                        StackTraceQuery.stackTraceFillsAnyCriteria(Arrays.asList(stackTraceQuery), workspaceUseMetaData.getStackTrace())) {
                    return true;
                }
                if(associatedWorkspaceEnum != null && associatedWorkspaceEnum.equals(workspaceUseMetaData.getAssociatedEnum())) {
                    return true;
                }
            }
        }

        if(summaryOfArrayEvents.getNdArrayEvents() != null) {
            for(NDArrayEvent ndArrayEvent : summaryOfArrayEvents.getNdArrayEvents()) {
                if(targetType != null && ndArrayEvent.getNdArrayEventType() == ndArrayEventType) {
                    return true;
                }
                if(stackTraceQuery != null && StackTraceQuery.stackTraceFillsAnyCriteria(Arrays.asList(stackTraceQuery), ndArrayEvent.getStackTrace())) {
                    return true;
                }
                if(dataAtEventRegex != null) {
                    Pattern pattern = Pattern.compile(dataAtEventRegex);
                    Matcher m = pattern.matcher(ndArrayEvent.getDataAtEvent().getData());
                    if(m.groupCount() > 0) {
                        return true;
                    }

                }

            }
        }


        return false;
    }


}
