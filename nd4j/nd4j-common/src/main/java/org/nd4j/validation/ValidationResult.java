package org.nd4j.validation;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.apache.commons.lang3.exception.ExceptionUtils;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Represents a standard way of validating models, files, etc before attempting to load them.
 *
 * @author Alex Black
 */
@AllArgsConstructor
@NoArgsConstructor
@Builder
@Data
public class ValidationResult implements Serializable {

    private String formatType;       //Human readable format/model type
    private Class<?> formatClass;    //Actual class the format/model is (or should be)
    private String path;             //Path of file (if applicable)
    private boolean valid;           //Whether the file/model is valid
    private List<String> issues;     //List of issues (generally only present if not valid)
    private Throwable exception;     //Exception, if applicable



    @Override
    public String toString(){
        List<String> lines = new ArrayList<>();
        if(formatType != null) {
            lines.add("Format type: " + formatType);
        }
        if(formatClass != null){
            lines.add("Format class: " + formatClass.getName());
        }
        if(path != null){
            lines.add("Path: " + path);
        }
        lines.add("Model valid: " + valid);
        if(issues != null && !issues.isEmpty()){
            lines.add("Issues:\n");
            for(String s : issues){
                addWithIndent(s, lines, "- ", "  ");
            }
        }
        if(exception != null){
            String ex = ExceptionUtils.getStackTrace(exception);
            lines.add("Stack Trace:\n");
            addWithIndent(ex, lines, "  ", "  ");
        }
        //Would use String.join but that's Java 8...
        StringBuilder sb = new StringBuilder();
        boolean first = true;
        for(String s : lines){
            if(!first)
                sb.append("\n");
            sb.append(s);
            first = false;
        }
        return sb.toString();
    }

    protected static void addWithIndent(String toAdd, List<String> list, String firstLineIndent, String laterLineIndent){
        String[] split = toAdd.split("\n");
        boolean first = true;
        for(String issueLine : split){
            list.add((first ? firstLineIndent : laterLineIndent) + issueLine);
            first = false;
        }
    }

}
