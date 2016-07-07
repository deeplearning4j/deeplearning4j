package org.datavec.api.util;

import java.util.concurrent.TimeUnit;

/**
 * Created by Alex on 2/06/2016.
 */
public class TransformUtils {

    public static String timeUnitToString(long time, TimeUnit unit){
        String str = String.valueOf(time);
        switch (unit){
            case MILLISECONDS:
                str += "Millisecond";
                break;
            case SECONDS:
                str += "Second";
                break;
            case MINUTES:
                str += "Minute";
                break;
            case HOURS:
                str += "Hour";
                break;
            case DAYS:
                str += "Day";
                break;
            default:
                throw new RuntimeException();
        }
        if(time == 1) return str;
        return str + "s";
    }

    public static TimeUnit stringToTimeUnit(String str){
        switch(str.toLowerCase()){
            case "ms":
            case "millisecond":
            case "milliseconds":
                return TimeUnit.MILLISECONDS;
            case "s":
            case "sec":
            case "second":
            case "seconds":
                return TimeUnit.SECONDS;
            case "min":
            case "minute":
            case "minutes":
                return TimeUnit.MINUTES;
            case "h":
            case "hour":
            case "hours":
                return TimeUnit.HOURS;
            case "day":
            case "days":
                return TimeUnit.DAYS;
            default:
                throw new RuntimeException("Unknown time unit: \"" + str + "\"");
        }
    }

}
