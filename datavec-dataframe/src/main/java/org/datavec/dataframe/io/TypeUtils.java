package org.datavec.dataframe.io;

import org.datavec.dataframe.api.ColumnType;
import org.datavec.dataframe.api.BooleanColumn;
import org.datavec.dataframe.api.CategoryColumn;
import org.datavec.dataframe.columns.Column;
import org.datavec.dataframe.api.DateColumn;
import org.datavec.dataframe.api.DateTimeColumn;
import org.datavec.dataframe.api.FloatColumn;
import org.datavec.dataframe.api.IntColumn;
import org.datavec.dataframe.api.LongColumn;
import org.datavec.dataframe.api.ShortColumn;
import org.datavec.dataframe.api.TimeColumn;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;

import edu.umd.cs.findbugs.annotations.NonNull;
import net.jcip.annotations.Immutable;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeFormatterBuilder;
import java.time.format.DateTimeParseException;
import java.util.Arrays;
import java.util.List;

/**
 * Utilities for working with {@link ColumnType}s
 */
@Immutable
public final class TypeUtils {

    /**
     * Private constructor to prevent instantiation
     */
    private TypeUtils() {}

    // These Strings will convert to true booleans
    public static final List<String> TRUE_STRINGS = Arrays.asList("T", "t", "Y", "y", "TRUE", "true", "1");

    // A more restricted set of 'true' strings that is used for column type detection
    public static final List<String> TRUE_STRINGS_FOR_DETECTION = Arrays.asList("T", "t", "Y", "y", "TRUE", "true");

    // These Strings will convert to false booleans
    public static final List<String> FALSE_STRINGS = Arrays.asList("F", "f", "N", "n", "FALSE", "false", "0");

    // A more restricted set of 'false' strings that is used for column type detection
    public static final List<String> FALSE_STRINGS_FOR_DETECTION = Arrays.asList("F", "f", "N", "n", "FALSE", "false");

    // Formats that we accept in parsing dates from strings
    // TODO: Add more types, especially dates with month names spelled-out fully.
    private static final DateTimeFormatter dtf1 = DateTimeFormatter.ofPattern("yyyyMMdd");
    private static final DateTimeFormatter dtf2 = DateTimeFormatter.ofPattern("MM/dd/yyyy");
    private static final DateTimeFormatter dtf3 = DateTimeFormatter.ofPattern("MM-dd-yyyy");
    private static final DateTimeFormatter dtf4 = DateTimeFormatter.ofPattern("MM.dd.yyyy");
    private static final DateTimeFormatter dtf5 = DateTimeFormatter.ofPattern("yyyy-MM-dd");
    private static final DateTimeFormatter dtf6 = DateTimeFormatter.ofPattern("yyyy/MM/dd");
    private static final DateTimeFormatter dtf7 = DateTimeFormatter.ofPattern("dd/MMM/yyyy");
    private static final DateTimeFormatter dtf8 = DateTimeFormatter.ofPattern("dd-MMM-yyyy");
    private static final DateTimeFormatter dtf9 = DateTimeFormatter.ofPattern("M/d/yyyy");
    private static final DateTimeFormatter dtf10 = DateTimeFormatter.ofPattern("M/d/yy");
    private static final DateTimeFormatter dtf11 = DateTimeFormatter.ofPattern("MMM/dd/yyyy");
    private static final DateTimeFormatter dtf12 = DateTimeFormatter.ofPattern("MMM-dd-yyyy");
    private static final DateTimeFormatter dtf13 = DateTimeFormatter.ofPattern("MMM/dd/yy");
    private static final DateTimeFormatter dtf14 = DateTimeFormatter.ofPattern("MMM-dd-yy");
    private static final DateTimeFormatter dtf15 = DateTimeFormatter.ofPattern("MMM/dd/yyyy");
    private static final DateTimeFormatter dtf16 = DateTimeFormatter.ofPattern("MMM/d/yyyy");
    private static final DateTimeFormatter dtf17 = DateTimeFormatter.ofPattern("MMM-dd-yy");
    private static final DateTimeFormatter dtf18 = DateTimeFormatter.ofPattern("MMM dd, yyyy");
    private static final DateTimeFormatter dtf19 = DateTimeFormatter.ofPattern("MMM d, yyyy");

    /**
     * List of formatters for use in code that selects the correct one for a given Date string
     */
    private static ImmutableList<DateTimeFormatter> dateFormatters = ImmutableList.of(dtf1, dtf2, dtf3, dtf4, dtf5,
                    dtf6, dtf7, dtf8, dtf9, dtf10, dtf11, dtf12, dtf13, dtf14, dtf15, dtf16, dtf17, dtf18, dtf19);

    private static final DateTimeFormatter dtTimef0 = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
    private static final DateTimeFormatter dtTimef1 = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS");
    private static final DateTimeFormatter dtTimef2 = DateTimeFormatter.ofPattern("MM/dd/yyyy hh:mm:ss a");
    private static final DateTimeFormatter dtTimef3 = DateTimeFormatter.ofPattern("dd-MMM-yyyy HH:mm");
    private static final DateTimeFormatter dtTimef4 = DateTimeFormatter.ofPattern("dd-MMM-yyyy HH:mm");
    private static final DateTimeFormatter dtTimef5 = DateTimeFormatter.ISO_LOCAL_DATE_TIME;

    private static final DateTimeFormatter dtTimef6;

    static {
        dtTimef6 = new DateTimeFormatterBuilder().parseCaseInsensitive().append(DateTimeFormatter.ISO_LOCAL_DATE_TIME)
                        .appendLiteral('.').appendPattern("SSS").toFormatter();
    }

    /**
     * List of formatters for use in code that selects the correct one for a given DateTime string
     */
    private static ImmutableList<DateTimeFormatter> dateTimeFormatters =
                    ImmutableList.of(dtTimef0, dtTimef1, dtTimef2, dtTimef3, dtTimef4, dtTimef5, dtTimef6);


    private static final DateTimeFormatter timef1 = DateTimeFormatter.ofPattern("HH:mm:ss.SSS");
    private static final DateTimeFormatter timef2 = DateTimeFormatter.ofPattern("hh:mm:ss a");
    private static final DateTimeFormatter timef3 = DateTimeFormatter.ofPattern("h:mm:ss a");
    private static final DateTimeFormatter timef4 = DateTimeFormatter.ISO_LOCAL_TIME;
    private static final DateTimeFormatter timef5 = DateTimeFormatter.ofPattern("hh:mm a");
    private static final DateTimeFormatter timef6 = DateTimeFormatter.ofPattern("h:mm a");
    private static final DateTimeFormatter timef7 = DateTimeFormatter.ofPattern("HHmm");

    /**
     * List of formatters for use in code that selects the correct one for a given Time string
     */
    private static ImmutableList<DateTimeFormatter> timeFormatters =
                    ImmutableList.of(timef1, timef2, timef3, timef4, timef5, timef6
                    //, timef7
                    );

    // A formatter that handles all the date formats defined above
    public static final DateTimeFormatter DATE_FORMATTER = new DateTimeFormatterBuilder().appendOptional(dtf1)
                    .appendOptional(dtf2).appendOptional(dtf2).appendOptional(dtf3).appendOptional(dtf4)
                    .appendOptional(dtf5).appendOptional(dtf6).appendOptional(dtf7).appendOptional(dtf8)
                    .appendOptional(dtf9).appendOptional(dtf10).appendOptional(dtf11).appendOptional(dtf12)
                    .appendOptional(dtf13).appendOptional(dtf14).appendOptional(dtf15).appendOptional(dtf16)
                    .appendOptional(dtf17).appendOptional(dtf18).appendOptional(dtf19).toFormatter();

    // A formatter that handles date time formats defined above
    public static final DateTimeFormatter DATE_TIME_FORMATTER = new DateTimeFormatterBuilder().appendOptional(dtTimef2)
                    .appendOptional(dtTimef3).appendOptional(dtTimef4).appendOptional(dtTimef1).appendOptional(dtTimef0)
                    .appendOptional(dtTimef5).appendOptional(dtTimef6).toFormatter();

    // A formatter that handles time formats defined above
    public static final DateTimeFormatter TIME_FORMATTER = new DateTimeFormatterBuilder().appendOptional(timef5)
                    .appendOptional(timef2).appendOptional(timef3).appendOptional(timef1).appendOptional(timef4)
                    .appendOptional(timef6).appendOptional(timef7).toFormatter();

    // A formatter that handles time formats defined above used for type detection.
    // It is more conservative than the converter
    public static final DateTimeFormatter TIME_DETECTION_FORMATTER =
                    new DateTimeFormatterBuilder().appendOptional(timef5).appendOptional(timef2).appendOptional(timef3)
                                    .appendOptional(timef1).appendOptional(timef4).appendOptional(timef6)
                                    //  .appendOptional(timef7)
                                    .toFormatter();

    /**
     * Strings representing missing values in, for example, a CSV file that is being imported
     */
    private static final String missingInd1 = "NaN";
    private static final String missingInd2 = "*";
    private static final String missingInd3 = "NA";
    private static final String missingInd4 = "null";

    public static final ImmutableList<String> MISSING_INDICATORS =
                    ImmutableList.of(missingInd1, missingInd2, missingInd3, missingInd4);

    /**
     * Constructs and returns a column for the given {@code name} and {@code type}
     */
    public static Column newColumn(@NonNull String name, @NonNull ColumnType type) {

        Preconditions.checkArgument(!Strings.isNullOrEmpty(name), "There must be a valid name for a new column");

        Preconditions.checkArgument(type != ColumnType.SKIP,
                        "SKIP-ped columns should be handled outside of this method.");

        switch (type) {
            case LOCAL_DATE:
                return DateColumn.create(name);
            case LOCAL_TIME:
                return TimeColumn.create(name);
            case LOCAL_DATE_TIME:
                return DateTimeColumn.create(name);
            case INTEGER:
                return IntColumn.create(name);
            case FLOAT:
                return FloatColumn.create(name);
            case BOOLEAN:
                return BooleanColumn.create(name);
            case CATEGORY:
                return CategoryColumn.create(name);
            case SHORT_INT:
                return ShortColumn.create(name);
            case LONG_INT:
                return LongColumn.create(name);
            default:
                throw new IllegalArgumentException("Unknown ColumnType: " + type);
        }
    }

    /**
     * Returns the first DateTimeFormatter to parse the string, which represents a DATE
     * <p>
     * It's intended to be called at the start of a large formatting job so that it picks the write format and is not
     * called again. This is an optimization, because the older version, which will try multiple formatters was too
     * slow for large data sets.
     */
    public static DateTimeFormatter getDateFormatter(String dateValue) {

        for (DateTimeFormatter formatter : dateFormatters) {
            try {
                formatter.parse(dateValue);
                return formatter;
            } catch (DateTimeParseException e) {
                // ignore;
            }
        }
        return DATE_FORMATTER;
    }

    /**
     * Returns the first DateTimeFormatter to parse the string, which represents a DATE_TIME
     * <p>
     * It's intended to be called at the start of a large formatting job so that it picks the write format and is not
     * called again. This is an optimization, because the older version, which will try multiple formatters was too
     * slow for large data sets.
     */
    public static DateTimeFormatter getDateTimeFormatter(String dateTimeValue) {

        for (DateTimeFormatter formatter : dateTimeFormatters) {
            try {
                formatter.parse(dateTimeValue);
                return formatter;
            } catch (DateTimeParseException e) {
                // ignore;
            }
        }
        return DATE_FORMATTER;
    }

    /**
     * Returns the first DateTimeFormatter to parse the string, which represents a TIME
     * <p>
     * It's intended to be called at the start of a large formatting job so that it picks the write format and is not
     * called again. This is an optimization, because the older version, which will try multiple formatters was too
     * slow for large data sets.
     */
    public static DateTimeFormatter getTimeFormatter(String timeValue) {

        for (DateTimeFormatter formatter : timeFormatters) {
            try {
                formatter.parse(timeValue);
                return formatter;
            } catch (DateTimeParseException e) {
                // ignore;
            }
        }
        return DATE_FORMATTER;
    }
}
