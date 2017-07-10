package org.datavec.dataframe.columns;

import org.datavec.dataframe.api.IntColumn;
import org.datavec.dataframe.columns.packeddata.PackedLocalDate;
import org.datavec.dataframe.filtering.*;
import org.datavec.dataframe.filtering.columnbased.ColumnEqualTo;
import org.datavec.dataframe.filtering.dates.LocalDateIsAfter;
import org.datavec.dataframe.filtering.dates.LocalDateIsBefore;
import org.datavec.dataframe.filtering.datetimes.*;
import org.datavec.dataframe.filtering.text.*;
import org.datavec.dataframe.filtering.times.*;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;

/**
 * A reference to a column that can be used in evaluating query predicates. It is a key part of having a fluent API
 * for querying tables.
 * <p>
 * Basically, it lets you write a query like this:
 * <p>
 * table.selectWhere(column("foo").isEqualTo("Bar"));
 * <p>
 * In that example, column() is a static method that returns a ColumnReference for a column named "foo".
 * The method isEqualTo(), is implemented on ColumnReference in a way that it can be applied to potentially, multiple
 * column types, although in this case, it only makes sense for CategoryColumns since the argument is a string.
 * <p>
 * When selectWhere() isExecuted, it supplies the table to the ColumnReference. The ColumnReference uses the table
 * and columnName to get access to the right column, and then fulfils its role by ensuring that the filtering
 * "isEqualTo("Bar") is applied to all the cells in the column.
 */
public class ColumnReference {

    private String columnName;

    public ColumnReference(String column) {
        this.columnName = column;
    }

    public Filter isNotMissing() {
        return new IsNotMissing(this);
    }

    public Filter isMissing() {
        return new IsMissing(this);
    }

    public Filter isEqualTo(int value) {
        return new IntEqualTo(this, value);
    }

    public Filter isEqualTo(ColumnReference reference) {
        return new ColumnEqualTo(this, reference);
    }

    public Filter isBetween(int low, int high) {
        return new IntBetween(this, low, high);
    }

    public Filter isBetween(LocalDate low, LocalDate high) {
        return new LocalDateBetween(this, low, high);
    }

    public Filter isEqualTo(float value) {
        return new FloatEqualTo(this, value);
    }

    public Filter isEqualTo(LocalTime value) {
        return new TimeEqualTo(this, value);
    }

    public Filter isEqualTo(LocalDate value) {
        return new DateEqualTo(this, value);
    }

    public Filter isEqualTo(String value) {
        return new StringEqualTo(this, value);
    }

    public Filter isNotEqualTo(String value) {
        return new StringNotEqualTo(this, value);
    }

    public Filter isGreaterThan(int value) {
        return new IntGreaterThan(this, value);
    }

    public Filter isIn(IntColumn intColumn) {
        return new IntIsIn(this, intColumn);
    }

    public Filter isIn(String... strings) {
        return new TextIsIn(this, strings);
    }

    public Filter isIn(int... ints) {
        return new IntIsIn(this, ints);
    }

    public Filter isLessThan(int value) {
        return new IntLessThan(this, value);
    }

    public Filter isLessThanOrEqualTo(int value) {
        return new IntLessThanOrEqualTo(this, value);
    }

    public Filter isGreaterThanOrEqualTo(int value) {
        return new IntGreaterThanOrEqualTo(this, value);
    }

    public Filter isGreaterThan(float value) {
        return new FloatGreaterThan(this, value);
    }

    public Filter isLessThan(float value) {
        return new FloatLessThan(this, value);
    }

    public Filter isLessThanOrEqualTo(float value) {
        return new FloatLessThanOrEqualTo(this, value);
    }

    public Filter isGreaterThanOrEqualTo(float value) {
        return new FloatGreaterThanOrEqualTo(this, value);
    }

    public String getColumnName() {
        return columnName;
    }

    public Filter isMidnight() {
        return new IsMidnight(this);
    }

    public Filter isNoon() {
        return new IsNoon(this);
    }

    public Filter isBeforeNoon() {
        return new IsBeforeNoon(this);
    }

    public Filter isAfterNoon() {
        return new IsAfterNoon(this);
    }

    public Filter isBefore(LocalTime value) {
        return new IsBefore(this, value);
    }

    public Filter isBefore(LocalDateTime value) {
        return new DateTimeIsBefore(this, value);
    }

    public Filter isAfter(LocalTime value) {
        return new IsAfter(this, value);
    }

    public Filter isSunday() {
        return new IsSunday(this);
    }

    public Filter isMonday() {
        return new IsMonday(this);
    }

    public Filter isTuesday() {
        return new IsTuesday(this);
    }

    public Filter isWednesday() {
        return new IsWednesday(this);
    }

    public Filter isThursday() {
        return new IsThursday(this);
    }

    public Filter isFriday() {
        return new IsFriday(this);
    }

    public Filter isSaturday() {
        return new IsSaturday(this);
    }

    public Filter isInJanuary() {
        return new IsInJanuary(this);
    }

    public Filter isInFebruary() {
        return new IsInFebruary(this);
    }

    public Filter isInMarch() {
        return new IsInMarch(this);
    }

    public Filter isInApril() {
        return new IsInApril(this);
    }

    public Filter isInMay() {
        return new IsInMay(this);
    }

    public Filter isInJune() {
        return new IsInJune(this);
    }

    public Filter isInJuly() {
        return new IsInJuly(this);
    }

    public Filter isInAugust() {
        return new IsInAugust(this);
    }

    public Filter isInSeptember() {
        return new IsInSeptember(this);
    }

    public Filter isInOctober() {
        return new IsInOctober(this);
    }

    public Filter isInNovember() {
        return new IsInNovember(this);
    }

    public Filter isInDecember() {
        return new IsInDecember(this);
    }

    public Filter isInQ1() {
        return new IsInQ1(this);
    }

    public Filter isInQ2() {
        return new IsInQ2(this);
    }

    public Filter isInQ3() {
        return new IsInQ3(this);
    }

    public Filter isInQ4() {
        return new IsInQ4(this);
    }

    public Filter isFirstDayOfMonth() {
        return new IsFirstDayOfTheMonth(this);
    }

    public Filter isLastDayOfMonth() {
        return new IsLastDayOfTheMonth(this);
    }

    public Filter isInYear(int year) {
        return new IsInYear(this, year);
    }

    public Filter isBefore(LocalDate date) {
        return new LocalDateIsBefore(this, PackedLocalDate.pack(date));
    }

    public Filter isAfter(LocalDate date) {
        return new LocalDateIsAfter(this, PackedLocalDate.pack(date));
    }

    public Filter isUpperCase() {
        return new TextIsUpperCase(this);
    }

    public Filter isLowerCase() {
        return new TextIsLowerCase(this);
    }

    public Filter isAlpha() {
        return new TextIsAlpha(this);
    }

    public Filter isAlphaNumeric() {
        return new TextIsAlphaNumeric(this);
    }

    public Filter isNumeric() {
        return new TextIsNumeric(this);
    }

    public Filter isEmpty() {
        return new TextIsEmpty(this);
    }

    public Filter isLongerThan(int length) {
        return new TextIsLongerThan(this, length);
    }

    public Filter isShorterThan(int length) {
        return new TextIsShorterThan(this, length);
    }

    public Filter hasLengthEqualTo(int length) {
        return new TextHasLengthEqualTo(this, length);
    }

    public Filter equalToIgnoringCase(String string) {
        return new TextEqualToIgnoringCase(this, string);
    }

    public Filter startsWith(String string) {
        return new TextStartsWith(this, string);
    }

    public Filter endsWith(String string) {
        return new TextEndsWith(this, string);
    }

    public Filter contains(String string) {
        return new TextContains(this, string);
    }

    public Filter matchesRegex(String string) {
        return new TextMatchesRegex(this, string);
    }

    public Filter isTrue() {
        return new BooleanIsTrue(this);
    }

    public Filter isFalse() {
        return new BooleanIsFalse(this);
    }
}
