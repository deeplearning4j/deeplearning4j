package org.deeplearning4j.keras;

import com.google.common.base.Predicate;

import edu.umd.cs.findbugs.annotations.Nullable;

public class StringsEndsWithPredicate implements Predicate<String> {

    private final String anEnd;

    public StringsEndsWithPredicate(String anEnd) {
        this.anEnd = anEnd;
    }

    @Override
    public boolean apply(@Nullable String s) {
        return s.endsWith(anEnd);
    }

    public static Predicate<String> endsWith(String anEnd) {
        return new StringsEndsWithPredicate(anEnd);
    }
}
