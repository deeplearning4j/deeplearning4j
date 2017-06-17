package org.apdplat.word.vector;

import java.util.*;

/**
 * Created by apple on 7/14/15.
 */
public class T {
    public static void main(String[] args) {
        String s1="hw";
        String s2=new String("hw");
        final String s3="h";
        String s4=s3+"w";
        String s5=new String("h")+"w";
        String s6="h"+"w";
        System.out.println(s1==s2);
        System.out.println(s1==s4);
        System.out.println(s1==s5);
        System.out.println(s1==s6);

        Integer a=1;
        Integer b=1;
        Integer c=200;
        Integer d=200;

        System.out.println(a==b);
        System.out.println(c == d);

        List<String> list = new ArrayList<>();
        list.add("one");
        list.add(null);
        list.add("two");
        list.add(null);
        list.add("three");
        list.forEach(i -> System.out.println(i));

        list = new LinkedList<>();
        list.add("one");
        list.add(null);
        list.add("two");
        list.add(null);
        list.add("three");
        list.forEach(i -> System.out.println(i));

        list = new Vector<>();
        Stack stack = new Stack();
    }
}
