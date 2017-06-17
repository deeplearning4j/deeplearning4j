package org.apdplat.word.vector;

/**
 * Created by apple on 7/14/15.
 */
public class F {
    public F() {
        System.out.println("F");
    }

    public static void main(String[] args) {
        new F();
        new C();
    }
}

class C extends F{
    public C(){
        System.out.println("C");
    }
}
