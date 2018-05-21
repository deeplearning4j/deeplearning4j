package org.datavec.api.transform.schema.conversion;

import org.datavec.api.writable.Writable;

public class TypeConversion {

    private static TypeConversion SINGLETON = new TypeConversion();

    private TypeConversion() {}

    public static TypeConversion getInstance() {
        return SINGLETON;
    }

    public int convertInt(Object o) {
        if(o instanceof Writable) {
            Writable writable = (Writable) o;
            return convertInt(writable);
        }
        else {
            return convertInt(o.toString());
        }
    }



    public int convertInt(Writable writable) {
        return writable.toInt();
    }

    public int convertInt(String o) {
        return Integer.parseInt(o);
    }

    public double convertDouble(Writable writable) {
        return writable.toDouble();
    }

    public double convertDouble(String o) {
        return Double.parseDouble(o);
    }

    public double convertDouble(Object o) {
        if(o instanceof Writable) {
            Writable writable = (Writable) o;
            return convertDouble(writable);
        }
        else {
            return convertDouble(o.toString());
        }
    }


    public float convertFloat(Writable writable) {
        return writable.toFloat();
    }

    public float convertFloat(String o) {
        return Float.parseFloat(o);
    }

    public float convertFloat(Object o) {
        if(o instanceof Writable) {
            Writable writable = (Writable) o;
            return convertFloat(writable);
        }
        else {
            return convertFloat(o.toString());
        }
    }

    public long convertLong(Writable writable) {
        return writable.toLong();
    }


    public long convertLong(String o)  {
        return Long.parseLong(o);
    }

    public long convertLong(Object o) {
        if(o instanceof Writable) {
            Writable writable = (Writable) o;
            return convertLong(writable);
        }
        else {
            return convertLong(o.toString());
        }
    }

    public String convertString(Writable writable) {
        return writable.toString();
    }

    public String convertString(String s) {
        return s;
    }

    public String convertString(Object o) {
        if(o instanceof Writable) {
            Writable writable = (Writable) o;
            return convertString(writable);
        }
        else {
            return convertString(o.toString());
        }
    }

}
