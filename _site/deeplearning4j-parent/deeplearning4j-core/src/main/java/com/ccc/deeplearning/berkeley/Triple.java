package com.ccc.deeplearning.berkeley;
public class Triple<S,T,U> {
	S first;
	T second;
	U third;
	
	public Triple(S first, T second, U third) {
		this.first = first;
		this.second = second;
		this.third = third;
	}
	
	public S getFirst() {
		return first;
	}



	public void setFirst(S first) {
		this.first = first;
	}



	public T getSecond() {
		return second;
	}



	public void setSecond(T second) {
		this.second = second;
	}



	public U getThird() {
		return third;
	}



	public void setThird(U third) {
		this.third = third;
	}



	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((first == null) ? 0 : first.hashCode());
		result = prime * result + ((second == null) ? 0 : second.hashCode());
		result = prime * result + ((third == null) ? 0 : third.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		final Triple other = (Triple) obj;
		if (first == null) {
			if (other.first != null)
				return false;
		} else if (!first.equals(other.first))
			return false;
		if (second == null) {
			if (other.second != null)
				return false;
		} else if (!second.equals(other.second))
			return false;
		if (third == null) {
			if (other.third != null)
				return false;
		} else if (!third.equals(other.third))
			return false;
		return true;
	}
	
	public String toString() {
		return String.format("(%s,%s,%s)",first,second,third);
	}

	public static <S,T,U> Triple<S,T,U> makeTriple(S s, T t, U u) {
		// TODO Auto-generated method stub
		return new Triple<S, T, U>(s,t,u);
	}
	
}
