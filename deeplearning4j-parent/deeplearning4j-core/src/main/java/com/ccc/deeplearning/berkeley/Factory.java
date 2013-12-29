package com.ccc.deeplearning.berkeley;


public interface Factory<T> {
	T newInstance(Object...args);
	public static class DefaultFactory<T> implements Factory<T> {
		private final Class c;
		public DefaultFactory(Class c) {
      this.c = c;
		}
		public T newInstance(Object... args) {
      try {
        return (T) c.newInstance();
      } catch (Exception e) {
        e.printStackTrace();
      }
      return null;
    }
	}
  
}
