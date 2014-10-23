package org.deeplearning4j.iterativereduce.actor.core;

import java.io.Serializable;

public class NeedsModelMessage  implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1282144180750694842L;
	private String id;
	
	
	public NeedsModelMessage(String id) {
		super();
		this.id = id;
	}

	public synchronized String getId() {
		return id;
	}
	
	public  void setId(String id) {
		this.id = id;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((id == null) ? 0 : id.hashCode());
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
		NeedsModelMessage other = (NeedsModelMessage) obj;
		if (id == null) {
			if (other.id != null)
				return false;
		} else if (!id.equals(other.id))
			return false;
		return true;
	}
	
	
	
}
