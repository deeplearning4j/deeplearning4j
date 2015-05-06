/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.scaleout.actor.core;

import java.util.ArrayList;
import java.util.List;

import akka.actor.ActorRef;
import akka.actor.Cancellable;
import akka.actor.UntypedActor;
import akka.cluster.Cluster;
import akka.cluster.ClusterEvent.MemberEvent;
import akka.cluster.ClusterEvent.MemberRemoved;
import akka.cluster.ClusterEvent.MemberUp;
import akka.cluster.ClusterEvent.UnreachableMember;
import akka.contrib.pattern.DistributedPubSubExtension;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.event.Logging;
import akka.event.LoggingAdapter;

public class ClusterListener extends UntypedActor {
	LoggingAdapter log = Logging.getLogger(getContext().system(), this);
	Cluster cluster = Cluster.get(getContext().system());
	protected ActorRef mediator = DistributedPubSubExtension.get(getContext().system()).mediator();
	public final static String TOPICS = "topics";
	private List<String> topics = new ArrayList<String>();
	private Cancellable topicTask;
	//subscribe to cluster changes
	@Override
	public void preStart() {
		//#subscribe
		cluster.subscribe(getSelf(), MemberEvent.class);
		//replicate the network
		mediator.tell(new DistributedPubSubMediator.Subscribe(TOPICS,getSelf()), getSelf());
		log.info("Subscribed to cluster events");
		/*topicTask = context().system().scheduler().schedule(Duration.createComplex(10,TimeUnit.SECONDS), Duration.createComplex(10,TimeUnit.SECONDS), new Runnable() {

			@Override
			public void run() {
				log.info("Current topics " + topics);
				//reply
				mediator.tell(new DistributedPubSubMediator.Publish(ClusterListener.TOPICS,
						topics), getSelf());
			}

		}, context().dispatcher());
		 */


		//#subscribe
	}

	//re-subscribe when restart
	@Override
	public void postStop() {
		cluster.unsubscribe(getSelf());
		log.info("UnSubscribed to cluster events");
		if(topicTask != null)
			topicTask.cancel();


	}

	@Override
	public void onReceive(Object message) {
		if (message instanceof MemberUp) {
			MemberUp mUp = (MemberUp) message;
			log.info("Member is Up: {}", mUp.member());

		} else if (message instanceof UnreachableMember) {
			UnreachableMember mUnreachable = (UnreachableMember) message;
			log.info("Member detected as unreachable: {}", mUnreachable.member());

		} else if (message instanceof MemberRemoved) {
			MemberRemoved mRemoved = (MemberRemoved) message;
			log.info("Member is Removed: {}", mRemoved.member());

		} else if (message instanceof MemberEvent) {
			// ignore

		} 

		else if(message instanceof DistributedPubSubMediator.SubscribeAck) {
			DistributedPubSubMediator.SubscribeAck ack = (DistributedPubSubMediator.SubscribeAck) message;
			topics.add(ack.subscribe().topic());
		}

		else if(message instanceof DistributedPubSubMediator.UnsubscribeAck) {
			DistributedPubSubMediator.UnsubscribeAck unsub = (DistributedPubSubMediator.UnsubscribeAck) message;
			topics.remove(unsub.unsubscribe().topic());
		}

		else if(message instanceof List) {
			log.info("Topics sent " + message);
		}


		else {
			unhandled(message);
		}

	}
}