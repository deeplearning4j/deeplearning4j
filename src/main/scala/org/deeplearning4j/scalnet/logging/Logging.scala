package org.deeplearning4j.scalnet.logging

import org.slf4j.Logger
import org.slf4j.LoggerFactory

trait Logging {
  lazy val logger: Logger = { LoggerFactory.getLogger(getClass.getName) }
}
