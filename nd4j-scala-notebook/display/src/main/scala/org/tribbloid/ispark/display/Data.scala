package org.tribbloid.ispark.display

case class Data(items: (MIME, String)*) {
    def apply(mime: MIME): Option[String] = items.find(_._1 == mime).map(_._2)

    def default: Option[String] = apply(MIME.`text/plain`)
}

object Data {

  def parse(obj: Any): Data = {

    var list = MIME.all.flatMap(mime => mime.parse(obj).map(string => mime->string))
//    if (list.map(_._1).contains(MIME.`text/plain`) && list.size >= 2) {
//      list = list.filterNot(_._1 == MIME.`text/plain`)
//    }
    new Data(list: _*)
  }
}