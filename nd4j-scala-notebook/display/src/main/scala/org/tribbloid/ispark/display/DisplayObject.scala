package org.tribbloid.ispark.display

trait DisplayObject

trait HTMLDisplayObject extends DisplayObject {
  def toHTML: String
}
//
//trait MarkdownDisplayObject extends DisplayObject {
//  def toMarkdown: String
//}

trait LatexDisplayObject extends DisplayObject {
  def toLatex: String
}

trait JSONDisplayObject extends DisplayObject {
  def toJSON: String
}

trait JavascriptDisplayObject extends DisplayObject {
  def toJavascript: String
}

trait SVGDisplayObject extends DisplayObject {
  def toSVG: String
}

trait PNGDisplayObject extends DisplayObject {
  def toPNG: String
}

trait JPEGDisplayObject extends DisplayObject {
  def toJPEG: String
}
