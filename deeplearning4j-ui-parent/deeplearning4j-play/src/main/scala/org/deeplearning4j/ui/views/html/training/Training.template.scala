
package org.deeplearning4j.ui.views.html.training

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object Training_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class Training extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template1[org.deeplearning4j.ui.api.I18N,play.twirl.api.HtmlFormat.Appendable] {

  /**/
  def apply/*1.2*/(i18n:org.deeplearning4j.ui.api.I18N):play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.39*/("""
"""),format.raw/*2.1*/("""<html>
    <head>
        <link rel="stylesheet" type="text/css" href="assets/css/training.css"/><meta http-equiv="Content-Type" content="text/html;charset=UTF-8">
        <title>Document</title>
    </head>
    <body>

        <div class="outerDiv">

            <div class="navDiv">
                <div class="navElement">First Nav Element</div>
                <div class="navElement">Second Nav Element</div>
                
            </div>
            <div class="contentDiv">
                Content Div<br>
                Language: """),_display_(/*18.28*/i18n/*18.32*/.getDefaultLanguage),format.raw/*18.51*/("""
            """),format.raw/*19.13*/("""</div>
        </div>
    </body>
</html>"""))
      }
    }
  }

  def render(i18n:org.deeplearning4j.ui.api.I18N): play.twirl.api.HtmlFormat.Appendable = apply(i18n)

  def f:((org.deeplearning4j.ui.api.I18N) => play.twirl.api.HtmlFormat.Appendable) = (i18n) => apply(i18n)

  def ref: this.type = this

}


}

/**/
object Training extends Training_Scope0.Training
              /*
                  -- GENERATED --
                  DATE: Fri Oct 14 20:50:02 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/Training.scala.html
                  HASH: 1d9d50794144df7a599ca6521e4b9cd591c4f5d0
                  MATRIX: 588->1|720->38|748->40|1337->602|1350->606|1390->625|1432->639
                  LINES: 20->1|25->1|26->2|42->18|42->18|42->18|43->19
                  -- GENERATED --
              */
          