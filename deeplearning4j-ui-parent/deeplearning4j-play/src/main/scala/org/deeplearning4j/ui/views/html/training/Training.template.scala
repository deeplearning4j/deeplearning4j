
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
  def apply/*1.2*/(i18n: org.deeplearning4j.ui.api.I18N):play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.40*/("""
"""),format.raw/*2.1*/("""<html>
    <head>
        <link rel="stylesheet" type="text/css" href="assets/css/train.css"/><meta http-equiv="Content-Type" content="text/html;charset=UTF-8">
        <title>Document</title>
    </head>
    <body>

        <div class="outerDiv">

            <div class="navDiv">
                <div class="navTopSpacer"></div>
                <div class="navElement">
                    <img src="assets/img/train/nav_icon_home_28.png">
                        """),_display_(/*15.26*/i18n/*15.30*/.getMessage("train.nav.overview")),format.raw/*15.63*/("""
                """),format.raw/*16.17*/("""</div>
                <div class="navElement">
                    <img src="assets/img/train/nav_icon_model_28.png">
                    """),_display_(/*19.22*/i18n/*19.26*/.getMessage("train.nav.model")),format.raw/*19.56*/("""
                """),format.raw/*20.17*/("""</div>
                <div class="navElement">
                    <img src="assets/img/train/nav_icon_system_28.png">
                """),_display_(/*23.18*/i18n/*23.22*/.getMessage("train.nav.hwsw")),format.raw/*23.51*/("""
                """),format.raw/*24.17*/("""</div>
                <div class="navElement">
                    <img src="assets/img/train/nav_icon_help_28.png">
                    """),_display_(/*27.22*/i18n/*27.26*/.getMessage("train.nav.help")),format.raw/*27.55*/("""
                """),format.raw/*28.17*/("""</div>
            </div>
            <div class="contentDiv">
                Content Div<br>
                Language: """),_display_(/*32.28*/i18n/*32.32*/.getDefaultLanguage),format.raw/*32.51*/("""
            """),format.raw/*33.13*/("""</div>
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
                  DATE: Sat Oct 15 10:37:02 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/Training.scala.html
                  HASH: 04e88a5f4d0f4f05043202ec08a253f1f6ed93eb
                  MATRIX: 588->1|721->39|749->41|1256->521|1269->525|1323->558|1369->576|1539->719|1552->723|1603->753|1649->771|1816->911|1829->915|1879->944|1925->962|2094->1104|2107->1108|2157->1137|2203->1155|2356->1281|2369->1285|2409->1304|2451->1318
                  LINES: 20->1|25->1|26->2|39->15|39->15|39->15|40->16|43->19|43->19|43->19|44->20|47->23|47->23|47->23|48->24|51->27|51->27|51->27|52->28|56->32|56->32|56->32|57->33
                  -- GENERATED --
              */
          