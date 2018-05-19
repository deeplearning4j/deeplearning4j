
package org.deeplearning4j.ui.views.html.convolutional

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object Activations_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class Activations extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template0[play.twirl.api.HtmlFormat.Appendable] {

  /**/
  def apply():play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.1*/("""<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="utf-8" />
        <title>Neural Network activations</title>
            <!-- jQuery -->
        <script src="/assets/legacy/jquery-2.2.0.min.js"></script>
        """),format.raw/*8.68*/("""
            """),format.raw/*9.13*/("""<!-- Latest compiled and minified CSS -->
        <link rel="stylesheet" href="/assets/legacy/bootstrap.min.css" />

            <!-- Optional theme -->
        <link rel="stylesheet" href="/assets/legacy/bootstrap-theme.min.css" />

            <!-- Latest compiled and minified JavaScript -->
        <script src="/assets/legacy/bootstrap.min.js" ></script>

        <style>
        body """),format.raw/*19.14*/("""{"""),format.raw/*19.15*/("""
            """),format.raw/*20.13*/("""font-family: 'Roboto', sans-serif;
            color: #333;
            font-weight: 300;
            font-size: 16px;
        """),format.raw/*24.9*/("""}"""),format.raw/*24.10*/("""

        """),format.raw/*26.9*/(""".hd """),format.raw/*26.13*/("""{"""),format.raw/*26.14*/("""
            """),format.raw/*27.13*/("""background-color: #000000;
            font-size: 18px;
            color: #FFFFFF;
        """),format.raw/*30.9*/("""}"""),format.raw/*30.10*/("""

        """),format.raw/*32.9*/(""".block """),format.raw/*32.16*/("""{"""),format.raw/*32.17*/("""
            """),format.raw/*33.13*/("""width: 250px;
            height: 350px;
            display: inline-block;
            border: 1px solid #DEDEDE;
            margin-right: 64px;
        """),format.raw/*38.9*/("""}"""),format.raw/*38.10*/("""

        """),format.raw/*40.9*/(""".hd-small """),format.raw/*40.19*/("""{"""),format.raw/*40.20*/("""
            """),format.raw/*41.13*/("""background-color: #000000;
            font-size: 14px;
            color: #FFFFFF;
        """),format.raw/*44.9*/("""}"""),format.raw/*44.10*/("""
        """),format.raw/*45.9*/("""</style>

        <script type="text/javascript">
        setInterval(function () """),format.raw/*48.33*/("""{"""),format.raw/*48.34*/("""
            """),format.raw/*49.13*/("""var d = new Date();
            $("#pic").removeAttr("src").attr("src", "/activations/data?timestamp=" + new Date().getTime());
        """),format.raw/*51.9*/("""}"""),format.raw/*51.10*/(""", 3000);
        </script>

    </head>



    <body>
        <table style="width: 100%;
            padding: 5px;" class="hd">
            <tbody>
                <tr>
                    <td style="width: 48px;"><a href="/"><img src="/assets/legacy/deeplearning4j.img" border="0"/></a></td>
                    <td>DeepLearning4j UI</td>
                    <td style="width: 128px;">&nbsp; <!-- placeholder for future use --></td>
                </tr>
            </tbody>
        </table>
        <br /> <br />
        <div style="width: 100%;
            text-align: center">
            <div id="embed" style="display: inline-block;"> <!-- style="border: 1px solid #CECECE;" -->
                <img src="/activations/data" id="pic" />
            </div>
        </div>
    </body>

</html>"""))
      }
    }
  }

  def render(): play.twirl.api.HtmlFormat.Appendable = apply()

  def f:(() => play.twirl.api.HtmlFormat.Appendable) = () => apply()

  def ref: this.type = this

}


}

/**/
object Activations extends Activations_Scope0.Activations
              /*
                  -- GENERATED --
                  DATE: Fri May 18 19:33:53 PDT 2018
                  SOURCE: C:/develop/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/convolutional/Activations.scala.html
                  HASH: 72d099499f7738ea1bd20d82a51bc9fed78e75bb
                  MATRIX: 657->0|921->296|962->310|1390->710|1419->711|1461->725|1619->856|1648->857|1687->869|1719->873|1748->874|1790->888|1912->983|1941->984|1980->996|2015->1003|2044->1004|2086->1018|2273->1178|2302->1179|2341->1191|2379->1201|2408->1202|2450->1216|2572->1311|2601->1312|2638->1322|2751->1407|2780->1408|2822->1422|2987->1560|3016->1561
                  LINES: 25->1|32->8|33->9|43->19|43->19|44->20|48->24|48->24|50->26|50->26|50->26|51->27|54->30|54->30|56->32|56->32|56->32|57->33|62->38|62->38|64->40|64->40|64->40|65->41|68->44|68->44|69->45|72->48|72->48|73->49|75->51|75->51
                  -- GENERATED --
              */
          