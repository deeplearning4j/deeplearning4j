
package org.deeplearning4j.ui.views.html.histogram

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object Histogram_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class Histogram extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template0[play.twirl.api.HtmlFormat.Appendable] {

  /**/
  def apply():play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.1*/("""<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8"/>
        <title>Weights/Gradients/Score</title>
        <style>
        .bar rect """),format.raw/*7.19*/("""{"""),format.raw/*7.20*/("""
            """),format.raw/*8.13*/("""fill: steelblue;
            shape-rendering: crispEdges;
        """),format.raw/*10.9*/("""}"""),format.raw/*10.10*/("""

        """),format.raw/*12.9*/(""".bar text """),format.raw/*12.19*/("""{"""),format.raw/*12.20*/("""
            """),format.raw/*13.13*/("""fill: #EFEFEF;
        """),format.raw/*14.9*/("""}"""),format.raw/*14.10*/("""

        """),format.raw/*16.9*/(""".area """),format.raw/*16.15*/("""{"""),format.raw/*16.16*/("""
            """),format.raw/*17.13*/("""fill: steelblue;
        """),format.raw/*18.9*/("""}"""),format.raw/*18.10*/("""

        """),format.raw/*20.9*/(""".axis path, .axis line """),format.raw/*20.32*/("""{"""),format.raw/*20.33*/("""
            """),format.raw/*21.13*/("""fill: none;
            stroke: #000;
            shape-rendering: crispEdges;
        """),format.raw/*24.9*/("""}"""),format.raw/*24.10*/("""

        """),format.raw/*26.9*/(""".tick line """),format.raw/*26.20*/("""{"""),format.raw/*26.21*/("""
            """),format.raw/*27.13*/("""opacity: 0.2;
            shape-rendering: crispEdges;
        """),format.raw/*29.9*/("""}"""),format.raw/*29.10*/("""

        """),format.raw/*31.9*/("""path """),format.raw/*31.14*/("""{"""),format.raw/*31.15*/("""
            """),format.raw/*32.13*/("""stroke: steelblue;
            stroke-width: 2;
            fill: none;
        """),format.raw/*35.9*/("""}"""),format.raw/*35.10*/("""

        """),format.raw/*37.9*/(""".legend """),format.raw/*37.17*/("""{"""),format.raw/*37.18*/("""
            """),format.raw/*38.13*/("""font-size: 12px;
            text-anchor: middle;
        """),format.raw/*40.9*/("""}"""),format.raw/*40.10*/("""

        """),format.raw/*42.9*/(""".brush .extent """),format.raw/*42.24*/("""{"""),format.raw/*42.25*/("""
            """),format.raw/*43.13*/("""stroke: #fff;
            fill-opacity: .125;
            shape-rendering: crispEdges;
        """),format.raw/*46.9*/("""}"""),format.raw/*46.10*/("""

        """),format.raw/*48.9*/("""</style>

            <!-- jQuery -->
        <script src="/assets/legacy/jquery-2.2.0.min.js"></script>

        <link href='/assets/legacy/roboto.css' rel='stylesheet' type='text/css'>

            <!-- Latest compiled and minified CSS -->
        <link rel="stylesheet" href="/assets/legacy/bootstrap.min.css" />

            <!-- Optional theme -->
        <link rel="stylesheet" href="/assets/legacy/bootstrap-theme.min.css" />

            <!-- Latest compiled and minified JavaScript -->
        <script src="/assets/legacy/bootstrap.min.js" ></script>

            <!-- d3 -->
        <script src="/assets/legacy/d3.v3.min.js" charset="utf-8"></script>

        <script src="/assets/legacy/jquery-fileupload.js"></script>

            <!-- Booststrap Notify plugin-->
        <script src="/assets/legacy/bootstrap-notify.min.js"></script>

            <!-- Latest compiled and minified CSS -->
        <link rel="stylesheet" href="/assets/legacy/bootstrap-select.min.css" />

            <!-- Latest compiled and minified JavaScript -->
        <script src="/assets/legacy/bootstrap-select.min.js"></script>

            <!-- DateTime formatter-->
        <script src="/assets/legacy/DateTimeFormat.js"></script>

        <script src="/assets/legacy/renderWeightsProper.js"></script>

        <script src="/assets/legacy/common.js"></script>

        <style>
        body """),format.raw/*86.14*/("""{"""),format.raw/*86.15*/("""
            """),format.raw/*87.13*/("""font-family: 'Roboto', sans-serif;
            color: #333;
            font-weight: 300;
            font-size: 16px;
        """),format.raw/*91.9*/("""}"""),format.raw/*91.10*/("""

        """),format.raw/*93.9*/(""".hd """),format.raw/*93.13*/("""{"""),format.raw/*93.14*/("""
            """),format.raw/*94.13*/("""background-color: #000000;
            font-size: 18px;
            color: #FFFFFF;
        """),format.raw/*97.9*/("""}"""),format.raw/*97.10*/("""

        """),format.raw/*99.9*/(""".block """),format.raw/*99.16*/("""{"""),format.raw/*99.17*/("""
            """),format.raw/*100.13*/("""width: 250px;
            height: 350px;
            display: inline-block;
            border: 1px solid #DEDEDE;
            margin-right: 64px;
        """),format.raw/*105.9*/("""}"""),format.raw/*105.10*/("""

        """),format.raw/*107.9*/(""".hd-small """),format.raw/*107.19*/("""{"""),format.raw/*107.20*/("""
            """),format.raw/*108.13*/("""background-color: #000000;
            font-size: 14px;
            color: #FFFFFF;
        """),format.raw/*111.9*/("""}"""),format.raw/*111.10*/("""

        """),format.raw/*113.9*/(""".charts, .chart """),format.raw/*113.25*/("""{"""),format.raw/*113.26*/("""
            """),format.raw/*114.13*/("""font-size: 10px;
            font-color: #000000;
            position: relative;
        """),format.raw/*117.9*/("""}"""),format.raw/*117.10*/("""

        """),format.raw/*119.9*/(""".scoreboard """),format.raw/*119.21*/("""{"""),format.raw/*119.22*/("""
            """),format.raw/*120.13*/("""position: absolute;
            top: 20px;
            right: 10px;
            z-index: 1000;
            font-size: 11px;
        """),format.raw/*125.9*/("""}"""),format.raw/*125.10*/("""

        """),format.raw/*127.9*/(""".score """),format.raw/*127.16*/("""{"""),format.raw/*127.17*/("""
            """),format.raw/*128.13*/("""font-size: 11px;
        """),format.raw/*129.9*/("""}"""),format.raw/*129.10*/("""
        """),format.raw/*130.9*/("""</style>

    </head>
    <body>
        <table style="width: 100%;
            padding: 5px;" class="hd">
            <tbody>
                <tr>
                    <td style="width: 48px;"><a href="/"><img src="/assets/legacy/deeplearning4j.img" border="0"/></a></td>
                    <td>DeepLearning4j UI</td>
                    <td style="width: 512px;
                        text-align: right;" class="hd-small">&nbsp; Available sessions: <select class="selectpicker" id="sessionSelector" onchange="window.location.href = 'weights?sid=' + this.options[this.selectedIndex].value ;" style="color: #000000;
                        display: inline-block;
                        width: 256px;">
                        <option value="0" selected="selected">Pick a session to track</option>
                    </select> &nbsp;&nbsp;
                        <script>
            buildSessionSelector("HISTOGRAM");
          </script>
                    </td>
                    <td style="width: 256px;" class="hd-small">&nbsp;Updated at: <b><span id="updatetime">No updates so far</span></b> &nbsp;</td>
                </tr>
            </tbody>
        </table>

        <div style="width: 100%;
            text-align: center;">
            <div id="display" style="width: 1540px;
                height: 900px;
                text-align: left;
                background-color: #FFFFFF;
                display: inline-block;
                overflow: hidden; ">
                <div id="scores" style="background-color: #EFEFEF;
                    display: block;
                    float: left;
                    width: 750px;
                    height: 400px;
                    border: 1px solid #CECECE;
                    margin: 10px;">
                    <h5>&nbsp;&nbsp;Score vs. Iteration #</h5>
                    <div class="chart" id="schart">
        </div>
                </div>
                <div id="model" style="position: relative;
                    background-color: #EFEFEF;
                    display: block;
                    float: left;
                    width: 750px;
                    height: 400px;
                    border: 1px solid #CECECE;
                    margin: 10px;">
                    <h5>&nbsp;&nbsp;Model</h5>
                    <div class="charts"></div>
                    <div style="position: absolute;
                        top: 5px;
                        right: 5px;">
                        <select id="modelSelector" onchange="selectModel();">
          </select>
                    </div>
                </div>
                <div id="gradient" style="position: relative;
                    background-color: #EFEFEF;
                    display: block;
                    float: left;
                    width: 750px;
                    height: 400px;
                    border: 1px solid #CECECE;
                    margin: 10px;">
                    <h5>&nbsp;&nbsp;Gradient</h5>
                    <div class="charts"></div>
                    <div style="position: absolute;
                        top: 5px;
                        right: 5px;">
                        <select id="gradientSelector" onchange="selectGradient();">
          </select>
                    </div>
                </div>
                <div id="magnitudes" style="position: relative;
                    background-color: #EFEFEF;
                    display: block;
                    float: left;
                    width: 750px;
                    height: 400px;
                    border: 1px solid #CECECE;
                    margin: 10px;">
                    <h5>&nbsp;&nbsp;Mean Magnitudes: Parameters and Updates</h5>
                    <div class="charts"></div>
                    <div style="position: absolute;
                        top: 5px;
                        right: 5px;">
                        <select id="magnitudeSelector" onchange="selectMagnitude();">
          </select>
                    </div>
                </div>
                    <!--<div id="lastupdate">
            <div class="updatetime">-1</div>
        </div>-->
            </div>

                <!--
    <div style="display: block;">
        nav bar
    </div> -->
        </div>
            <!--
<div id="score" style="display: inline-block; width: 650px; height: 400px; border: 1px solid #CECECE;">
    <h4>Score</h4>
    <div class="score"></div>
</div>-->

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
object Histogram extends Histogram_Scope0.Histogram
              /*
                  -- GENERATED --
                  DATE: Fri May 18 18:41:46 PDT 2018
                  SOURCE: C:/develop/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/histogram/Histogram.scala.html
                  HASH: 3727d1ee009725c0fa87af96b5bccdf0ee513442
                  MATRIX: 649->0|829->153|857->154|898->168|993->236|1022->237|1061->249|1099->259|1128->260|1170->274|1221->298|1250->299|1289->311|1323->317|1352->318|1394->332|1447->358|1476->359|1515->371|1566->394|1595->395|1637->409|1754->499|1783->500|1822->512|1861->523|1890->524|1932->538|2024->603|2053->604|2092->616|2125->621|2154->622|2196->636|2306->719|2335->720|2374->732|2410->740|2439->741|2481->755|2568->815|2597->816|2636->828|2679->843|2708->844|2750->858|2875->956|2904->957|2943->969|4389->2387|4418->2388|4460->2402|4618->2533|4647->2534|4686->2546|4718->2550|4747->2551|4789->2565|4911->2660|4940->2661|4979->2673|5014->2680|5043->2681|5086->2695|5274->2855|5304->2856|5344->2868|5383->2878|5413->2879|5456->2893|5579->2988|5609->2989|5649->3001|5694->3017|5724->3018|5767->3032|5888->3125|5918->3126|5958->3138|5999->3150|6029->3151|6072->3165|6237->3302|6267->3303|6307->3315|6343->3322|6373->3323|6416->3337|6470->3363|6500->3364|6538->3374
                  LINES: 25->1|31->7|31->7|32->8|34->10|34->10|36->12|36->12|36->12|37->13|38->14|38->14|40->16|40->16|40->16|41->17|42->18|42->18|44->20|44->20|44->20|45->21|48->24|48->24|50->26|50->26|50->26|51->27|53->29|53->29|55->31|55->31|55->31|56->32|59->35|59->35|61->37|61->37|61->37|62->38|64->40|64->40|66->42|66->42|66->42|67->43|70->46|70->46|72->48|110->86|110->86|111->87|115->91|115->91|117->93|117->93|117->93|118->94|121->97|121->97|123->99|123->99|123->99|124->100|129->105|129->105|131->107|131->107|131->107|132->108|135->111|135->111|137->113|137->113|137->113|138->114|141->117|141->117|143->119|143->119|143->119|144->120|149->125|149->125|151->127|151->127|151->127|152->128|153->129|153->129|154->130
                  -- GENERATED --
              */
          