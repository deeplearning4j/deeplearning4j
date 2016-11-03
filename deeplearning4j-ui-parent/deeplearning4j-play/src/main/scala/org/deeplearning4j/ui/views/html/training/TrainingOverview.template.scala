
package org.deeplearning4j.ui.views.html.training

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object TrainingOverview_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class TrainingOverview extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template1[org.deeplearning4j.ui.api.I18N,play.twirl.api.HtmlFormat.Appendable] {

  /**/
  def apply/*1.2*/(i18n: org.deeplearning4j.ui.api.I18N):play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.40*/("""
"""),format.raw/*2.1*/("""<!DOCTYPE html>
<html lang="en">
    <head>

        <meta charset="utf-8">
        <title>"""),_display_(/*7.17*/i18n/*7.21*/.getMessage("train.pagetitle")),format.raw/*7.51*/("""</title>
            <!-- Start Mobile Specific -->
        <meta name="viewport" content="width=device-width, initial-scale=1">
            <!-- End Mobile Specific -->

        <link id="bootstrap-style" href="/assets/css/bootstrap.min.css" rel="stylesheet">
        <link href="/assets/css/bootstrap-responsive.min.css" rel="stylesheet">
        <link id="base-style" href="/assets/css/style.css" rel="stylesheet">
        <link id="base-style-responsive" href="/assets/css/style-responsive.css" rel="stylesheet">
        <link href='http://fonts.googleapis.com/css?family=Open+Sans:300italic,400italic,600italic,700italic,800italic,400,300,600,700,800&subset=latin,cyrillic-ext,latin-ext' rel='stylesheet' type='text/css'>
        <link rel="shortcut icon" href="/assets/img/favicon.ico">

            <!-- The HTML5 shim, for IE6-8 support of HTML5 elements -->
            <!--[if lt IE 9]>
	  	<script src="http://html5shim.googlecode.com/svn/trunk/html5.js"></script>
		<link id="ie-style" href="/assets/css/ie.css" rel="stylesheet"/>
	<![endif]-->

            <!--[if IE 9]>
		<link id="ie9style" href="/assets/css/ie9.css" rel="stylesheet"/>
	<![endif]-->

    </head>

    <body>
            <!-- Start Header -->
        <div class="navbar">
            <div class="navbar-inner">
                <div class="container-fluid">
                    <a class="btn btn-navbar" data-toggle="collapse" data-target=".top-nav.nav-collapse,.sidebar-nav.nav-collapse">
                        <span class="icon-bar"></span>
                        <span class="icon-bar"></span>
                        <span class="icon-bar"></span>
                    </a>
                    <a class="brand" href="#"><span>"""),_display_(/*41.54*/i18n/*41.58*/.getMessage("train.pagetitle")),format.raw/*41.88*/("""</span></a>
                </div>
            </div>
        </div>
            <!-- End Header -->

        <div class="container-fluid-full">
            <div class="row-fluid">

                    <!-- Start Main Menu -->
                <div id="sidebar-left" class="span2">
                    <div class="nav-collapse sidebar-nav">
                        <ul class="nav nav-tabs nav-stacked main-menu">
                            <li class="active"><a href="javascript:void(0);"><i class="icon-bar-chart"></i><span class="hidden-tablet">"""),_display_(/*54.137*/i18n/*54.141*/.getMessage("train.nav.overview")),format.raw/*54.174*/("""</span></a></li>
                            <li><a href="model"><i class="icon-tasks"></i><span class="hidden-tablet">"""),_display_(/*55.104*/i18n/*55.108*/.getMessage("train.nav.model")),format.raw/*55.138*/("""</span></a></li>
                            <li><a href="system"><i class="icon-dashboard"></i><span class="hidden-tablet">"""),_display_(/*56.109*/i18n/*56.113*/.getMessage("train.nav.system")),format.raw/*56.144*/("""</span></a></li>
                            <li><a href="help"><i class="icon-star"></i><span class="hidden-tablet">"""),_display_(/*57.102*/i18n/*57.106*/.getMessage("train.nav.userguide")),format.raw/*57.140*/("""</span></a></li>
                            <li>
                                <a class="dropmenu" href="javascript:void(0);"><i class="icon-folder-close-alt"></i><span class="hidden-tablet">"""),_display_(/*59.146*/i18n/*59.150*/.getMessage("train.nav.language")),format.raw/*59.183*/("""</span></a>
                                <ul>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('en','overview')"><i class="icon-file-alt"></i><span class="hidden-tablet"> English</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ja','overview')"><i class="icon-file-alt"></i><span class="hidden-tablet"> 日本語</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('zh','overview')"><i class="icon-file-alt"></i><span class="hidden-tablet"> 中文</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ko','overview')"><i class="icon-file-alt"></i><span class="hidden-tablet"> 한글</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ru','overview')"><i class="icon-file-alt"></i><span class="hidden-tablet"> русский</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('uk','overview')"><i class="icon-file-alt"></i><span class="hidden-tablet"> український</span></a></li>
                                </ul>
                            </li>
                        </ul>
                    </div>
                </div>
                    <!-- End Main Menu -->

                <noscript>
                    <div class="alert alert-block span10">
                        <h4 class="alert-heading">Warning!</h4>
                        <p>You need to have <a href="http://en.wikipedia.org/wiki/JavaScript" target="_blank">
                            JavaScript</a> enabled to use this site.</p>
                    </div>
                </noscript>

                <!-- Start Score Chart-->
                <div id="content" class="span10">

                    <div class="row-fluid">

                        <div class="box span8">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*89.41*/i18n/*89.45*/.getMessage("train.overview.chart.scoreTitle")),format.raw/*89.91*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <div id="scoreiterchart" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*93.55*/i18n/*93.59*/.getMessage("train.overview.chart.scoreTitleShort")),format.raw/*93.110*/(""":</b> <span id="y">0</span>, <b>"""),_display_(/*93.143*/i18n/*93.147*/.getMessage("train.overview.charts.iteration")),format.raw/*93.193*/(""":</b> <span id="x">
                                    0</span></p>
                            </div>
                        </div>
                        <!-- End Score Chart-->
                        <!-- Start Model Table-->
                        <div class="box span4">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*101.41*/i18n/*101.45*/.getMessage("train.overview.perftable.title")),format.raw/*101.90*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <table class="table table-bordered table-striped table-condensed">
                                    <tr>
                                        <td>"""),_display_(/*106.46*/i18n/*106.50*/.getMessage("train.overview.modeltable.modeltype")),format.raw/*106.100*/("""</td>
                                        <td id="modelType">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*110.46*/i18n/*110.50*/.getMessage("train.overview.modeltable.nLayers")),format.raw/*110.98*/("""</td>
                                        <td id="nLayers">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*114.46*/i18n/*114.50*/.getMessage("train.overview.modeltable.nParams")),format.raw/*114.98*/("""</td>
                                        <td id="nParams">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*118.46*/i18n/*118.50*/.getMessage("train.overview.perftable.startTime")),format.raw/*118.99*/("""</td>
                                        <td id="startTime">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*122.46*/i18n/*122.50*/.getMessage("train.overview.perftable.totalRuntime")),format.raw/*122.102*/("""</td>
                                        <td id="totalRuntime">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*126.46*/i18n/*126.50*/.getMessage("train.overview.perftable.lastUpdate")),format.raw/*126.100*/("""</td>
                                        <td id="lastUpdate">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*130.46*/i18n/*130.50*/.getMessage("train.overview.perftable.totalParamUpdates")),format.raw/*130.107*/("""</td>
                                        <td id="totalParamUpdates">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*134.46*/i18n/*134.50*/.getMessage("train.overview.perftable.updatesPerSec")),format.raw/*134.103*/("""</td>
                                        <td id="updatesPerSec">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*138.46*/i18n/*138.50*/.getMessage("train.overview.perftable.examplesPerSec")),format.raw/*138.104*/("""</td>
                                        <td id="examplesPerSec">Loading...</td>
                                    </tr>
                                </table>
                            </div>
                        </div>
                        <!--End Model Table -->
                    </div>


                    <div class="row-fluid">
                        <!--Start Ratio Table -->
                        <div class="box span6">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*152.41*/i18n/*152.45*/.getMessage("train.overview.chart.updateRatioTitle")),format.raw/*152.97*/(""": log<sub>10</sub></b></h2>
                            </div>
                            <div class="box-content">
                                <div id="updateRatioChart"  class="center" style="height:300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*156.55*/i18n/*156.59*/.getMessage("train.overview.chart.updateRatioTitleShort")),format.raw/*156.116*/(""":</b> <span id="yRatio">0</span>, <b>log<sub>10</sub> """),_display_(/*156.171*/i18n/*156.175*/.getMessage("train.overview.chart.updateRatioTitleShort")),format.raw/*156.232*/(""":</b> <span id="yLogRatio">0</span>, <b>"""),_display_(/*156.273*/i18n/*156.277*/.getMessage("train.overview.charts.iteration")),format.raw/*156.323*/(""":</b> <span id="xRatio">0</span></p>
                            </div>
                        </div>
                        <!--End Ratio Table -->
                        <!--Start Variance Table -->
                        <div class="box span6">
                            <div class="box-header">
                                <h2><b><h2><b>"""),_display_(/*163.48*/i18n/*163.52*/.getMessage("train.overview.chart.stdevTitle")),format.raw/*163.98*/(""": log<sub>10</sub></b></h2></b></h2>
                                <div style="float: right">
                                    <p class="stackControls center">
                                        <input class="btn btn-small" type="button" value=""""),_display_(/*166.92*/i18n/*166.96*/.getMessage("train.overview.chart.stdevBtn.activations")),format.raw/*166.152*/("""" onclick="selectStdevChart('stdevActivations')">
                                        <input class="btn btn-small" type="button" value=""""),_display_(/*167.92*/i18n/*167.96*/.getMessage("train.overview.chart.stdevBtn.gradients")),format.raw/*167.150*/("""" onclick="selectStdevChart('stdevGradients')">
                                        <input class="btn btn-small" type="button" value=""""),_display_(/*168.92*/i18n/*168.96*/.getMessage("train.overview.chart.stdevBtn.updates")),format.raw/*168.148*/("""" onclick="selectStdevChart('stdevUpdates')">
                                    </p>
                                </div>
                            </div>
                            <div class="box-content">
                                <div id="stdevChart"  class="center" style="height:300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*174.55*/i18n/*174.59*/.getMessage("train.overview.chart.stdevTitleShort")),format.raw/*174.110*/(""":</b> <span id="yStdev">0</span>, <b>log<sub>10</sub> """),_display_(/*174.165*/i18n/*174.169*/.getMessage("train.overview.chart.stdevTitleShort")),format.raw/*174.220*/(""":</b> <span id="yLogStdev">0</span>, <b>"""),_display_(/*174.261*/i18n/*174.265*/.getMessage("train.overview.charts.iteration")),format.raw/*174.311*/(""":</b> <span id="xStdev">0</span></p>
                            </div>
                        </div>
                        <!-- End Variance Table -->
                    </div>

                </div><!-- End Content Span10-->
            </div><!--End Row Fluid-->

            <!-- Start JavaScript-->
            <script src="/assets/js/jquery-1.9.1.min.js"></script>
            <script src="/assets/js/jquery-migrate-1.0.0.min.js"></script>
            <script src="/assets/js/jquery-ui-1.10.0.custom.min.js"></script>
            <script src="/assets/js/jquery.ui.touch-punch.js"></script>
            <script src="/assets/js/modernizr.js"></script>
            <script src="/assets/js/bootstrap.min.js"></script>
            <script src="/assets/js/jquery.cookie.js"></script>
            <script src="/assets/js/fullcalendar.min.js"></script>
            <script src="/assets/js/jquery.dataTables.min.js"></script>
            <script src="/assets/js/excanvas.js"></script>
            <script src="/assets/js/jquery.flot.js"></script>
            <script src="/assets/js/jquery.flot.pie.js"></script>
            <script src="/assets/js/jquery.flot.stack.js"></script>
            <script src="/assets/js/jquery.flot.resize.min.js"></script>
            <script src="/assets/js/jquery.chosen.min.js"></script>
            <script src="/assets/js/jquery.uniform.min.js"></script>
            <script src="/assets/js/jquery.cleditor.min.js"></script>
            <script src="/assets/js/jquery.noty.js"></script>
            <script src="/assets/js/jquery.elfinder.min.js"></script>
            <script src="/assets/js/jquery.raty.min.js"></script>
            <script src="/assets/js/jquery.iphone.toggle.js"></script>
            <script src="/assets/js/jquery.uploadify-3.1.min.js"></script>
            <script src="/assets/js/jquery.gritter.min.js"></script>
            <script src="/assets/js/jquery.imagesloaded.js"></script>
            <script src="/assets/js/jquery.masonry.min.js"></script>
            <script src="/assets/js/jquery.knob.modified.js"></script>
            <script src="/assets/js/jquery.sparkline.min.js"></script>
            <script src="/assets/js/counter.js"></script>
            <script src="/assets/js/retina.js"></script>
            <script src="/assets/js/train/overview.js"></script>    <!-- Charts and tables are generated here! -->
            <script src="/assets/js/train/train.js"></script>   <!-- Common (lang selection, etc) -->

            <!-- Execute once on page load -->
            <script>
                    $(document).ready(function()"""),format.raw/*218.49*/("""{"""),format.raw/*218.50*/("""
                        """),format.raw/*219.25*/("""renderOverviewPage();
                    """),format.raw/*220.21*/("""}"""),format.raw/*220.22*/(""");
            </script>

            <!-- Execute periodically (every 2 sec) -->
            <script>
                    setInterval(function()"""),format.raw/*225.43*/("""{"""),format.raw/*225.44*/("""
                        """),format.raw/*226.25*/("""renderOverviewPage();
                    """),format.raw/*227.21*/("""}"""),format.raw/*227.22*/(""", 2000);
            </script>
            <!-- End JavaScript-->

    </body>
</html>
"""))
      }
    }
  }

  def render(i18n:org.deeplearning4j.ui.api.I18N): play.twirl.api.HtmlFormat.Appendable = apply(i18n)

  def f:((org.deeplearning4j.ui.api.I18N) => play.twirl.api.HtmlFormat.Appendable) = (i18n) => apply(i18n)

  def ref: this.type = this

}


}

/**/
object TrainingOverview extends TrainingOverview_Scope0.TrainingOverview
              /*
                  -- GENERATED --
                  DATE: Thu Nov 03 19:04:01 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingOverview.scala.html
                  HASH: e8acc83294a535d7a0848b408c594b88be637638
                  MATRIX: 604->1|737->39|765->41|888->138|900->142|950->172|2726->1921|2739->1925|2790->1955|3379->2516|3393->2520|3448->2553|3597->2674|3611->2678|3663->2708|3817->2834|3831->2838|3884->2869|4031->2988|4045->2992|4101->3026|4326->3223|4340->3227|4395->3260|6616->5454|6629->5458|6696->5504|6983->5764|6996->5768|7069->5819|7130->5852|7144->5856|7212->5902|7622->6284|7636->6288|7703->6333|8020->6622|8034->6626|8107->6676|8348->6889|8362->6893|8432->6941|8671->7152|8685->7156|8755->7204|8994->7415|9008->7419|9079->7468|9320->7681|9334->7685|9409->7737|9653->7953|9667->7957|9740->8007|9982->8221|9996->8225|10076->8282|10325->8503|10339->8507|10415->8560|10660->8777|10674->8781|10751->8835|11340->9396|11354->9400|11428->9452|11736->9732|11750->9736|11830->9793|11914->9848|11929->9852|12009->9909|12079->9950|12094->9954|12163->10000|12550->10359|12564->10363|12632->10409|12919->10668|12933->10672|13012->10728|13182->10870|13196->10874|13273->10928|13441->11068|13455->11072|13530->11124|13932->11498|13946->11502|14020->11553|14104->11608|14119->11612|14193->11663|14263->11704|14278->11708|14347->11754|17026->14404|17056->14405|17111->14431|17183->14474|17213->14475|17392->14625|17422->14626|17477->14652|17549->14695|17579->14696
                  LINES: 20->1|25->1|26->2|31->7|31->7|31->7|65->41|65->41|65->41|78->54|78->54|78->54|79->55|79->55|79->55|80->56|80->56|80->56|81->57|81->57|81->57|83->59|83->59|83->59|113->89|113->89|113->89|117->93|117->93|117->93|117->93|117->93|117->93|125->101|125->101|125->101|130->106|130->106|130->106|134->110|134->110|134->110|138->114|138->114|138->114|142->118|142->118|142->118|146->122|146->122|146->122|150->126|150->126|150->126|154->130|154->130|154->130|158->134|158->134|158->134|162->138|162->138|162->138|176->152|176->152|176->152|180->156|180->156|180->156|180->156|180->156|180->156|180->156|180->156|180->156|187->163|187->163|187->163|190->166|190->166|190->166|191->167|191->167|191->167|192->168|192->168|192->168|198->174|198->174|198->174|198->174|198->174|198->174|198->174|198->174|198->174|242->218|242->218|243->219|244->220|244->220|249->225|249->225|250->226|251->227|251->227
                  -- GENERATED --
              */
          