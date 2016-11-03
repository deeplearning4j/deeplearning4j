
package org.deeplearning4j.ui.views.html.training

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object TrainingModel_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class TrainingModel extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template1[org.deeplearning4j.ui.api.I18N,play.twirl.api.HtmlFormat.Appendable] {

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
            <!-- start: Mobile Specific -->
        <meta name="viewport" content="width=device-width, initial-scale=1">
            <!-- end: Mobile Specific -->

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
                    <a class="brand" href="index.html"><span>"""),_display_(/*40.63*/i18n/*40.67*/.getMessage("train.pagetitle")),format.raw/*40.97*/("""</span></a>
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
                            <li><a href="overview"><i class="icon-bar-chart"></i> <span class="hidden-tablet">"""),_display_(/*53.112*/i18n/*53.116*/.getMessage("train.nav.overview")),format.raw/*53.149*/("""</span></a></li>
                            <li class="active"><a href="javascript:void(0);"><i class="icon-tasks"></i> <span class="hidden-tablet">"""),_display_(/*54.134*/i18n/*54.138*/.getMessage("train.nav.model")),format.raw/*54.168*/("""</span></a></li>
                            <li><a href="system"><i class="icon-dashboard"></i> <span class="hidden-tablet">"""),_display_(/*55.110*/i18n/*55.114*/.getMessage("train.nav.system")),format.raw/*55.145*/("""</span></a></li>
                            <li><a href="help"><i class="icon-star"></i> <span class="hidden-tablet">"""),_display_(/*56.103*/i18n/*56.107*/.getMessage("train.nav.userguide")),format.raw/*56.141*/("""</span></a></li>
                            <li>
                                <a class="dropmenu" href="javascript:void(0);"><i class="icon-folder-close-alt"></i> <span class="hidden-tablet"> Language</span></a>
                                <ul>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('en', 'model')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> English</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ja', 'model')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> 日本語</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('zh', 'model')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> 中文</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ko', 'model')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> 한글</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ru', 'model')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> русский</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('uk', 'model')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> український</span></a></li>
                                </ul>
                            </li>
                        </ul>
                    </div>
                </div>
                    <!-- End Main Menu -->

                <noscript>
                    <div class="alert alert-block span10">
                        <h4 class="alert-heading">Warning!</h4>
                        <p>You need to have <a href="http://en.wikipedia.org/wiki/JavaScript" target="_blank">JavaScript</a>
                            enabled to use this site.</p>
                    </div>
                </noscript>

                <style>
                /* Graph */
                #layers """),format.raw/*83.25*/("""{"""),format.raw/*83.26*/("""
                    """),format.raw/*84.21*/("""height: 100%;
                    width: 50%;
                    position: absolute;
                    left: 0;
                    top: 0;
                """),format.raw/*89.17*/("""}"""),format.raw/*89.18*/("""
                """),format.raw/*90.17*/("""</style>

                    <!-- Start Content -->
                <div id="content" class="span10">

                    <div class="row-fluid span6">
                        <div id="layers"></div>
                    </div>
                        <!-- Start Layer Details -->
                    <div class="row-fluid span6" id="0">

                        <div class="box">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*103.41*/i18n/*103.45*/.getMessage("train.model.layerInfoTable.title")),format.raw/*103.92*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <table class="table table-bordered table-striped table-condensed" id="layerInfo"></table>
                            </div>
                        </div>

                        <div class="box">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*112.41*/i18n/*112.45*/.getMessage("train.overview.chart.updateRatioTitle")),format.raw/*112.97*/(""": log<sub>10</sub></b></h2>
                            </div>
                            <div class="box-content">
                                <div id="meanmag" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>log<sub>
                                    10</sub> """),_display_(/*117.47*/i18n/*117.51*/.getMessage("train.overview.chart.updateRatioTitleShort")),format.raw/*117.108*/("""</b> <span id="yMeanMagnitudes">0</span>
                                    , <b>"""),_display_(/*118.43*/i18n/*118.47*/.getMessage("train.overview.charts.iteration")),format.raw/*118.93*/(""":</b> <span id="xMeanMagnitudes">0</span></p>
                            </div>
                        </div>

                        <div class="box">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*124.41*/i18n/*124.45*/.getMessage("train.model.activationsChart.title")),format.raw/*124.94*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <div id="activations" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*128.55*/i18n/*128.59*/.getMessage("train.model.activationsChart.titleShort")),format.raw/*128.113*/(""":</b> <span id="yActivations">0</span>
                                    , <b>"""),_display_(/*129.43*/i18n/*129.47*/.getMessage("train.overview.charts.iteration")),format.raw/*129.93*/(""":</b> <span id="xActivations">0</span></p>
                            </div>
                        </div>

                        <div class="box">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*135.41*/i18n/*135.45*/.getMessage("train.model.lrChart.title")),format.raw/*135.85*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <div id="learningrate" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*139.55*/i18n/*139.59*/.getMessage("train.model.lrChart.titleShort")),format.raw/*139.104*/(""":</b> <span id="yLearningRate">0</span>
                                    , <b>"""),_display_(/*140.43*/i18n/*140.47*/.getMessage("train.overview.charts.iteration")),format.raw/*140.93*/(""":</b> <span id="xLearningRate">0</span></p>
                            </div>
                        </div>

                        <div class="box">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*146.41*/i18n/*146.45*/.getMessage("train.model.paramHistChart.title")),format.raw/*146.92*/("""</b></h2>
                                <div id="paramhistSelected" style="float:left"></div>
                                <div id="paramHistButtonsDiv" style="float:right"></div>
                            </div>
                            <div class="box-content">
                                <div id="parametershistogram" class="center" style="height: 300px;"></div>
                            </div>
                        </div>

                        <div class="box">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*157.41*/i18n/*157.45*/.getMessage("train.model.updateHistChart.title")),format.raw/*157.93*/("""</b></h2>
                                <div id="updatehistSelected" style="float:left"></div>
                                <div id="updateHistButtonsDiv" style="float:right"></div>
                            </div>
                            <div class="box-content">
                                <div id="updateshistogram" class="center" style="height: 300px;"></div>
                            </div>
                        </div>

                    </div>
                        <!-- End Layer Details-->
                </div>
                    <!-- End Content -->
            </div> <!-- End Container -->
        </div> <!-- End Row Fluid-->

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
        <script src="/assets/js/custom.js"></script>
        <script src="/assets/js/cytoscape.min.js"></script>
        <script src="/assets/js/dagre.min.js"></script>
        <script src="/assets/js/cytoscape-dagre.js"></script>
        <script src="/assets/js/train/model.js"></script> <!-- Charts and tables are generated here! -->
        <script src="/assets/js/train/model-graph.js"></script> <!-- Layer graph generated here! -->
        <script src="/assets/js/train/train.js"></script> <!-- Common (lang selection, etc) -->

        <!-- Execute once on page load -->
        <script>
				$(document).ready(function () """),format.raw/*213.35*/("""{"""),format.raw/*213.36*/("""
                    """),format.raw/*214.21*/("""renderModelGraph();
                    renderModelPage();
                """),format.raw/*216.17*/("""}"""),format.raw/*216.18*/(""");
		</script>

            <!-- Execute periodically (every 2 sec) -->
        <script>
				setInterval(function () """),format.raw/*221.29*/("""{"""),format.raw/*221.30*/("""
                    """),format.raw/*222.21*/("""renderModelPage();
                """),format.raw/*223.17*/("""}"""),format.raw/*223.18*/(""", 2000);
		</script>
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
object TrainingModel extends TrainingModel_Scope0.TrainingModel
              /*
                  -- GENERATED --
                  DATE: Thu Nov 03 22:57:22 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingModel.scala.html
                  HASH: 0a8441bb9ae51c738ba23a9b7b890692604fb170
                  MATRIX: 598->1|731->39|759->41|882->138|894->142|944->172|2729->1930|2742->1934|2793->1964|3357->2500|3371->2504|3426->2537|3605->2688|3619->2692|3671->2722|3826->2849|3840->2853|3893->2884|4041->3004|4055->3008|4111->3042|6324->5227|6353->5228|6403->5250|6595->5414|6624->5415|6670->5433|7186->5921|7200->5925|7269->5972|7729->6404|7743->6408|7817->6460|8171->6786|8185->6790|8265->6847|8377->6931|8391->6935|8459->6981|8741->7235|8755->7239|8826->7288|9111->7545|9125->7549|9202->7603|9312->7685|9326->7689|9394->7735|9673->7986|9687->7990|9749->8030|10035->8288|10049->8292|10117->8337|10228->8420|10242->8424|10310->8470|10590->8722|10604->8726|10673->8773|11295->9367|11309->9371|11379->9419|14640->12651|14670->12652|14721->12674|14827->12751|14857->12752|15008->12874|15038->12875|15089->12897|15154->12933|15184->12934
                  LINES: 20->1|25->1|26->2|31->7|31->7|31->7|64->40|64->40|64->40|77->53|77->53|77->53|78->54|78->54|78->54|79->55|79->55|79->55|80->56|80->56|80->56|107->83|107->83|108->84|113->89|113->89|114->90|127->103|127->103|127->103|136->112|136->112|136->112|141->117|141->117|141->117|142->118|142->118|142->118|148->124|148->124|148->124|152->128|152->128|152->128|153->129|153->129|153->129|159->135|159->135|159->135|163->139|163->139|163->139|164->140|164->140|164->140|170->146|170->146|170->146|181->157|181->157|181->157|237->213|237->213|238->214|240->216|240->216|245->221|245->221|246->222|247->223|247->223
                  -- GENERATED --
              */
          