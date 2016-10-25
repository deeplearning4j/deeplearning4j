
package org.deeplearning4j.ui.views.html.tsne

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object Tsne_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class Tsne extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template0[play.twirl.api.HtmlFormat.Appendable] {

  /**/
  def apply():play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.1*/("""<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="utf-8" />
        <title>T-SNE renders</title>

            <!-- jQuery -->
            <!--
    <script src="//ajax.googleapis.com/ajax/libs/jquery/1/jquery.min.js"></script>
    <script src="/assets/d3.min.js"></script>
    <script src="/assets/render.js"></script>
    <link href='http://fonts.googleapis.com/css?family=Roboto:400,300' rel='stylesheet' type='text/css'>
    <link rel="stylesheet" href="/assets/bootstrap-3.3.4-dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="/assets/bootstrap-3.3.4-dist/css/bootstrap-theme.min.css">
    <link rel="stylesheet" href="/assets/css/simple-sidebar.css">
    <link rel="stylesheet" href="/assets/css/style.css">
    <script src="https://code.jquery.com/jquery-2.1.3.min.js"></script>
    <script src="/assets/jquery-fileupload.js"></script>
    <script src="/assets/bootstrap-3.3.4-dist/js/bootstrap.min.js"></script>
    -->
            <!-- jQuery -->
        <script src="/assets/jquery-2.2.0.min.js"></script>

        <link href='/assets/roboto.css' rel='stylesheet' type='text/css'>

            <!-- Latest compiled and minified CSS -->
        <link rel="stylesheet" href="/assets/bootstrap.min.css" />

            <!-- Optional theme -->
        <link rel="stylesheet" href="/assets/bootstrap-theme.min.css" />

            <!-- Latest compiled and minified JavaScript -->
        <script src="/assets/bootstrap.min.js" ></script>


            <!-- d3 -->
        <script src="/assets/d3.v3.min.js" charset="utf-8"></script>


        <script src="/assets/jquery-fileupload.js"></script>

            <!-- Booststrap Notify plugin-->
        <script src="/assets/bootstrap-notify.min.js"></script>

        <script src="/assets/common.js"></script>

            <!-- dl4j plot setup -->
        <script src="/assets/renderTsne.js"></script>


        <style>
        .hd """),format.raw/*52.13*/("""{"""),format.raw/*52.14*/("""
            """),format.raw/*53.13*/("""background-color: #000000;
            font-size: 18px;
            color: #FFFFFF;
        """),format.raw/*56.9*/("""}"""),format.raw/*56.10*/("""
        """),format.raw/*57.9*/(""".block """),format.raw/*57.16*/("""{"""),format.raw/*57.17*/("""
            """),format.raw/*58.13*/("""width: 250px;
            height: 350px;
            display: inline-block;
            border: 1px solid #DEDEDE;
            margin-right: 64px;
        """),format.raw/*63.9*/("""}"""),format.raw/*63.10*/("""
        """),format.raw/*64.9*/(""".hd-small """),format.raw/*64.19*/("""{"""),format.raw/*64.20*/("""
            """),format.raw/*65.13*/("""background-color: #000000;
            font-size: 14px;
            color: #FFFFFF;
        """),format.raw/*68.9*/("""}"""),format.raw/*68.10*/("""

        """),format.raw/*70.9*/("""body """),format.raw/*70.14*/("""{"""),format.raw/*70.15*/("""
            """),format.raw/*71.13*/("""font-family: 'Roboto', sans-serif;
            color: #333;
            font-weight: 300;
            font-size: 16px;
        """),format.raw/*75.9*/("""}"""),format.raw/*75.10*/("""
        """),format.raw/*76.9*/("""#wrap """),format.raw/*76.15*/("""{"""),format.raw/*76.16*/("""
            """),format.raw/*77.13*/("""width: 800px;
            margin-left: auto;
            margin-right: auto;
        """),format.raw/*80.9*/("""}"""),format.raw/*80.10*/("""
        """),format.raw/*81.9*/("""#embed """),format.raw/*81.16*/("""{"""),format.raw/*81.17*/("""
            """),format.raw/*82.13*/("""margin-top: 10px;
        """),format.raw/*83.9*/("""}"""),format.raw/*83.10*/("""
        """),format.raw/*84.9*/("""h1 """),format.raw/*84.12*/("""{"""),format.raw/*84.13*/("""
            """),format.raw/*85.13*/("""text-align: center;
            font-weight: normal;
        """),format.raw/*87.9*/("""}"""),format.raw/*87.10*/("""
        """),format.raw/*88.9*/(""".tt """),format.raw/*88.13*/("""{"""),format.raw/*88.14*/("""
            """),format.raw/*89.13*/("""margin-top: 10px;
            background-color: #EEE;
            border-bottom: 1px solid #333;
            padding: 5px;
        """),format.raw/*93.9*/("""}"""),format.raw/*93.10*/("""
        """),format.raw/*94.9*/(""".txth """),format.raw/*94.15*/("""{"""),format.raw/*94.16*/("""
            """),format.raw/*95.13*/("""color: #F55;
        """),format.raw/*96.9*/("""}"""),format.raw/*96.10*/("""
        """),format.raw/*97.9*/(""".cit """),format.raw/*97.14*/("""{"""),format.raw/*97.15*/("""
            """),format.raw/*98.13*/("""font-family: courier;
            padding-left: 20px;
            font-size: 14px;
        """),format.raw/*101.9*/("""}"""),format.raw/*101.10*/("""
        """),format.raw/*102.9*/(""".axis """),format.raw/*102.15*/("""{"""),format.raw/*102.16*/("""

        """),format.raw/*104.9*/("""}"""),format.raw/*104.10*/("""
        """),format.raw/*105.9*/(""".axis path,
        .axis line """),format.raw/*106.20*/("""{"""),format.raw/*106.21*/("""
            """),format.raw/*107.13*/("""fill: none;
            stroke: rgba(0,0,0,0.1);
            shape-rendering: crispEdges;
        """),format.raw/*110.9*/("""}"""),format.raw/*110.10*/("""
        """),format.raw/*111.9*/(""".axis text """),format.raw/*111.20*/("""{"""),format.raw/*111.21*/("""
            """),format.raw/*112.13*/("""font-family: sans-serif;
            font-size: 11px;
            fill: #666;
        """),format.raw/*115.9*/("""}"""),format.raw/*115.10*/("""
        """),format.raw/*116.9*/(""".label """),format.raw/*116.16*/("""{"""),format.raw/*116.17*/("""
            """),format.raw/*117.13*/("""font-size:14px;
            fill:rgba(0,0,0,0.5);
            shape-rendering:auto;
        """),format.raw/*120.9*/("""}"""),format.raw/*120.10*/("""
        """),format.raw/*121.9*/("""</style>

        <script>
        $(document).ready(function() """),format.raw/*124.38*/("""{"""),format.raw/*124.39*/("""
            """),format.raw/*125.13*/("""$('#filenamebutton').click(function() """),format.raw/*125.51*/("""{"""),format.raw/*125.52*/("""
                """),format.raw/*126.17*/("""document.getElementById('form').reset();
                $('#form').hide();
                var filename = $('#filename').val();
                $('#filename').val('');
                updateFileName(filename);
                drawTsne();
            """),format.raw/*132.13*/("""}"""),format.raw/*132.14*/(""");

            $('#form').fileUpload("""),format.raw/*134.35*/("""{"""),format.raw/*134.36*/("""success : function(data, textStatus, jqXHR)"""),format.raw/*134.79*/("""{"""),format.raw/*134.80*/("""
                """),format.raw/*135.17*/("""var fullPath = document.getElementById('form').value;
                var filename = data['name'];
                if (fullPath) """),format.raw/*137.31*/("""{"""),format.raw/*137.32*/("""
                    """),format.raw/*138.21*/("""var startIndex = (fullPath.indexOf('\\') >= 0 ? fullPath.lastIndexOf('\\') : fullPath.lastIndexOf('/'));
                    var filename = fullPath.substring(startIndex);
                    if (filename.indexOf('\\') === 0 || filename.indexOf('/') === 0) """),format.raw/*140.86*/("""{"""),format.raw/*140.87*/("""
                        """),format.raw/*141.25*/("""filename = filename.substring(1);
                    """),format.raw/*142.21*/("""}"""),format.raw/*142.22*/("""
                """),format.raw/*143.17*/("""}"""),format.raw/*143.18*/("""

                """),format.raw/*145.17*/("""document.getElementById('form').reset();
                //$('#form').hide();

                updateFileName(filename);
                drawTsne();

            """),format.raw/*151.13*/("""}"""),format.raw/*151.14*/(""",error : function(err) """),format.raw/*151.37*/("""{"""),format.raw/*151.38*/("""
                """),format.raw/*152.17*/("""console.log(err);
            """),format.raw/*153.13*/("""}"""),format.raw/*153.14*/("""}"""),format.raw/*153.15*/(""");


            function updateFileName(name) """),format.raw/*156.43*/("""{"""),format.raw/*156.44*/("""
                """),format.raw/*157.17*/("""/*
                 $.ajax("""),format.raw/*158.25*/("""{"""),format.raw/*158.26*/("""
                 """),format.raw/*159.18*/("""url: './api/update',
                 type: 'POST',
                 dataType: 'json',
                 data: JSON.stringify("""),format.raw/*162.39*/("""{"""),format.raw/*162.40*/(""""url" : name"""),format.raw/*162.52*/("""}"""),format.raw/*162.53*/("""),
                 cache: false,
                 success: function(data, textStatus, jqXHR) """),format.raw/*164.61*/("""{"""),format.raw/*164.62*/("""


                 """),format.raw/*167.18*/("""}"""),format.raw/*167.19*/(""",
                 error: function(jqXHR, textStatus, errorThrown) """),format.raw/*168.66*/("""{"""),format.raw/*168.67*/("""
                 """),format.raw/*169.18*/("""// Handle errors here
                 console.log('ERRORS: ' + textStatus);
                 """),format.raw/*171.18*/("""}"""),format.raw/*171.19*/(""",
                 complete: function() """),format.raw/*172.39*/("""{"""),format.raw/*172.40*/("""
                 """),format.raw/*173.18*/("""}"""),format.raw/*173.19*/("""
                 """),format.raw/*174.18*/("""}"""),format.raw/*174.19*/(""");
                 */
            """),format.raw/*176.13*/("""}"""),format.raw/*176.14*/("""


            """),format.raw/*179.13*/("""drawTsne();

        """),format.raw/*181.9*/("""}"""),format.raw/*181.10*/(""") ;

    </script>

    </head>

    <body>
        <table style="width: 100%; padding: 5px;" class="hd">
            <tbody>
                <tr>
                    <td style="width: 48px;"><a href="/"><img src="/assets/deeplearning4j.img" border="0"/></a></td>
                    <td>DeepLearning4j UI</td>
                    <td style="width: 512px; text-align: right;" class="hd-small">&nbsp; Available sessions: <select class="selectpicker" id="sessionSelector" onchange="window.location.href = 'tsne?sid='+ this.options[this.selectedIndex].value ;" style="color: #000000; display: inline-block; width: 256px;">
                        <option value="0" selected="selected">Pick a session to track</option>
                    </select>&nbsp;&nbsp;
                        <script>
                            buildSessionSelector("TSNE");
                        </script>
                    </td>
                    <td style="width: 256px;">&nbsp; <!-- placeholder for future use --></td>
                </tr>
            </tbody>
        </table>

        <br />
        <div style="text-align: center">
            <div id="embed" style="display: inline-block; width: 1024px; height: 700px; border: 1px solid #DEDEDE;"></div>
        </div>
        <br/>
        <br/>
        <div style="text-align:center; width: 100%; position: fixed; bottom: 0px; left: 0px; margin-bottom: 15px;">
            <div style="display: inline-block; margin-right: 48px;">
                <h5>Upload a file to UI server.</h5>
                <form encType="multipart/form-data" action="/api/upload" method="POST" id="form">
                    <div>

                        <input name="file" type="file" style="width:300px; display: inline-block;" /><input type="submit" value="Upload file" style="display: inline-block;"/>

                    </div>
                </form>
            </div>

            <div style="display: inline-block;">
                <h5>If a file is already present on the server, specify the path/name.</h5>
                <div id="filebutton">
                    <input type="text" id="filename"/>
                    <button id="filenamebutton">Submit</button>
                </div>
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
object Tsne extends Tsne_Scope0.Tsne
              /*
                  -- GENERATED --
                  DATE: Tue Oct 25 20:32:36 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/tsne/Tsne.scala.html
                  HASH: 150b088d50e1d3f6602d743de700905e2426c102
                  MATRIX: 634->0|2624->1962|2653->1963|2695->1977|2817->2072|2846->2073|2883->2083|2918->2090|2947->2091|2989->2105|3176->2265|3205->2266|3242->2276|3280->2286|3309->2287|3351->2301|3473->2396|3502->2397|3541->2409|3574->2414|3603->2415|3645->2429|3803->2560|3832->2561|3869->2571|3903->2577|3932->2578|3974->2592|4089->2680|4118->2681|4155->2691|4190->2698|4219->2699|4261->2713|4315->2740|4344->2741|4381->2751|4412->2754|4441->2755|4483->2769|4573->2832|4602->2833|4639->2843|4671->2847|4700->2848|4742->2862|4904->2997|4933->2998|4970->3008|5004->3014|5033->3015|5075->3029|5124->3051|5153->3052|5190->3062|5223->3067|5252->3068|5294->3082|5416->3176|5446->3177|5484->3187|5519->3193|5549->3194|5589->3206|5619->3207|5657->3217|5718->3249|5748->3250|5791->3264|5920->3365|5950->3366|5988->3376|6028->3387|6058->3388|6101->3402|6218->3491|6248->3492|6286->3502|6322->3509|6352->3510|6395->3524|6518->3619|6548->3620|6586->3630|6682->3697|6712->3698|6755->3712|6822->3750|6852->3751|6899->3769|7185->4026|7215->4027|7284->4067|7314->4068|7386->4111|7416->4112|7463->4130|7623->4261|7653->4262|7704->4284|7992->4543|8022->4544|8077->4570|8161->4625|8191->4626|8238->4644|8268->4645|8317->4665|8514->4833|8544->4834|8596->4857|8626->4858|8673->4876|8733->4907|8763->4908|8793->4909|8872->4959|8902->4960|8949->4978|9006->5006|9036->5007|9084->5026|9241->5154|9271->5155|9312->5167|9342->5168|9467->5264|9497->5265|9549->5288|9579->5289|9676->5357|9706->5358|9754->5377|9879->5473|9909->5474|9979->5515|10009->5516|10057->5535|10087->5536|10135->5555|10165->5556|10231->5593|10261->5594|10308->5612|10359->5635|10389->5636
                  LINES: 25->1|76->52|76->52|77->53|80->56|80->56|81->57|81->57|81->57|82->58|87->63|87->63|88->64|88->64|88->64|89->65|92->68|92->68|94->70|94->70|94->70|95->71|99->75|99->75|100->76|100->76|100->76|101->77|104->80|104->80|105->81|105->81|105->81|106->82|107->83|107->83|108->84|108->84|108->84|109->85|111->87|111->87|112->88|112->88|112->88|113->89|117->93|117->93|118->94|118->94|118->94|119->95|120->96|120->96|121->97|121->97|121->97|122->98|125->101|125->101|126->102|126->102|126->102|128->104|128->104|129->105|130->106|130->106|131->107|134->110|134->110|135->111|135->111|135->111|136->112|139->115|139->115|140->116|140->116|140->116|141->117|144->120|144->120|145->121|148->124|148->124|149->125|149->125|149->125|150->126|156->132|156->132|158->134|158->134|158->134|158->134|159->135|161->137|161->137|162->138|164->140|164->140|165->141|166->142|166->142|167->143|167->143|169->145|175->151|175->151|175->151|175->151|176->152|177->153|177->153|177->153|180->156|180->156|181->157|182->158|182->158|183->159|186->162|186->162|186->162|186->162|188->164|188->164|191->167|191->167|192->168|192->168|193->169|195->171|195->171|196->172|196->172|197->173|197->173|198->174|198->174|200->176|200->176|203->179|205->181|205->181
                  -- GENERATED --
              */
          