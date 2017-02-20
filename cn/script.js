jQuery(function($) {
  var $navbar = $('#navbar').closest('.navbar');

  var $searchDesktop = $navbar.find('.search-desktop');
  var $ulDesktop = $searchDesktop.find('ul');
  var $inputDesktop = $searchDesktop.find('input');
  $inputDesktop.on('click', show);

  var $searchMobile = $navbar.find('.search-mobile');
  var $ulMobile = $searchMobile.find('ul');
  var $inputMobile = $searchMobile.find('input');

  $navbar.find('.navbar-header .glyphicon-search').click(function(e) {
    e.preventDefault();
    show();
  });

  $navbar.find('.glyphicon-remove').on('click', hide);

  $inputDesktop.on('input', input);
  $inputMobile.on('input', input);

  var isShowing = false;
  function show() {
    if (isShowing) {
      return;
    }
    $navbar.addClass('search-visible');
    $searchDesktop.siblings().hide();
    $('.navbar-header .navbar-brand').hide();
    $searchMobile.show();
    $inputDesktop.trigger('input');
    isShowing = true;
  }

  function hide() {
    if (!isShowing) {
      return;
    }
    $navbar.removeClass('search-visible');
    setTimeout(function() {
      $searchDesktop.siblings().show();
    }, 200);
    $('.navbar-header .navbar-brand').show();
    $searchMobile.hide();
    isShowing = false;
  }

  function input() {
    var query = $(this).val();
    $inputDesktop.val(query);
    $inputMobile.val(query);
    var filtered = filterPages(query);
    var $results = $('<ul>');
    filtered.slice(0, 10).forEach(function(page) {
      var $a = $('<a>').attr('href', page.url).text(page.title);
      $results.append($('<li>').append($a));
    })
    if (filtered.length > 10) {
      var $a = $('<a>').attr('href', '/cn/search?q=' + encodeURIComponent(query)).text('See all ' + filtered.length + ' results for "' + query + '"');
      $('<li>').attr('class', 'see-more').append($a)
        .appendTo($results);
    }
    $ulDesktop.empty().append($results.clone().children());
    $ulMobile.empty().append($results.clone().children());
    if (filtered.length === 0) {
      $ulDesktop.hide();
      $ulMobile.hide();
    } else {
      $ulDesktop.show();
      $ulMobile.show();
    }
  }

  function filterPages(query) {
    if (query.trim() === '') {
      return [];
    }

    var regexes = query.trim().split(/\s+/).map(function(q) {
      return new RegExp(q, 'i');
    });
    return PAGES.map(function(page) {
      var matches = countMatches(regexes, page.title) * 2 +
                    countMatches(regexes, page.description) +
                    countMatches(regexes, page.tags.join(' '));
      return { page: page, matches: matches };
    }).filter(function(d) {
      return d.matches > 0;
    }).sort(function(d1, d2) {
      return d2.matches - d1.matches;
    }).map(function(d) {
      return d.page;
    });
  }

  var $searchPage = $('.search-page');
  var $searchPageInput = $searchPage.find('input');
  var $searchPageResults = $searchPage.find('ul');

  $searchPageInput.on('input', function() {
    $searchPageResults.empty();
    filterPages($(this).val()).forEach(function(page) {
      var $a = $('<a>').attr('href', page.url).text(page.title);
      var $h6 = $('<h6>').append($a.clone());
      var $div = $('<div>')
        .append($('<p>').text(page.description))
        .append($('<p>').text(page.tags.join(', ')));
      var $li = $('<li>').append($h6).append($div);
      $searchPageResults.append($li);
    });
  });

  // http://stackoverflow.com/a/3855394
  var qs = (function(a) {
    if (a == "") return {};
    var b = {};
    for (var i = 0; i < a.length; ++i)
    {
      var p=a[i].split('=', 2);
      if (p.length == 1) b[p[0]] = "";
      else b[p[0]] = decodeURIComponent(p[1].replace(/\+/g, " "));
    }
    return b;
  })(window.location.search.substr(1).split('&'));

  if (qs.q) {
    $searchPageInput.val(qs.q).trigger('input');
  }

  function countMatches(regexes, string) {
    var count = 0, from = 0;
    for(var i = 0; i < regexes.length; i++) {
      var regex = regexes[i];
      var match = string.slice(from).match(regex);
      if (match) {
        count++;
        from += match[0].length + match.index;
      }
    }
    return count;
  }
});
