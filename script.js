function scrollResizeEvents() {
    window.addEventListener("scroll", function() {
        const right = document.getElementById("right");
        const left = document.getElementById("left");
        var offset = window.pageYOffset;    // Max 4959.33349609375
        //console.log(offset)
        var posX = (offset / 4959.33349609375) * 3;
        //var posY = 100 - (offset*0.05);
        if(posX <= 3) {
            right.style.backgroundPositionX = 103 - posX + "%";
            left.style.backgroundPositionX = -3 + posX + "%";
            //console.log(posX)
        }
        //right.style.backgroundPositionY = posY + "px";
        //left.style.backgroundPositionY = posY + "px";
        //console.log(window.screen.width);
    });
    
    window.addEventListener("resize", function() {
        width = this.innerWidth;
        //console.log(width);
        if (width < 1400) {
            document.getElementById("right").style.display = "none";
            document.getElementById("left").style.display = "none";
            document.getElementById("content").style.width = "80vw";
        } else {
            document.getElementById("right").style.display = "";
            document.getElementById("left").style.display = "";
            document.getElementById("content").style.width = "50vw";
        }
    });
}

function setPosOnload() {
    const right = document.getElementById("right");
    const left = document.getElementById("left");
    right.style.backgroundPositionX = "103%";
    left.style.backgroundPositionX = "-3%";
    right.style.backgroundPositionY = "80px";
    left.style.backgroundPositionY = "80px";
    width = this.innerWidth;
    //console.log(width);
    if (width < 1400) {
        document.getElementById("right").style.display = "none";
        document.getElementById("left").style.display = "none";
        document.getElementById("content").style.width = "80vw";
    } else {
        document.getElementById("right").style.display = "";
        document.getElementById("left").style.display = "";
        document.getElementById("content").style.width = "50vw";
    }
}

function populateCarousel(year) {
    $.getJSON("./datasets/curated.json", function(json){
        let patent = json[year];
        let matches_cos = patent['matches_cos'];
        let matches_mnh = patent['matches_mnh'];
        let matches_euc = patent['matches_euc'];
        let matches_list = [matches_cos, matches_mnh, matches_euc];

        const ref = {
            0: "cos",
            1: "mnh",
            2: "euc"
        }
        const titles = {
            0: "Cosine Similarity Match",
            1: "Manhattan Distance Match",
            2: "Euclidean Distance Match"
        }

        for (matches in matches_list) {
            for (let i=0; i<5; i++) {
                var patent_match = matches_list[matches][i];
                div_id = i + ref[matches];
                var parent = document.getElementById(div_id);
                var ttid = 'id=\"' + div_id + 'tt\"';
                parent.innerHTML = '<div ' + ttid  + ' class="matches_tooltip">Copied embedding</div>';
                var fields = Object.keys(patent_match);
                var title = document.createElement("div");
                title.innerHTML = titles[matches] + '<br><br>'
                parent.appendChild(title);
                for (let j=0; j<fields.length; j++) {
                    var child = document.createElement("div");
                    child.id = div_id + fields[j];
                    var html = fields[j] + ':<br>' + patent_match[fields[j]] + '<br><br>';
                    child.innerHTML = html;
                    parent.appendChild(child);
                }
                $(parent).prepend(
                    `<div class="matches_click"></div>
                    <svg class="clipboard_icon" xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="#b6b6b6" class="bi bi-files" viewBox="0 0 16 16">
                      <path d="M13 0H6a2 2 0 0 0-2 2 2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h7a2 2 0 0 0 2-2 2 2 0 0 0 2-2V2a2 2 0 0 0-2-2zm0 13V4a2 2 0 0 0-2-2H5a1 1 0 0 1 1-1h7a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1zM3 4a1 1 0 0 1 1-1h7a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V4z"/>
                    </svg>
                    <svg class="checkmark_icon" xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="#b6b6b6" class="bi bi-check2" viewBox="0 0 16 16">
                      <path d="M13.854 3.646a.5.5 0 0 1 0 .708l-7 7a.5.5 0 0 1-.708 0l-3.5-3.5a.5.5 0 1 1 .708-.708L6.5 10.293l6.646-6.647a.5.5 0 0 1 .708 0z"/>
                    </svg>`
                );
                var click_id = div_id + '_click';
                //const elm_parent = $(parent).find($('.matches_click')).parent();
                $(parent).find($('.matches_click')).attr('id', click_id)
                document.getElementById(click_id).addEventListener('click', function(e) {
                    copyPasta(e.target.id);
                });

                //Change background color
                switch(parseInt(matches)) {
                    case 0:
                        parent.style.backgroundColor = "#fffff5";
                        break;
                    case 1:
                        parent.style.backgroundColor = "#ffffeb";
                        break;
                    case 2:
                        parent.style.backgroundColor = "#ffffe3";
                        break;
                    default:
                        parent.style.backgroundColor = "#f5f5ff";
                }
            }
        }
    });
}

function initSamples() {
    $.getJSON("./datasets/curated.json", function(json){
        let years = Object.keys(json)
        for (let i=0; i<years.length; i++) {
            var parent = document.getElementById(years[i]);
            var ttid = 'id=\"' + years[i] + 'tt\"';
            parent.innerHTML = '<div ' + ttid + ' class="samples_tooltip">Copied embedding</div>'
            var patent = json[years[i]];
            var fields = Object.keys(patent);
            var title = document.createElement("div");
            title.innerHTML = years[i] + 's <br><br>'
            parent.appendChild(title);
            for (let j=0; j<fields.length-3; j++) {
                var child = document.createElement("div");
                child.id = years[i] + fields[j]
                var html = fields[j] + ':<br>' + patent[fields[j]] + '<br><br>';
                child.innerHTML = html;
                parent.appendChild(child);
            }
            $(parent).prepend(
                `<div class="samples_click"></div>
                <svg class="clipboard_icon" xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="#b6b6b6" class="bi bi-files" viewBox="0 0 16 16">
                  <path d="M13 0H6a2 2 0 0 0-2 2 2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h7a2 2 0 0 0 2-2 2 2 0 0 0 2-2V2a2 2 0 0 0-2-2zm0 13V4a2 2 0 0 0-2-2H5a1 1 0 0 1 1-1h7a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1zM3 4a1 1 0 0 1 1-1h7a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V4z"/>
                </svg>
                <svg class="checkmark_icon" xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="#b6b6b6" class="bi bi-check2" viewBox="0 0 16 16">
                  <path d="M13.854 3.646a.5.5 0 0 1 0 .708l-7 7a.5.5 0 0 1-.708 0l-3.5-3.5a.5.5 0 1 1 .708-.708L6.5 10.293l6.646-6.647a.5.5 0 0 1 .708 0z"/>
                </svg>`
            );
            var click_id = years[i] + '_click';
            //const elm_parent = $(parent).find($('.matches_click')).parent();
            $(parent).find($('.samples_click')).attr('id', click_id)
            document.getElementById(click_id).addEventListener('click', function(e) {
                copyPasta(e.target.id);
            });
        }
    });
}

function copyPasta(click_id) {
    let element = document.getElementById(click_id);
    let parent_element = element.parentElement;
    let clipboard_icon = $(parent_element).find('.clipboard_icon');
    let checkmark_icon = $(parent_element).find('.checkmark_icon');
    clipboard_icon.css('visibility', 'hidden')
    checkmark_icon.css('visibility', 'visible')

    let select_id = click_id.substring(0, 4) + 'pca_embedding';
    let tt_id = click_id.substring(0, 4) +  'tt';
    console.log(tt_id)

    let elm = document.getElementById(select_id);
    let copy = elm.textContent.substring(14);
    console.log(copy)

    var dummy = document.getElementById("copyfrom");
    dummy.value = copy;
    dummy.select();
    document.execCommand("copy");

    document.getElementById(tt_id).style.visibility = 'visible';
    setTimeout(function() {
        clipboard_icon.css('visibility', 'visible')
        checkmark_icon.css('visibility', 'hidden')
        document.getElementById(tt_id).style.visibility = 'hidden';
    }, 8000);
}

$(document).ready(function() {
    $("html, body").animate({ scrollTop: 0 }, "slow");

    $('#left').hide();
    $('#left').show();

    scrollResizeEvents();
    setPosOnload();

    const ref = {
        0: "1970",
        1: "1980",
        2: "1990",
        3: "2000",
        4: "2010",
        5: "2020",
    }
    const next = {
        "0": 1,
        "1": 2,
        "2": 3,
        "3": 4,
        "4": 5,
        "5": 0,
    }
    const prev = {
        "0": 5,
        "1": 0,
        "2": 1,
        "3": 2,
        "4": 3,
        "5": 4,
    }

    initSamples();
    populateCarousel("1970");
    idx = 0;

    $('#plus').on('click', function(){
        idx = next[idx.toString()];
        populateCarousel(ref[idx]);
        var c = document.querySelector('#patent_matches');
        var carousel = new bootstrap.Carousel(c);
        carousel.to(0)
        carousel.pause()
    });
    $('#minus').on('click', function(){
        idx = prev[idx.toString()];
        populateCarousel(ref[idx]);
        var c = document.querySelector('#patent_matches');
        var carousel = new bootstrap.Carousel(c);
        carousel.to(0)
        carousel.pause()
    });

    $('#back-top').on('click', function(){
        $("html, body").animate({ scrollTop: 0 }, "slow");
    });
    
    var suffix = ['cos', 'mnh', 'euc']
    for (var i=0; i<15; i++) {
        for (var j=0; j<suffix.length; j++) {
            var click_id = i + suffix[j] + '_click';
            //console.log(click_id)
        }
    }

    const hug = document.getElementById('hug');
    const sbert = document.getElementById('sbert');

    $(hug).mouseover(function() {
        $(this).css("background-size", "100% auto");
    });
    $(hug).mouseout(function() {
        $(this).css("background-size", "95% auto");
    });
    $(sbert).mouseover(function() {
        $(this).css("background-size", "100% auto");
    });
    $(sbert).mouseout(function() {
        $(this).css("background-size", "95% auto");
    });

});