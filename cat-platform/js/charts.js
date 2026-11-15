(function(){
'use strict';
var App=window.App=window.App||{};
var esc=App.ui.esc;
var COL=['var(--s1)','var(--s2)','var(--s3)','var(--s4)','var(--s5)'];
function niceMax(v){if(v<=0)return 1;var p=Math.pow(10,Math.floor(Math.log10(v))),f=v/p;return (f<=1?1:f<=2?2:f<=5?5:10)*p}
function svg(w,h,inner,label){return '<svg class="chart" viewBox="0 0 '+w+' '+h+'" role="img" aria-label="'+esc(label||'chart')+'" preserveAspectRatio="xMidYMid meet">'+inner+'</svg>'}
function legend(names){return '<div class="legend">'+names.map(function(n,i){return '<span><i style="background:'+COL[i%5]+'"></i>'+esc(n)+'</span>'}).join('')+'</div>'}
var C=App.charts={
  COL:COL,
  /* items:[{label,value,href}] */
  bar:function(items,o){o=o||{};if(!items.length)return '';var W=560,H=o.h||220,L=40,B=40,T=10,R=8,max=o.max||niceMax(Math.max.apply(null,items.map(function(i){return i.value||0})));
    var bw=(W-L-R)/items.length,s='';
    for(var g=0;g<=4;g++){var y=T+(H-T-B)*(1-g/4);s+='<line class="grid" x1="'+L+'" x2="'+(W-R)+'" y1="'+y+'" y2="'+y+'"/><text x="'+(L-5)+'" y="'+(y+4)+'" text-anchor="end">'+Math.round(max*g/4*10)/10+'</text>'}
    items.forEach(function(it,i){var v=it.value||0,hh=(H-T-B)*v/max,x=L+i*bw+bw*.15,y=H-B-hh;
      var r='<rect x="'+x+'" y="'+y+'" width="'+bw*.7+'" height="'+hh+'" rx="2" fill="'+(it.color||COL[0])+'"><title>'+esc(it.label)+': '+v+'</title></rect><text x="'+(x+bw*.35)+'" y="'+(H-B+14)+'" text-anchor="middle">'+esc(String(it.label).slice(0,10))+'</text>';
      if(items.length<=14)r+='<text x="'+(x+bw*.35)+'" y="'+(y-3)+'" text-anchor="middle">'+(Math.round(v*10)/10)+'</text>';
      s+=it.href?'<a href="'+it.href+'">'+r+'</a>':r});
    return svg(W,H,s,o.label||'bar chart')},
  /* categories:[label], series:[{name,values:[]}] */
  stacked:function(cats,series,o){o=o||{};if(!cats.length)return '';var W=560,H=o.h||240,L=40,B=40,T=10,R=8;
    var tot=cats.map(function(c,i){return series.reduce(function(a,s){return a+(s.values[i]||0)},0)}),max=niceMax(Math.max.apply(null,tot)),bw=(W-L-R)/cats.length,s='';
    for(var g=0;g<=4;g++){var y=T+(H-T-B)*(1-g/4);s+='<line class="grid" x1="'+L+'" x2="'+(W-R)+'" y1="'+y+'" y2="'+y+'"/><text x="'+(L-5)+'" y="'+(y+4)+'" text-anchor="end">'+Math.round(max*g/4*10)/10+'</text>'}
    cats.forEach(function(c,i){var acc=0;series.forEach(function(se,si){var v=se.values[i]||0,hh=(H-T-B)*v/max,y=H-B-(H-T-B)*acc/max-hh;acc+=v;
      s+='<rect x="'+(L+i*bw+bw*.15)+'" y="'+y+'" width="'+bw*.7+'" height="'+hh+'" fill="'+COL[si%5]+'"><title>'+esc(c)+' '+esc(se.name)+': '+v+'</title></rect>'});
      s+='<text x="'+(L+i*bw+bw*.5)+'" y="'+(H-B+14)+'" text-anchor="middle">'+esc(c)+'</text>'});
    return svg(W,H,s,o.label||'stacked bar chart')+legend(series.map(function(x){return x.name}))},
  /* series:[{name,points:[{x:label,y:number}]}] shared x labels from first series */
  line:function(series,o){o=o||{};var all=[];series.forEach(function(s){s.points.forEach(function(p){if(p.y!=null)all.push(p.y)})});if(!all.length)return '';
    var W=560,H=o.h||220,L=42,B=36,T=10,R=12,max=o.max!=null?o.max:niceMax(Math.max.apply(null,all)),min=o.min!=null?o.min:Math.min(0,Math.min.apply(null,all)),n=Math.max.apply(null,series.map(function(s){return s.points.length})),s='';
    var sx=function(i){return n<2?(L+W-R)/2:L+(W-L-R)*i/(n-1)},sy=function(v){return T+(H-T-B)*(1-(v-min)/((max-min)||1))};
    for(var g=0;g<=4;g++){var v=min+(max-min)*g/4,y=sy(v);s+='<line class="grid" x1="'+L+'" x2="'+(W-R)+'" y1="'+y+'" y2="'+y+'"/><text x="'+(L-5)+'" y="'+(y+4)+'" text-anchor="end">'+Math.round(v*10)/10+'</text>'}
    var lab=series[0].points;var step=Math.ceil(lab.length/10);lab.forEach(function(p,i){if(i%step===0)s+='<text x="'+sx(i)+'" y="'+(H-B+15)+'" text-anchor="middle">'+esc(String(p.x).slice(0,8))+'</text>'});
    series.forEach(function(se,si){var d='',pen=false;se.points.forEach(function(p,i){if(p.y==null){pen=false;return}d+=(pen?'L':'M')+sx(i)+' '+sy(p.y)+' ';pen=true});
      s+='<path d="'+d+'" fill="none" stroke="'+COL[si%5]+'" stroke-width="2.2" stroke-linejoin="round"/>';
      se.points.forEach(function(p,i){if(p.y!=null)s+='<circle cx="'+sx(i)+'" cy="'+sy(p.y)+'" r="3.5" fill="'+COL[si%5]+'"><title>'+esc(se.name)+' '+esc(p.x)+': '+(Math.round(p.y*10)/10)+'</title></circle>'})});
    return svg(W,H,s,o.label||'line chart')+(series.length>1?legend(series.map(function(x){return x.name})):'')},
  spark:function(vals,w,h){w=w||110;h=h||28;vals=vals.filter(function(v){return v!=null});if(vals.length<2)return '<span class="muted small">no history</span>';var mn=Math.min.apply(null,vals),mx=Math.max.apply(null,vals),d=vals.map(function(v,i){return (i?'L':'M')+(i*(w-4)/(vals.length-1)+2)+' '+(h-3-(h-6)*(v-mn)/((mx-mn)||1))}).join(' ');
    return '<svg class="spark" width="'+w+'" height="'+h+'" viewBox="0 0 '+w+' '+h+'" role="img" aria-label="history sparkline"><path d="'+d+'" fill="none" stroke="var(--s1)" stroke-width="2"/></svg>'},
  ring:function(value,o){o=o||{};var size=o.size||110,r=size/2-9,c=2*Math.PI*r,v=value==null?0:Math.max(0,Math.min(100,value)),col=o.color||'var(--s1)';
    return '<svg width="'+size+'" height="'+size+'" viewBox="0 0 '+size+' '+size+'" role="img" aria-label="'+esc(o.label||'progress')+' '+(value==null?'no data':Math.round(v)+'%')+'"><circle cx="'+size/2+'" cy="'+size/2+'" r="'+r+'" fill="none" stroke="var(--surface2)" stroke-width="10"/>'+(value==null?'':'<circle cx="'+size/2+'" cy="'+size/2+'" r="'+r+'" fill="none" stroke="'+col+'" stroke-width="10" stroke-linecap="round" stroke-dasharray="'+c*v/100+' '+c+'" transform="rotate(-90 '+size/2+' '+size/2+')"/>')+
      '<text x="50%" y="50%" text-anchor="middle" dominant-baseline="central" style="fill:var(--text);font:700 '+size/5+'px var(--font)">'+(value==null?'-':Math.round(v)+'%')+'</text></svg>'},
  donut:function(parts,o){o=o||{};var size=o.size||140,r=size/2-12,c=2*Math.PI*r,tot=parts.reduce(function(a,p){return a+p.value},0);if(!tot)return '';var off=0,s='';
    parts.forEach(function(p,i){var len=c*p.value/tot;s+='<circle cx="'+size/2+'" cy="'+size/2+'" r="'+r+'" fill="none" stroke="'+(p.color||COL[i%5])+'" stroke-width="18" stroke-dasharray="'+len+' '+(c-len)+'" stroke-dashoffset="'+(-off)+'" transform="rotate(-90 '+size/2+' '+size/2+')"><title>'+esc(p.label)+': '+p.value+'</title></circle>';off+=len});
    return '<svg width="'+size+'" height="'+size+'" viewBox="0 0 '+size+' '+size+'" role="img" aria-label="'+esc(o.label||'donut')+'">'+s+'</svg>'+legend(parts.map(function(p){return p.label+' ('+p.value+')'}))},
  /* rows:[label], cols:[label], m[r][c] number|null ; lv by thresholds */
  heatmap:function(rows,cols,m,o){o=o||{};if(!rows.length||!cols.length)return '';var cw=o.cw||34,ch=22,L=o.L||150,T=24,W=L+cols.length*cw+6,H=T+rows.length*ch+6,max=0,s='';
    m.forEach(function(r){r.forEach(function(v){if(v!=null&&v>max)max=v})});
    cols.forEach(function(c,i){s+='<text x="'+(L+i*cw+cw/2)+'" y="'+(T-8)+'" text-anchor="middle">'+esc(c)+'</text>'});
    rows.forEach(function(r,ri){s+='<text x="'+(L-6)+'" y="'+(T+ri*ch+15)+'" text-anchor="end">'+esc(String(r).slice(0,24))+'</text>';
      cols.forEach(function(c,ci){var v=m[ri][ci],a=v==null?0:(max?v/max:0);
        s+='<rect x="'+(L+ci*cw+1)+'" y="'+(T+ri*ch+1)+'" width="'+(cw-2)+'" height="'+(ch-2)+'" rx="3" fill="'+(v==null?'var(--surface2)':'var(--s1)')+'" fill-opacity="'+(v==null?1:(0.12+0.88*a))+'"><title>'+esc(r)+' / '+esc(c)+': '+(v==null?'no data':v)+'</title></rect>';
        if(v!=null&&v>0)s+='<text x="'+(L+ci*cw+cw/2)+'" y="'+(T+ri*ch+15)+'" text-anchor="middle" style="fill:'+(a>.55?'#fff':'var(--text)')+'">'+v+'</text>'})});
    return '<div style="overflow-x:auto">'+svg(W,H,s,o.label||'heatmap').replace('class="chart"','class="chart" style="min-width:'+Math.min(W,640)+'px;max-height:none"')+'</div>'},
  /* pts:[{x,y,label,href}] */
  scatter:function(pts,o){o=o||{};if(!pts.length)return '';var W=560,H=o.h||280,L=44,B=40,T=10,R=12,mx=niceMax(Math.max.apply(null,pts.map(function(p){return p.x}).concat([o.xref||1]))*1.1),my=o.ymax||100,s='';
    var sx=function(v){return L+(W-L-R)*v/mx},sy=function(v){return T+(H-T-B)*(1-v/my)};
    for(var g=0;g<=4;g++){var y=sy(my*g/4);s+='<line class="grid" x1="'+L+'" x2="'+(W-R)+'" y1="'+y+'" y2="'+y+'"/><text x="'+(L-5)+'" y="'+(y+4)+'" text-anchor="end">'+Math.round(my*g/4)+'</text><text x="'+sx(mx*g/4)+'" y="'+(H-B+15)+'" text-anchor="middle">'+(Math.round(mx*g/4*10)/10)+'</text>'}
    if(o.xref)s+='<line x1="'+sx(o.xref)+'" x2="'+sx(o.xref)+'" y1="'+T+'" y2="'+(H-B)+'" stroke="var(--s3)" stroke-dasharray="4 3"/>';
    s+='<text x="'+(W/2)+'" y="'+(H-4)+'" text-anchor="middle">'+esc(o.xlabel||'x')+'</text><text x="12" y="'+(H/2)+'" transform="rotate(-90 12 '+(H/2)+')" text-anchor="middle">'+esc(o.ylabel||'y')+'</text>';
    pts.forEach(function(p){var c='<circle cx="'+sx(p.x)+'" cy="'+sy(p.y)+'" r="'+(p.r||5)+'" fill="var(--s1)" fill-opacity=".75" stroke="var(--surface)"><title>'+esc(p.label)+' ('+(Math.round(p.x*100)/100)+', '+Math.round(p.y)+'%)</title></circle>';s+=p.href?'<a href="'+p.href+'">'+c+'</a>':c});
    return svg(W,H,s,o.label||'scatter')}
};
})();
