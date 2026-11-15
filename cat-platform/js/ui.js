(function(){
'use strict';
var App=window.App=window.App||{};
var U=App.ui={};
U.esc=function(s){return String(s==null?'':s).replace(/[&<>"']/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]})};
var esc=U.esc;
U.$=function(s,r){return (r||document).querySelector(s)};
U.$$=function(s,r){return Array.prototype.slice.call((r||document).querySelectorAll(s))};
U.pct=function(v,d){return v==null||isNaN(v)?'-':(+v).toFixed(d==null?0:d)+'%'};
U.num=function(v,d){return v==null||isNaN(v)?'-':(+v).toFixed(d==null?1:d).replace(/\.0+$/,'')};
U.dur=function(sec){sec=Math.round(sec||0);if(sec<60)return sec+'s';var m=Math.floor(sec/60),s=sec%60;if(m<60)return m+'m '+(s<10?'0':'')+s+'s';var h=Math.floor(m/60);return h+'h '+(m%60)+'m'};
U.clock=function(sec){sec=Math.max(0,Math.ceil(sec));var h=Math.floor(sec/3600),m=Math.floor(sec%3600/60),s=sec%60;return (h?h+':':'')+(h&&m<10?'0':'')+m+':'+(s<10?'0':'')+s};
U.date=function(ts){if(!ts)return '-';var d=new Date(ts);return d.toLocaleDateString(undefined,{year:'numeric',month:'short',day:'numeric'})};
U.dt=function(ts){var d=new Date(ts);return U.date(ts)+' '+d.toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit'})};
U.uid=function(p){return (p||'id')+'-'+Date.now().toString(36)+Math.random().toString(36).slice(2,7)};
U.badge=function(text,kind){return '<span class="badge '+(kind||'')+'">'+esc(text)+'</span>'};
U.verBadge=function(v){var k={official:'ok',documented:'ok',computed:'ok','re-solved':'info','reputable-third-party':'info','commonly-reported':'info',reconstructed:'warn',unverified:'warn',hypothesis:'warn'}[v]||'';return U.badge(v||'unverified',k)};
U.empty=function(title,text,cta){return '<div class="empty"><strong>'+esc(title)+'</strong>'+(text?'<div>'+text+'</div>':'')+(cta?'<div class="mt">'+cta+'</div>':'')+'</div>'};
U.sanitize=function(html){
  var doc=new DOMParser().parseFromString('<body>'+String(html==null?'':html)+'</body>','text/html');
  U.$$('script,iframe,object,embed,link,meta,style,base,form',doc.body).forEach(function(n){n.remove()});
  U.$$('*',doc.body).forEach(function(n){Array.prototype.slice.call(n.attributes).forEach(function(a){
    if(/^on/i.test(a.name)||(/^(href|src|xlink:href)$/i.test(a.name)&&/^\s*javascript:/i.test(a.value)))n.removeAttribute(a.name)})});
  return doc.body.innerHTML;
};
/* theme */
U.theme=function(){try{return localStorage.getItem('cat.theme')||'auto'}catch(e){return 'auto'}};
U.setTheme=function(t){try{localStorage.setItem('cat.theme',t)}catch(e){}
  if(t==='auto')document.documentElement.removeAttribute('data-theme');else document.documentElement.setAttribute('data-theme',t)};
/* save status */
U.setSaved=function(){var d=new Date(),p=function(n){return (n<10?'0':'')+n};var el=document.getElementById('saveStatus');if(el)el.textContent='Saved '+p(d.getHours())+':'+p(d.getMinutes())+':'+p(d.getSeconds())};
/* toasts */
U.toast=function(msg,opts){opts=opts||{};var box=document.getElementById('toasts'),t=document.createElement('div');t.className='toast';t.setAttribute('role','status');
  t.innerHTML='<span>'+esc(msg)+'</span>';
  if(opts.action){var b=document.createElement('button');b.textContent=opts.action.label;b.onclick=function(){t.remove();opts.action.fn()};t.appendChild(b)}
  box.appendChild(t);setTimeout(function(){t.remove()},opts.ms||(opts.action?9000:3500));return t};
/* modal */
U.modal=function(o){return new Promise(function(res){
  var back=document.createElement('div');back.className='modal-back';
  var btns=(o.buttons||[{label:'OK',value:true,kind:'primary'}]).map(function(b,i){return '<button class="'+(b.kind||'')+'" data-i="'+i+'" '+(b.id?'id="'+b.id+'"':'')+'>'+esc(b.label)+'</button>'}).join('');
  back.innerHTML='<div class="modal" role="dialog" aria-modal="true" aria-label="'+esc(o.title)+'"><h2>'+esc(o.title)+'</h2><div class="mbody">'+(o.body||'')+'</div><div class="actions">'+btns+'</div></div>';
  var prev=document.activeElement;
  function close(v){back.remove();document.removeEventListener('keydown',key);if(prev&&prev.focus)try{prev.focus()}catch(e){};res(v)}
  function key(e){if(e.key==='Escape'&&!o.noEsc)close(o.escValue===undefined?false:o.escValue)}
  document.addEventListener('keydown',key);
  back.addEventListener('click',function(e){var b=e.target.closest('button[data-i]');if(b){var bt=(o.buttons||[{value:true}])[+b.dataset.i];close(typeof bt.value==='function'?bt.value(back):bt.value)}});
  document.body.appendChild(back);var f=back.querySelector('.actions button.primary,.actions button');if(f)f.focus();
  if(o.onOpen)o.onOpen(back);
});};
U.confirm=function(title,body,o){o=o||{};return U.modal({title:title,body:body,buttons:[{label:o.cancel||'Cancel',value:false},{label:o.ok||'Confirm',value:true,kind:o.danger?'danger solid':'primary',id:'modalOk'}]})};
U.alert=function(title,body){return U.modal({title:title,body:body})};
/* sortable table. cols:[{key,label,num,get(row),render(row)->html,sort:false}] */
U.table=function(el,o){
  var st={key:o.sortKey||null,dir:o.sortDir||1},cols=o.cols,rows=o.rows;
  function val(c,r){return c.get?c.get(r):r[c.key]}
  function draw(){
    var data=rows.slice();
    if(st.key){var c=cols.filter(function(x){return x.key===st.key})[0];if(c)data.sort(function(a,b){var x=val(c,a),y=val(c,b);if(x==null&&y==null)return 0;if(x==null)return 1;if(y==null)return -1;return (typeof x==='number'&&typeof y==='number'?x-y:String(x).localeCompare(String(y),undefined,{numeric:true}))*st.dir})}
    var h='<div class="tablewrap"><table'+(o.id?' id="'+o.id+'"':'')+'><thead><tr>'+cols.map(function(c){
      var sortable=c.sort!==false;var ar=st.key===c.key?(st.dir>0?' &#9650;':' &#9660;'):'';
      return '<th class="'+(c.num?'num ':'')+(sortable?'sortable':'')+'" '+(sortable?'tabindex="0" role="button" data-k="'+c.key+'" aria-sort="'+(st.key===c.key?(st.dir>0?'ascending':'descending'):'none')+'"':'')+'>'+esc(c.label)+ar+'</th>'}).join('')+'</tr></thead><tbody>'+
      (data.length?data.map(function(r,i){return '<tr'+(o.rowAttr?' '+o.rowAttr(r):'')+'>'+cols.map(function(c){var v=val(c,r);return '<td class="'+(c.num?'num':'')+'">'+(c.render?c.render(r):esc(v==null?'-':v))+'</td>'}).join('')+'</tr>'}).join(''):'<tr><td colspan="'+cols.length+'" class="muted">'+(o.emptyText||'No rows')+'</td></tr>')+'</tbody></table></div>';
    el.innerHTML=h;
  }
  function sortBy(k){if(st.key===k)st.dir=-st.dir;else{st.key=k;st.dir=1}draw()}
  el.onclick=function(e){var th=e.target.closest('th[data-k]');if(th)return sortBy(th.dataset.k);if(o.onRow){var tr=e.target.closest('tr[data-id]');if(tr&&!e.target.closest('a,button,input,select'))o.onRow(tr.dataset.id)}};
  el.onkeydown=function(e){if(e.key==='Enter'||e.key===' '){var th=e.target.closest('th[data-k]');if(th){e.preventDefault();sortBy(th.dataset.k)}}};
  draw();return {redraw:function(r){if(r)rows=r;draw()}};
};
U.setRows=function(){};
U.page=function(title,sub,actions){return '<div class="page-head"><div><h1>'+esc(title)+'</h1>'+(sub?'<div class="muted">'+sub+'</div>':'')+'</div><div class="row">'+(actions||'')+'</div></div>'};
U.tabs=function(base,items,cur){return '<div class="tabs" role="tablist">'+items.map(function(i){return '<a role="tab" href="#/'+base+'/'+i[0]+'" class="'+(i[0]===cur?'active':'')+'" aria-selected="'+(i[0]===cur)+'">'+esc(i[1])+'</a>'}).join('')+'</div>'};
U.download=function(name,text,type){var b=new Blob([text],{type:type||'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=name;document.body.appendChild(a);a.click();setTimeout(function(){a.remove();URL.revokeObjectURL(a.href)},500)};
U.readFile=function(f){return new Promise(function(res,rej){var r=new FileReader();r.onload=function(){res(r.result)};r.onerror=function(){rej(r.error)};r.readAsText(f)})};
U.std=function(a){if(a.length<2)return null;var m=a.reduce(function(x,y){return x+y},0)/a.length;return Math.sqrt(a.reduce(function(s,v){return s+(v-m)*(v-m)},0)/a.length)};
U.mean=function(a){return a.length?a.reduce(function(x,y){return x+y},0)/a.length:null};
})();
