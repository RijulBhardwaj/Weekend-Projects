(function(){
'use strict';
var App=window.App=window.App||{};
var U=App.ui;
var TAX=App.taxonomy={
 VARC:['Reading Comprehension','Para Jumbles','Para Summary','Odd Sentence Out','Sentence Placement'],
 DILR:['Arrangements','Games & Tournaments','Scheduling','Selection & Distribution','Tables','Graphs & Charts','Caselets','Routes & Networks','Venn Diagrams','Quantitative Reasoning','Hybrid Sets'],
 QA:['Arithmetic','Algebra','Geometry & Mensuration','Number System','Modern Math']};
App.SECTIONS=['VARC','DILR','QA'];
App.SECNAME={VARC:'Verbal Ability & Reading Comprehension',DILR:'Data Interpretation & Logical Reasoning',QA:'Quantitative Ability'};
var D=App.data={imported:[],qIndex:{},patIdx:{}};
function cd(k){return (window.CATDATA&&window.CATDATA[k])||[]}
D.cd=cd;
D.papers=function(){return cd('papers').concat(D.imported)};
D.paper=function(id){return D.papers().filter(function(p){return p.id===id})[0]||null};
D.patterns=function(){return cd('patterns')};
D.buildIndex=function(){
  var ix=D.qIndex={};
  function addQ(q,sec,g,paper,pat){q._sec=q._sec||sec;if(pat&&!q.pattern)q.pattern=pat;ix[q.id]={q:q,group:g,paper:paper}}
  D.papers().forEach(function(p){(p.sections||[]).forEach(function(s){(s.groups||[]).forEach(function(g){(g.questions||[]).forEach(function(q){q._sec=s.id;q._paper=p.id;ix[q.id]={q:q,group:g,paper:p}})})})});
  D.patterns().forEach(function(p){
    (p.practice||[]).forEach(function(q){addQ(q,p.section,{id:'pg-'+q.id,type:'standalone',text:'',questions:[q]},null,p.id);q._pat=p.id;q.topic=q.topic||p.topic});
    (p.practiceGroups||[]).forEach(function(g){(g.questions||[]).forEach(function(q){addQ(q,p.section,g,null,p.id);q._pat=p.id;q.topic=q.topic||p.topic})});
  });
};
D.findQ=function(id){return D.qIndex[id]||null};
D.allQuestions=function(){return Object.keys(D.qIndex).map(function(k){return D.qIndex[k]})};
/* ---- scoring ---- */
function normTita(a){return String(a==null?'':a).replace(/[\s,]/g,'').toLowerCase()}
D.titaEqual=function(user,key){var u=normTita(user),k=normTita(key);if(u===''||k==='')return false;
  var nu=Number(u),nk=Number(k);if(isFinite(nu)&&isFinite(nk)&&/^[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/.test(u)&&/^[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/.test(k))return Math.abs(nu-nk)<=1e-9*Math.max(1,Math.abs(nk));return u===k};
D.scoreQ=function(q,ans){
  var has=ans!==null&&ans!==undefined&&String(ans).trim()!=='';
  if(!has)return {status:'unattempted',marks:0};
  var ok=q.type==='tita'?D.titaEqual(ans,q.answer):(+ans===+q.answer);
  return ok?{status:'correct',marks:q.marks||0}:{status:'incorrect',marks:q.type==='tita'?-(q.negative||0):-(q.negative||0)};
};
D.answerText=function(q){if(q.type==='tita')return String(q.answer);var i=+q.answer;return String.fromCharCode(65+i)+'. '+((q.options||[])[i]||'')};
D.userAnswerText=function(q,a){if(a==null||a==='')return '(not attempted)';if(q.type==='tita')return String(a);return String.fromCharCode(65+ +a)+'. '+((q.options||[])[+a]||'')};
/* ---- paper validation (import) ---- */
D.validatePaper=function(o){
  var errs=[],warn=[];
  function E(m){if(errs.length<40)errs.push(m)}
  if(!o||typeof o!=='object'||Array.isArray(o))return {ok:false,errors:['Top level must be a JSON object (a single paper)']};
  if(typeof o.id!=='string'||!/^[A-Za-z0-9_-]{2,60}$/.test(o.id))E('id must be a string of 2-60 chars (letters, digits, - or _)');
  if(typeof o.title!=='string'||!o.title.trim())E('title is required');
  if(!Array.isArray(o.sections)||!o.sections.length)E('sections must be a non-empty array');
  var seen={},nq=0;
  (Array.isArray(o.sections)?o.sections:[]).forEach(function(s,si){
    var sp='sections['+si+']';
    if(!s||typeof s.id!=='string'||!s.id)E(sp+'.id is required');
    if(!s||typeof s.name!=='string'||!s.name)E(sp+'.name is required');
    if(!s||typeof s.timeMin!=='number'||s.timeMin<=0)E(sp+'.timeMin must be a positive number');
    if(!s||!Array.isArray(s.groups)||!s.groups.length){E(sp+'.groups must be a non-empty array');return}
    s.groups.forEach(function(g,gi){var gp=sp+'.groups['+gi+']';
      if(!g||typeof g.id!=='string'||!g.id)E(gp+'.id is required');
      if(!g||['passage','set','standalone'].indexOf(g.type)<0)E(gp+'.type must be passage, set or standalone');
      if(!g||!Array.isArray(g.questions)||!g.questions.length){E(gp+'.questions must be a non-empty array');return}
      g.questions.forEach(function(q,qi){var qp=gp+'.questions['+qi+']';nq++;
        if(!q||typeof q.id!=='string'||!q.id){E(qp+'.id is required');return}
        if(seen[q.id])E(qp+': duplicate question id '+q.id);seen[q.id]=1;
        if(q.type!=='mcq'&&q.type!=='tita'){E(qp+'.type must be mcq or tita');return}
        if(typeof q.stem!=='string'||!q.stem.trim())E(qp+'.stem is required');
        if(q.type==='mcq'){
          if(!Array.isArray(q.options)||q.options.length<2)E(qp+'.options must list at least 2 options');
          else if(!Number.isInteger(q.answer)||q.answer<0||q.answer>=q.options.length)E(qp+'.answer must be a 0-based option index (0..'+(q.options.length-1)+')');
        }else if(q.answer===undefined||q.answer===null||String(q.answer).trim()==='')E(qp+'.answer must be a non-empty string or number for TITA');
        if(typeof q.marks!=='number'||q.marks<0)E(qp+'.marks must be a number >= 0');
        if(q.negative!==undefined&&(typeof q.negative!=='number'||q.negative<0))E(qp+'.negative must be a number >= 0');
      });});});
  if(errs.length)return {ok:false,errors:errs};
  if(D.cd('papers').some(function(p){return p.id===o.id}))return {ok:false,errors:['id "'+o.id+'" collides with a bundled paper; choose a different id']};
  var p=JSON.parse(JSON.stringify(o));
  p.kind=p.kind==='mock'?'mock':'pyq';p.origin='user-import';p.year=p.year||null;p.slot=p.slot||null;p.sources=p.sources||[];p.philosophy=p.philosophy||'';
  p.sectionalTimeLimit=!!p.sectionalTimeLimit;
  p.durationMin=p.durationMin||p.sections.reduce(function(a,s){return a+s.timeMin},0);
  p.title=U.sanitize(p.title).replace(/<[^>]+>/g,'');
  p.sections.forEach(function(s){s.groups.forEach(function(g){g.text=U.sanitize(g.text||'');g.title=g.title||'';g.meta=g.meta||{};
    g.questions.forEach(function(q,i){q.n=q.n||i+1;q.stem=U.sanitize(q.stem);if(q.options)q.options=q.options.map(function(x){return U.sanitize(x)});
      q.negative=q.negative==null?(q.type==='mcq'?1:0):q.negative;q.topic=q.topic||'Unclassified';q.subtopic=q.subtopic||'';q.pattern=q.pattern||null;
      q.difficulty=['easy','medium','hard'].indexOf(q.difficulty)>=0?q.difficulty:'medium';q.estTimeSec=q.estTimeSec>0?q.estTimeSec:120;
      q.solution=U.sanitize(q.solution||'');q.altMethod=U.sanitize(q.altMethod||'');q.commonMistakes=q.commonMistakes||[];q.concepts=q.concepts||[];q.distractors='';q.verification=q.verification||'unverified'})})});
  if(!p.sections.some(function(s){return s.timeMin}))warn.push('no section times');
  return {ok:true,paper:p,errors:[],warnings:warn,nq:nq};
};
D.sampleTemplate=function(){return {id:'my-paper-1',title:'My imported paper',kind:'pyq',year:2023,slot:1,durationMin:120,sectionalTimeLimit:true,sections:[
 {id:'VARC',name:'Verbal Ability & Reading Comprehension',timeMin:40,groups:[{id:'g1',type:'passage',title:'Passage 1',text:'<p>Passage text here.</p>',questions:[
   {id:'my1-v1',n:1,type:'mcq',stem:'<p>Question text?</p>',options:['Option A','Option B','Option C','Option D'],answer:1,marks:3,negative:1,topic:'Reading Comprehension',subtopic:'Main Idea',difficulty:'medium',estTimeSec:90,solution:'<p>Why B.</p>'}]}]},
 {id:'QA',name:'Quantitative Ability',timeMin:40,groups:[{id:'g2',type:'standalone',text:'',questions:[
   {id:'my1-q1',n:1,type:'tita',stem:'<p>Compute 6 &times; 7.</p>',answer:'42',marks:3,negative:0,topic:'Arithmetic',difficulty:'easy',estTimeSec:60}]}]}]}};
/* ---- derive a runnable exam paper from a spec ---- */
D.deriveExam=function(spec){
  var hook=window.__CAT_TEST__||{};
  var base=spec.snap||D.paper(spec.paperId);
  if(!base)return null;
  var secs=[],enforced=false,overall=null,timed=spec.timed!==false;
  if(spec.mode==='full'){
    secs=base.sections.map(function(s){return {id:s.id,name:s.name,timeSec:null,groups:s.groups,estTimeMin:s.timeMin}});
    if(timed){
      if(base.sectionalTimeLimit&&secs.length>1){enforced=true;secs.forEach(function(s){s.timeSec=(hook.sectionSec||s.estTimeMin*60)});overall=secs.reduce(function(a,s){return a+s.timeSec},0)}
      else overall=hook.overallSec||(base.durationMin||secs.reduce(function(a,s){return a+s.estTimeMin},0))*60;
    }
  }else if(spec.mode==='section'){
    var s=base.sections.filter(function(x){return x.id===spec.sectionId})[0]||base.sections[0];
    secs=[{id:s.id,name:s.name,groups:s.groups,timeSec:timed?(hook.sectionSec||s.timeMin*60):null}];overall=secs[0].timeSec;
  }else if(spec.mode==='group'){
    var g=null,sec=null;base.sections.forEach(function(s){s.groups.forEach(function(x){if(x.id===spec.groupId){g=x;sec=s}})});if(!g)return null;
    var est=g.questions.reduce(function(a,q){return a+(q.estTimeSec||120)},0);
    secs=[{id:sec.id,name:sec.name+' (set)',groups:[g],timeSec:timed?(hook.sectionSec||Math.max(120,est)):null}];overall=secs[0].timeSec;
  }else{ // virtual practice paper
    secs=base.sections.map(function(s){return {id:s.id,name:s.name,groups:s.groups,timeSec:timed?(hook.sectionSec||(spec.minutes||30)*60):null}});overall=secs[0].timeSec;
  }
  return {id:base.id,title:spec.title||base.title,sections:secs,enforced:enforced,overallSec:overall,timed:timed};
};
D.groupQs=function(g,sec){return (g.questions||[])};
/* ---- load all user data ---- */
D.load=function(){
  var S=App.store,names=['attempts','practiceLog','mistakes','revision','trackers','goals','notes','importedPapers','settings'];
  return Promise.all(names.map(function(n){return S.all(n)})).then(function(r){
    var o={};names.forEach(function(n,i){o[n]=r[i]});
    o.settingsMap={};o.settings.forEach(function(s){o.settingsMap[s.key]=s.value});
    o.attempts.sort(function(a,b){return a.submittedAt-b.submittedAt});
    return o;
  });
};
D.refreshImported=function(){return App.store.all('importedPapers').then(function(r){D.imported=r;D.buildIndex();return r})};
D.intervals=function(Dd){var v=Dd&&Dd.settingsMap&&Dd.settingsMap.intervals;return Array.isArray(v)&&v.length?v:[1,3,7,14,30]};
})();
