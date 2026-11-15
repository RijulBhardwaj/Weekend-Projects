(function(){
'use strict';
var App = window.App = window.App || {};
App.views = App.views || {};
var DB_NAME='catPlatform', DB_VERSION=1, DATA_VERSION=1;
var KEYS={attempts:'id',activeSession:'id',practiceLog:'id',mistakes:'id',revision:'id',trackers:'id',goals:'id',notes:'id',settings:'key',importedPapers:'id',backups:'id',undo:'id'};
var USER_STORES=['attempts','practiceLog','mistakes','revision','trackers','goals','notes','settings','importedPapers'];
// Migrations: index = target version. Add new entries for future schema changes and bump DB_VERSION.
var MIGRATIONS={1:function(db){Object.keys(KEYS).forEach(function(n){if(!db.objectStoreNames.contains(n))db.createObjectStore(n,{keyPath:KEYS[n]});});}};
var dbp=null, listeners=[];
function open(){
  if(dbp) return dbp;
  dbp=new Promise(function(res,rej){
    if(!window.indexedDB) return rej(new Error('IndexedDB unavailable'));
    var r=indexedDB.open(DB_NAME,DB_VERSION);
    r.onupgradeneeded=function(e){for(var v=e.oldVersion+1;v<=DB_VERSION;v++){if(MIGRATIONS[v])MIGRATIONS[v](r.result,r.transaction);}};
    r.onsuccess=function(){res(r.result)};r.onerror=function(){rej(r.error)};
  });
  return dbp;
}
function req(store,mode,fn){
  return open().then(function(db){return new Promise(function(res,rej){
    var tx=db.transaction(store,mode),out;
    out=fn(tx.objectStore(store));
    tx.oncomplete=function(){res(out&&'result' in out?out.result:out)};
    tx.onerror=tx.onabort=function(){rej(tx.error)};
  });});
}
function notify(store){ if(store==='backups'||store==='undo'||store==='activeSession'&&false) return; listeners.forEach(function(f){f(store)}); }
var S=App.store={
  DATA_VERSION:DATA_VERSION, KEYS:KEYS, USER_STORES:USER_STORES,
  open:open,
  onWrite:function(f){listeners.push(f)},
  get:function(s,k){return req(s,'readonly',function(o){return o.get(k)})},
  all:function(s){return req(s,'readonly',function(o){return o.getAll()})},
  count:function(s){return req(s,'readonly',function(o){return o.count()})},
  put:function(s,v){return req(s,'readwrite',function(o){return o.put(v)}).then(function(r){notify(s);return r})},
  putMany:function(s,arr){if(!arr.length)return Promise.resolve();return req(s,'readwrite',function(o){arr.forEach(function(v){o.put(v)});return {}}).then(function(){notify(s)})},
  del:function(s,k){return req(s,'readwrite',function(o){return o.delete(k)}).then(function(r){notify(s);return r})},
  clear:function(s){return req(s,'readwrite',function(o){return o.clear()}).then(function(r){notify(s);return r})},
  getSetting:function(k,d){return S.get('settings',k).then(function(r){return r&&r.value!==undefined?r.value:d})},
  setSetting:function(k,v){return S.put('settings',{key:k,value:v})},
  migrateData:function(){ // data-version hook: runs on boot
    return S.getSetting('dataVersion',0).then(function(v){
      if(v<DATA_VERSION){ /* future: transform stored records from v to DATA_VERSION here */ return S.setSetting('dataVersion',DATA_VERSION); }
    });
  },
  exportAll:function(){
    var out={app:'cat-platform',dataVersion:DATA_VERSION,exportedAt:new Date().toISOString(),stores:{}};
    return Promise.all(USER_STORES.map(function(n){return S.all(n).then(function(r){out.stores[n]=r})})).then(function(){return out});
  },
  validateExport:function(o){
    var errs=[],counts={};
    if(!o||typeof o!=='object') return {ok:false,errors:['File is not a JSON object'],counts:counts};
    if(o.app!=='cat-platform') errs.push('Missing app marker "cat-platform" - not a backup from this app');
    if(typeof o.dataVersion!=='number') errs.push('Missing numeric dataVersion');
    else if(o.dataVersion>DATA_VERSION) errs.push('Data version '+o.dataVersion+' is newer than this app supports ('+DATA_VERSION+')');
    if(!o.stores||typeof o.stores!=='object') errs.push('Missing "stores" object');
    else Object.keys(o.stores).forEach(function(n){
      if(USER_STORES.indexOf(n)<0){errs.push('Unknown store "'+n+'"');return;}
      var a=o.stores[n];
      if(!Array.isArray(a)){errs.push('Store "'+n+'" is not an array');return;}
      var bad=a.filter(function(r){return !r||typeof r!=='object'||r[KEYS[n]]===undefined}).length;
      if(bad) errs.push('Store "'+n+'" has '+bad+' record(s) missing key "'+KEYS[n]+'"');
      counts[n]=a.length;
    });
    return {ok:!errs.length,errors:errs,counts:counts};
  },
  createBackup:function(reason){
    return S.exportAll().then(function(d){
      var b={id:'bk-'+Date.now()+'-'+Math.random().toString(36).slice(2,6),ts:Date.now(),reason:reason||'manual',data:d};
      return S.put('backups',b).then(function(){return S.all('backups')}).then(function(all){
        all.sort(function(a,b){return b.ts-a.ts});
        return Promise.all(all.slice(5).map(function(x){return S.del('backups',x.id)}));
      }).then(function(){return b});
    });
  },
  listBackups:function(){return S.all('backups').then(function(a){return a.sort(function(x,y){return y.ts-x.ts})})},
  importData:function(o,mode){
    var v=S.validateExport(o); if(!v.ok) return Promise.reject(new Error(v.errors.join('; ')));
    return S.createBackup('pre-import').then(function(){
      var names=Object.keys(o.stores);
      return names.reduce(function(p,n){return p.then(function(){
        return (mode==='replace'?S.clear(n):Promise.resolve()).then(function(){return S.putMany(n,o.stores[n])});
      })},Promise.resolve());
    }).then(function(){return S.migrateData()});
  },
  restoreBackup:function(id){
    return S.get('backups',id).then(function(b){
      if(!b) throw new Error('Backup not found');
      return S.createBackup('pre-restore').then(function(){return S.importData(b.data,'replace')});
    });
  },
  snapshot:function(names){var out={};return Promise.all(names.map(function(n){return S.all(n).then(function(r){out[n]=r})})).then(function(){return out});},
  restoreSnapshot:function(snap){return Object.keys(snap).reduce(function(p,n){return p.then(function(){return S.clear(n)}).then(function(){return S.putMany(n,snap[n])})},Promise.resolve())},
  pushUndo:function(label,snap){
    var u={id:'u-'+Date.now()+'-'+Math.random().toString(36).slice(2,5),ts:Date.now(),label:label,snap:snap};
    return S.put('undo',u).then(function(){return S.all('undo')}).then(function(all){
      all.sort(function(a,b){return b.ts-a.ts});return Promise.all(all.slice(10).map(function(x){return S.del('undo',x.id)}));
    }).then(function(){return u});
  },
  undo:function(id){return S.get('undo',id).then(function(u){if(!u)throw new Error('Nothing to undo');return S.restoreSnapshot(u.snap).then(function(){return S.del('undo',id)}).then(function(){return u})})}
};
})();
