const $=s=>document.querySelector(s);let current=null,result=null,selected=0,revision=0;const types=['人的不安全行为','物的不安全状态','环境不安全因素','管理缺陷'];const colors=['#3973d6','#e79127','#279e93','#8361d1'];const labels=['发生频率','严重程度','可检测性','可控制性','传导速度'];
function message(text=''){$('#message').textContent=text}
async function api(path,options={}){const r=await fetch(path,options);let data;try{data=await r.json()}catch{throw Error('服务器返回异常，请重试')}if(!r.ok)throw Error(typeof data.detail==='string'?data.detail:'输入格式不正确，请检查节点、关系和五维评分');return data}
const post=(path,data)=>api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
function el(tag,text,cls){const x=document.createElement(tag);if(text!==undefined)x.textContent=text;if(cls)x.className=cls;return x}
function tab(id){
  document.querySelectorAll('.tab').forEach(x=>x.hidden=x.id!==id);
  document.querySelectorAll('.nav').forEach(x=>{x.classList.toggle('active',x.dataset.tab===id);x.setAttribute('aria-current',x.dataset.tab===id?'page':'false')});
  const title={workspace:'事故风险图分析',import:'导入事故报告',metrics:'模型实验结果'}[id];
  $('#page-title').replaceChildren(document.createTextNode(title),el('span','.','title-dot'));
  $('#breadcrumb-current').textContent={workspace:'风险图分析',import:'导入事故报告',metrics:'模型实验结果'}[id];
  $('#page-description').textContent={workspace:'从事故叙事到风险线索，连接每一个关键要素。',import:'把非结构化叙事，转化为可查看、可复核的风险图。',metrics:'回到实验记录，理解模型表现与适用边界。'}[id];
}
document.querySelectorAll('.nav').forEach(x=>x.onclick=()=>tab(x.dataset.tab));
function svg(tag,attrs={}){const x=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v]of Object.entries(attrs))x.setAttribute(k,v);return x}
function draw(){
  const root=$('#graph');root.replaceChildren();if(!current)return;
  const order=[3,2,1,0],shortNames=['人的不安全行为','物的不安全状态','环境不安全因素','管理缺陷'];
  const palette=['#6289e4','#d4a15f','#5ba696','#a58bc9'];
  const groups=order.map(t=>current.nodes.map((n,i)=>({n,i})).filter(({n})=>n.type===types[t]));
  const maxRows=Math.max(...groups.map(g=>g.length),2);
  const height=Math.max(470,maxRows*112+130);root.setAttribute('viewBox',`0 0 1000 ${height}`);
  root.setAttribute('height',height);
  const defs=svg('defs');
  ['#bdcbe5','#7c9beb'].forEach((color,i)=>{const m=svg('marker',{id:`arrow${i}`,viewBox:'0 0 10 10',refX:8,refY:5,markerWidth:5,markerHeight:5,orient:'auto-start-reverse'});m.append(svg('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:color}));defs.append(m)});
  const filter=svg('filter',{id:'node-shadow',x:'-20%',y:'-30%',width:'140%',height:'170%'});filter.append(svg('feDropShadow',{dx:0,dy:4,stdDeviation:5,'flood-color':'#466da1','flood-opacity':'.055'}));defs.append(filter);root.append(defs);
  const positions=new Map();
  groups.forEach((group,column)=>{
    const x=50+column*240,t=order[column];
    const stripe=svg('rect',{x:x-10,y:20,width:200,height:height-40,rx:12,fill:'#f1f5fc','fill-opacity':.48});root.append(stripe);
    root.append(svg('circle',{cx:x+9,cy:47,r:4,fill:palette[t]}));
    const heading=svg('text',{x:x+22,y:52,fill:'#8393ae','font-size':15,'font-weight':500});heading.textContent=shortNames[t];root.append(heading);
    const count=svg('text',{x:x+175,y:52,fill:'#a6b3c9','font-size':12,'text-anchor':'end'});count.textContent=group.length;root.append(count);
    const offset=(maxRows-group.length)*56;
    group.forEach(({n,i},j)=>positions.set(n.id,{x,y:85+offset+j*112,i}));
  });
  const selectedId=current.nodes[selected].id;
  const sortedEdges=[...current.edges].sort((a,b)=>Number(a.source===selectedId||a.target===selectedId)-Number(b.source===selectedId||b.target===selectedId));
  for(const e of sortedEdges){
    const a=positions.get(e.source),b=positions.get(e.target);if(!a||!b)continue;
    const active=e.source===selectedId||e.target===selectedId;
    let d;
    if(a.x===b.x){const bend=55;const x=a.x+180;d=`M ${x} ${a.y+39} C ${x+bend} ${a.y+39}, ${x+bend} ${b.y+39}, ${x} ${b.y+39}`}
    else{const forward=b.x>a.x;const sx=a.x+(forward?180:0),tx=b.x+(forward?0:180),shift=(tx-sx)*.5;d=`M ${sx} ${a.y+39} C ${sx+shift} ${a.y+39}, ${tx-shift} ${b.y+39}, ${tx} ${b.y+39}`}
    const path=svg('path',{d,fill:'none',stroke:active?'#8fa9e6':'#cfdaed','stroke-width':active?1.8:1.2,opacity:active?.95:.5,'marker-end':`url(#arrow${active?1:0})`});
    const tip=svg('title');tip.textContent=`${e.source} → ${e.target}：${e.relation}（提取置信度 ${e.confidence}）`;path.append(tip);root.append(path);
  }
  current.nodes.forEach((node,i)=>{
    const {x,y}=positions.get(node.id),active=i===selected,c=palette[types.indexOf(node.type)]||palette[0];
    const g=svg('g',{class:'graph-node',tabindex:0,role:'button','aria-label':node.name,'aria-pressed':active});
    if(active)g.append(svg('rect',{x:x-4,y:y-4,width:188,height:86,rx:13,fill:'#dce7ff',opacity:.55}));
    g.append(svg('rect',{class:'node-card',x,y,width:180,height:78,rx:10,fill:active?'#f5f8ff':'#ffffff',stroke:active?'#7a9aed':'#e2e9f5','stroke-width':active?1.5:1,filter:'url(#node-shadow)'}));
    g.append(svg('circle',{cx:x+15,cy:y+17,r:3,fill:c}));
    const id=svg('text',{x:x+24,y:y+21,fill:active?'#6f8dcc':'#a1aec3','font-size':13,'font-weight':500});id.textContent=node.id;g.append(id);
    if(result){const score=svg('text',{x:x+165,y:y+21,'text-anchor':'end',fill:'#8b9fbd','font-size':12});score.textContent=result.nodes[i].score.toFixed(3);g.append(score)}
    const chars=Array.from(node.name);const rows=[chars.slice(0,9).join(''),chars.slice(9,18).join('')+(chars.length>18?'…':'')];
    rows.forEach((text,j)=>{const label=svg('text',{x:x+14,y:y+43+j*18,fill:active?'#42639b':'#647b9b','font-size':15,'font-weight':active?500:400});label.textContent=text;g.append(label)});
    const tip=svg('title');tip.textContent=node.name;g.append(tip);g.onclick=()=>select(i);g.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select(i);root.querySelectorAll('.graph-node')[i]?.focus({preventScroll:true})}};root.append(g);
  });
  $('#graph-count').textContent=`${current.nodes.length} 个节点 · ${current.edges.length} 条关系`;
}
function select(i){
  selected=i;const n=current.nodes[i];$('#node-id').textContent=n.id;$('#node-name').textContent=n.name;$('#node-type').textContent=n.type;
  $('#features').replaceChildren();
  n.features.forEach((v,j)=>{
    const row=el('div',undefined,'feature');const track=el('span',undefined,'feature-track');track.setAttribute('role','meter');track.setAttribute('aria-label',labels[j]);track.setAttribute('aria-valuemin','1');track.setAttribute('aria-valuemax','5');track.setAttribute('aria-valuenow',v);
    for(let k=1;k<=5;k++)track.append(el('i',undefined,`feature-segment ${k<=v?'filled':''}`));
    row.append(el('span',labels[j]),track,el('b',v));$('#features').append(row);
  });
  const r=result?.nodes[i];$('#node-score').textContent=r?r.score.toFixed(4):'尚未运行';$('#score-caption').textContent=r?'/ 1.0000':'';
  $('#node-label').textContent=r?(r.predicted_high?'候选高风险节点 · 待复核':'未筛为高风险 · 仍需复核'):'运行筛查后查看模型输出';
  document.querySelectorAll('.node-row').forEach((tr,j)=>tr.classList.toggle('selected',i===j));draw();
}
function table(){
  const body=$('#nodes');body.replaceChildren();
  current.nodes.forEach((n,i)=>{
    const r=result?.nodes[i];const tr=el('tr',undefined,`node-row ${i===selected?'selected':''}`);tr.onclick=()=>select(i);
    const name=el('td');const nameWrap=el('div',undefined,'node-name-cell');nameWrap.append(el('span',n.id,'table-node-id'),el('span',n.name));name.append(nameWrap);tr.append(name);
    const type=el('td');const typeWrap=el('span',undefined,'category-cell');typeWrap.append(el('i',undefined,['human','object','environment','management'][types.indexOf(n.type)]),el('span',n.type));type.append(typeWrap);tr.append(type);
    const score=el('td');const scoreWrap=el('div',undefined,'score-cell');scoreWrap.append(el('span',r?r.score.toFixed(4):'—'));if(r){const bar=el('progress');bar.max=1;bar.value=r.score;bar.setAttribute('aria-label','模型高风险类别分数');scoreWrap.append(bar)}score.append(scoreWrap);tr.append(score);
    const prediction=el('td');prediction.append(el('span',r?(r.predicted_high?'候选高风险':'未筛为高风险'):'未运行',`badge ${r?.predicted_high?'high':''}`));tr.append(prediction);
    tr.append(el('td',n.is_high_risk==null?'未提供':n.is_high_risk?'高风险':'低风险','plain-label'));
    tr.append(el('td',n.features[1]>=4||(n.features[1]>=3&&n.features[0]>=4)?'高风险':'低风险','plain-label'));
    body.append(tr);
  });
  $('#summary').textContent=result?`${result.model} · 筛出 ${result.nodes.filter(x=>x.predicted_high).length} / ${result.nodes.length} 个候选节点`:'选择模型并运行筛查后，在此查看结果';
  $('#result-count').textContent=current.nodes.length;$('#export').disabled=!result;
}
function showImported(){let op=$('#reports').querySelector('option[value="imported"]');if(!op){op=el('option','当前导入的报告');op.value='imported';$('#reports').append(op)}$('#reports').value='imported';op.disabled=true}
function setGraph(graph,text=''){revision++;current=graph;result=null;selected=0;$('#source-text').textContent=text||'此图未附带报告原文。';select(0);table();tab('workspace')}
async function run(button,fn){button.disabled=true;button.classList.add('busy');button.setAttribute('aria-busy','true');message();try{await fn()}catch(e){message(e.message)}finally{button.disabled=false;button.classList.remove('busy');button.removeAttribute('aria-busy')}}
$('#reports').onchange=()=>run($('#predict'),async()=>{const d=await api('/api/reports/'+$('#reports').value);setGraph(d.graph,d.text)});
$('#model').onchange=()=>{revision++;result=null;if(current){table();select(selected)}};
$('#predict').onclick=()=>run($('#predict'),async()=>{if(!current)throw Error('请先选择报告');const rev=revision;const data=await post('/api/predict',{graph:current,model:$('#model').value});if(rev!==revision)return;result=data;table();select(selected)});
$('#export').onclick=()=>{if(!result)return;const blob=new Blob([JSON.stringify({...result,source:current._metadata||{},exported_at:new Date().toISOString()},null,2)],{type:'application/json;charset=utf-8'});const url=URL.createObjectURL(blob);const a=el('a');a.href=url;a.download='safety-analysis.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
$('#report-file').onchange=()=>run($('#extract'),async()=>{const f=$('#report-file').files[0];if(!f)return;if(f.size>10*1024*1024)throw Error('文件不得超过 10 MB');const fd=new FormData();fd.append('file',f);const d=await api('/api/read-file',{method:'POST',body:fd});$('#report-text').value=d.text;updateTextCount();if(d.too_long)message('报告超过 12,000 字符，请自行选取相关段落后再提取；系统不会自动截断。')});
$('#extract').onclick=()=>run($('#extract'),async()=>{const text=$('#report-text').value.trim();if(text.length<50||text.length>12000)throw Error('正文长度需为 50–12,000 字符');message('正在提取风险图，通常需要几十秒……');const graph=await post('/api/extract',{text});setGraph(graph,text);showImported();message('风险图已提取。请选择模型运行筛查。')});
$('#json-file').onchange=()=>run($('#predict'),async()=>{const f=$('#json-file').files[0];if(!f)return;if(f.size>2*1024*1024)throw Error('JSON 不得超过 2 MB');let g;try{g=JSON.parse(await f.text())}catch{throw Error('JSON 格式错误')}g=g.graph||g;const d=await post('/api/predict',{graph:g,model:$('#model').value});setGraph(g);result=d;showImported();table();select(0);message('JSON 已导入，并完成模型筛查。')});
(async()=>{try{const [o,reports]=await Promise.all([api('/api/overview'),api('/api/reports')]);$('#s-nodes').textContent=o.nodes.toLocaleString();$('#s-edges').textContent=o.edges.toLocaleString();$('#connection').textContent='服务已连接';$('#connection').classList.add('connected');$('#llm-status').textContent=o.llm_configured?'大模型已配置':'尚未配置 API；可使用样本或导入 JSON';for(const m of o.models){const op=el('option',m);op.value=m;$('#model').append(op)}$('#model').value='HierarchicalGATv2';for(const r of reports){const title=/^[a-f0-9]{20,}\.pdf$/i.test(r.title)?`事故报告 ${String(r.id+1).padStart(3,'0')} · ${r.nodes} 个节点`:r.title.replace(/\.pdf$/i,'');const op=el('option',`${String(r.id+1).padStart(3,'0')}  ${title}`);op.title=r.title;op.value=r.id;$('#reports').append(op)}for(const r of o.metrics){const item=el('div',undefined,'metric-item'+(r.model==='HierarchicalGATv2'?' featured':''));const top=el('div',undefined,'metric-item-top');top.append(el('span',r.model),el('strong','F1 '+r.f1.toFixed(4)));const bar=el('progress');bar.max=1;bar.value=r.f1;bar.setAttribute('aria-label',r.model+' F1');item.append(top,bar);$('#metric-bars').append(item);const tr=el('tr');[r.model,...['accuracy','precision','recall','f1','auc'].map(k=>r[k].toFixed(4))].forEach(v=>tr.append(el('td',v)));$('#metrics-table').append(tr)}const d=await api('/api/reports/0');setGraph(d.graph,d.text)}catch(e){$('#connection').textContent='连接失败';message(e.message)}})();

function updateTextCount(){const count=$('#report-text').value.length;$('#text-count').textContent=count.toLocaleString()+' / 12,000 字符';$('#text-count').classList.toggle('over-limit',count>12000)}
$('#report-text').addEventListener('input',updateTextCount);
