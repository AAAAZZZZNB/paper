import { ACCIDENTS, RISK_PREDICTIONS, SHAP_SUMMARY, EVT_SUMMARY, KNOWLEDGE_GRAPH } from './data.mjs';

const PREDICTION_BY_CASE = new Map(RISK_PREDICTIONS.map(row => [row.case_id, row]));
const ACCIDENT_BY_CASE = new Map(ACCIDENTS.map(row => [row.case_id, row]));
const LEVEL_ORDER = { '低': 1, '中': 2, '高': 3, '极高': 4 };

const json = (body, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' } });
const normalize = (value) => String(value ?? '').trim().toLowerCase();
const contains = (text, ...terms) => terms.some(term => text.includes(String(term).toLowerCase()));
const maxLevel = (...levels) => levels.sort((a, b) => (LEVEL_ORDER[b] || 1) - (LEVEL_ORDER[a] || 1))[0] || '低';
const value = (x) => Number.isFinite(Number(x)) ? Number(x) : null;

function enrichAccident(row) {
  const result = { ...row };
  const pred = PREDICTION_BY_CASE.get(row.case_id);
  if (pred) {
    result.risk_prediction = {
      gold_level: pred.gold_level_name || '', kg_level: pred.B2_kg_level || '', fusion_level: pred.B5_tuned_fusion_level || '', llm_level: pred.L1_kg_wiki_level || '',
      basic_score: pred.basic_score || '', kg_score: pred.kg_score || '', graph_evidence_count: pred.L1_direct_graph_evidence_count || '', similar_case_count: pred.L1_similar_case_evidence_count || '',
      key_factors: pred.L1_key_factors || '', graph_evidence_preview: pred.L1_graph_evidence_preview || '', similar_case_preview: pred.L1_similar_case_evidence_preview || ''
    };
  }
  return result;
}

function factorEvidence(input) {
  const text = normalize([input.description, input.road_scene, input.weather, input.time_period, input.vehicle_type, input.event_type].join(' '));
  const rules = [
    [['夜间','深夜','凌晨'], '夜间/深夜时段', .298, 'SHAP'], [['货车','拖挂','重型'], '货车参与', .219, 'SHAP'], [['高速','高速公路'], '高速公路道路类型', .205, 'SHAP'],
    [['匝道','出口','入口'], '匝道道路场景', .245, 'SHAP'], [['早高峰','晚高峰','高峰'], '高峰时段', .156, 'SHAP'], [['暴雨','大雨','降雨','雨天'], '降雨天气', .180, 'SCENE'],
    [['长下坡','下坡'], '长下坡场景', .190, 'SCENE'], [['追尾','碰撞'], '追尾/碰撞场景', 1, 'CASE']
  ];
  return rules.filter(([terms]) => contains(text, ...terms)).map(([, factor, weight, source]) => ({ factor, weight, source }));
}

function similarCases(input, limit = 5) {
  const query = normalize([input.description, input.road_scene, input.weather, input.time_period, input.vehicle_type, input.event_type].join(' '));
  const tokens = ['夜间','深夜','凌晨','晴','暴雨','雨','主线','匝道','高速','货车','追尾','侧翻','火灾'].filter(t => query.includes(t));
  return ACCIDENTS.map(row => {
    const hay = normalize(['day_night','time_period','road_geometry','operation_scene','weather','event_type','participant_types','road_name','cause_tags'].map(k => row[k] || '').join(' '));
    const score = tokens.reduce((sum, token) => sum + (hay.includes(token) ? 1 : 0), 0);
    return { score, row };
  }).filter(x => x.score > 0).sort((a,b) => b.score - a.score || String(a.row.case_id).localeCompare(String(b.row.case_id))).slice(0, limit).map(({score,row}) => ({
    case_id: row.case_id, date: row.date, road: row.road_name, location: row.mileage, event_type: row.event_type, weather: row.weather, scene: row.road_geometry, casualty: row.casualty, match_score: score, disposal_tags: row.disposal_tags, evidence_excerpt: row.evidence_excerpt
  }));
}

function recommendations(level, thresholdHit) {
  const recs = level === '高' || level === '极高' ? [
    { priority:'P0', title:'现场防护与二次事故预警', action:'建议立即设置现场警示和安全防护区，联动现场处置人员核验。', source:'规则 + 事故案例' },
    { priority:'P1', title:'动态限速/情报板提示', action:'在风险路段上游发布速度控制和前方事件提示，必要时进行车道引导。', source:'风险知识图谱' },
    { priority:'P1', title:'巡检与清障调度', action:'根据道路场景和事故类型安排巡检、清障或救援资源，记录到事件工单。', source:'历史案例处置措施' }
  ] : level === '中' ? [
    { priority:'P2', title:'加强路段巡检', action:'在当前时段和道路场景下增加巡检频次，核验是否存在持续性风险。', source:'SHAP 场景解释' },
    { priority:'P2', title:'发布驾驶提示', action:'通过情报板或管理渠道提醒降低车速、保持车距并注意前方风险。', source:'案例知识库' }
  ] : [{ priority:'P3', title:'纳入常规观察', action:'保留事件记录，持续观察同一路段是否出现风险聚集或指标上升。', source:'风险监测策略' }];
  if (thresholdHit) recs.unshift({ priority:'P0', title:'核验冲突指标与现场状态', action:'冲突指标已越过论文示例高风险阈值，需优先确认数据质量和现场状态。', source:'EVT 阈值规则' });
  return recs;
}

function overview() {
  const countBy = (key) => Object.entries(ACCIDENTS.reduce((a,r) => { const k = r[key] || '未知'; a[k] = (a[k] || 0) + 1; return a; }, {}));
  const risk = countBy('risk_level').sort((a,b) => (LEVEL_ORDER[b[0]] || 0) - (LEVEL_ORDER[a[0]] || 0)).map(([name,value]) => ({name,value}));
  const events = countBy('event_type').sort((a,b) => b[1] - a[1]).slice(0,8).map(([name,value]) => ({name,value}));
  const roads = countBy('road_name').sort((a,b) => b[1] - a[1]).slice(0,6).map(([name,value]) => ({name,value}));
  const monthly = countBy('date').reduce((a,[k,v]) => { const m=k.slice(0,7); a[m]=(a[m]||0)+v; return a; }, {});
  return { dataset:{accident_cases:ACCIDENTS.length,risk_prediction_cases:RISK_PREDICTIONS.length,shap_factors:SHAP_SUMMARY.length,evt_rules:EVT_SUMMARY.length,graph_nodes:KNOWLEDGE_GRAPH.nodes.length,graph_edges:KNOWLEDGE_GRAPH.edges.length}, risk_distribution:risk, event_distribution:events, top_roads:roads, monthly_trend:Object.entries(monthly).sort().map(([name,value])=>({name,value})), recent_cases:ACCIDENTS.slice().sort((a,b)=>String(b.date).localeCompare(String(a.date))).slice(0,6).map(enrichAccident) };
}

function analyzeRisk(input) {
  const factors = factorEvidence(input);
  const tdtc = value(input.tdtc), mttc = value(input.mttc);
  const thresholdHits = [];
  if (tdtc !== null && tdtc < 1.5) thresholdHits.push({ indicator:'TDTC', value:tdtc, threshold:1.5, level:'高', source:'EVT' });
  if (mttc !== null && mttc < 2) thresholdHits.push({ indicator:'MTTC', value:mttc, threshold:2, level:'高', source:'EVT' });
  let base = factors.length >= 3 ? '中' : '低';
  if (contains(normalize(`${input.description || ''} ${input.event_type || ''}`), '追尾','侧翻','火灾') && factors.length >= 2) base = '中';
  const level = maxLevel(base, thresholdHits.length ? '高' : '低');
  const confidence = thresholdHits.length ? '高' : (factors.length ? '中' : '低');
  const cases = similarCases(input);
  const evidence = factors.map(x => ({ type:x.source, statement:`${x.factor}与事故风险相关，示例权重 ${x.weight}`, source:'论文 SHAP/案例知识示例' }));
  thresholdHits.forEach(x => evidence.push({ type:'EVT', statement:`${x.indicator}=${x.value}，低于高风险阈值 ${x.threshold}`, source:'论文冲突阈值规则' }));
  cases.slice(0,3).forEach(x => evidence.push({ type:'CASE', statement:`相似案例 ${x.case_id}：${x.event_type}，${x.scene}，${x.weather}`, source:'论文事故案例标签数据' }));
  return {
    summary:`当前事件被判定为${level}风险，置信度${confidence}。系统依据论文事故风险知识、冲突阈值规则和历史案例生成演示性研判。`, risk_level:level, confidence, risk_index:Math.min(.99, .25 + .12*factors.length + .35*thresholdHits.length).toFixed(2),
    risk_causes:factors.map(x=>x.factor).length ? factors.map(x=>x.factor) : ['当前输入信息不足，需补充现场信息'], threshold_results:thresholdHits.length ? thresholdHits : [{indicator:'未触发',value:'-',threshold:'-',level:'未触发'}], similar_cases:cases, recommendations:recommendations(level, thresholdHits.length),
    unknown_or_to_verify:['现场人员伤亡和责任认定需人工核验','当前系统未对图片像素进行自动事实识别'], evidence, generated_by:'知识增强 Agent 演示适配器（模板推理；可替换为本地/远程 LLM）', versions:{knowledge:'paper-seed-v0.1',rule:'evt-demo-v0.1',prompt:'risk-agent-demo-v0.1'}
  };
}

async function route(req) {
  const url = new URL(req.url);
  let path = url.pathname.replace(/^\/\.netlify\/functions\/api/, '').replace(/^\/api/, '') || '/';
  if (path === '/health') return json({ status:'ok', service:'traffic-risk-road-management', demo_data:'论文数据快照' });
  if (path === '/overview') return json(overview());
  if (path === '/shap/summary') return json({ title:'事故风险 SHAP 先验知识（论文示例）', disclaimer:'当前页面展示论文中已整理的解释结果示例；SHAP 重要性是模型关联解释，不等同于因果效应。', items:SHAP_SUMMARY });
  if (path === '/evt/summary') return json({ title:'轨迹冲突极值模型规则（论文示例）', disclaimer:'TDTC/MTTC 阈值为论文系统设计中的示例规则。正式发布前应绑定路段、时间范围、样本量和模型置信信息。', items:EVT_SUMMARY });
  if (path === '/knowledge/graph') return json(KNOWLEDGE_GRAPH);
  if (path === '/cases/search') return json({ items:similarCases({ description:url.searchParams.get('q') || '' }, Number(url.searchParams.get('limit') || 10)) });
  if (path === '/accidents') {
    const q=normalize(url.searchParams.get('search') || ''), level=url.searchParams.get('risk_level') || '', limit=Math.min(200,Number(url.searchParams.get('limit')||20)), offset=Math.max(0,Number(url.searchParams.get('offset')||0));
    const items=ACCIDENTS.filter(row => (!q || normalize(Object.values(row).join(' ')).includes(q)) && (!level || row.risk_level===level)).map(enrichAccident);
    return json({ total:items.length, items:items.slice(offset,offset+limit) });
  }
  const detail = path.match(/^\/accidents\/([^/]+)$/); if (detail) { const row=ACCIDENT_BY_CASE.get(decodeURIComponent(detail[1])); return row ? json(enrichAccident(row)) : json({error:'未找到该论文案例'},404); }
  if (path === '/risk/analyze' && req.method === 'POST') { return json(analyzeRisk(await req.json())); }
  return json({ error:'Not Found', path },404);
}

export default async (req) => { try { return await route(req); } catch (error) { return json({ error:error.message || 'Server error' },500); } };
