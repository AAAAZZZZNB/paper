from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
FRONTEND_DIR = BASE_DIR / "app" / "frontend"


def load_json(name: str, default: Any) -> Any:
    path = DATA_DIR / name
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


ACCIDENTS: list[dict[str, Any]] = load_json("accidents.json", [])
RISK_PREDICTIONS: list[dict[str, Any]] = load_json("risk_predictions.json", [])
SHAP_SUMMARY: list[dict[str, Any]] = load_json("shap_summary.json", [])
EVT_SUMMARY: list[dict[str, Any]] = load_json("evt_summary.json", [])
GRAPH: dict[str, Any] = load_json("knowledge_graph.json", {"nodes": [], "edges": []})

PREDICTION_BY_CASE = {row.get("case_id"): row for row in RISK_PREDICTIONS}
ACCIDENT_BY_CASE = {row.get("case_id"): row for row in ACCIDENTS}
LEVEL_ORDER = {"低": 1, "中": 2, "高": 3, "极高": 4}

app = FastAPI(
    title="交通风险智能道路管理系统",
    version="0.1.0",
    description="基于论文事故 SHAP、轨迹 EVT、风险知识图谱和知识增强 Agent 的演示系统",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RiskRequest(BaseModel):
    description: str = Field(default="", description="事件描述")
    road_scene: str = Field(default="高速公路主线", description="道路场景")
    weather: str = Field(default="晴", description="天气")
    time_period: str = Field(default="日间", description="时段")
    vehicle_type: str = Field(default="小客车", description="车辆类型")
    event_type: str = Field(default="追尾/碰撞", description="事故类型")
    tdtc: Optional[float] = Field(default=None, description="TDTC 秒")
    mttc: Optional[float] = Field(default=None, description="MTTC 秒")


def normalize(text: Any) -> str:
    return str(text or "").strip().lower()


def contains(text: str, *terms: str) -> bool:
    return any(term.lower() in text for term in terms)


def max_level(*levels: str) -> str:
    return max(levels or ["低"], key=lambda x: LEVEL_ORDER.get(x, 1))


def enrich_accident(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    pred = PREDICTION_BY_CASE.get(row.get("case_id"))
    if pred:
        result["risk_prediction"] = {
            "gold_level": pred.get("gold_level_name", ""),
            "kg_level": pred.get("B2_kg_level", ""),
            "fusion_level": pred.get("B5_tuned_fusion_level", ""),
            "llm_level": pred.get("L1_kg_wiki_level", ""),
            "basic_score": pred.get("basic_score", ""),
            "kg_score": pred.get("kg_score", ""),
            "graph_evidence_count": pred.get("L1_direct_graph_evidence_count", ""),
            "similar_case_count": pred.get("L1_similar_case_evidence_count", ""),
            "key_factors": pred.get("L1_key_factors", ""),
            "graph_evidence_preview": pred.get("L1_graph_evidence_preview", ""),
            "similar_case_preview": pred.get("L1_similar_case_evidence_preview", ""),
        }
    return result


def factor_evidence(req: RiskRequest) -> list[dict[str, Any]]:
    text = " ".join([
        req.description, req.road_scene, req.weather,
        req.time_period, req.vehicle_type, req.event_type,
    ])
    text = normalize(text)
    evidence: list[dict[str, Any]] = []
    rules = [
        (("夜间", "深夜", "凌晨"), "夜间/深夜时段", 0.298, "SHAP"),
        (("货车", "拖挂", "重型"), "货车参与", 0.219, "SHAP"),
        (("高速", "高速公路"), "高速公路道路类型", 0.205, "SHAP"),
        (("匝道", "出口", "入口"), "匝道道路场景", 0.245, "SHAP"),
        (("早高峰", "晚高峰", "高峰"), "高峰时段", 0.156, "SHAP"),
        (("暴雨", "大雨", "降雨", "雨天"), "降雨天气", 0.180, "SCENE"),
        (("长下坡", "下坡"), "长下坡场景", 0.190, "SCENE"),
        (("追尾", "碰撞"), "追尾/碰撞场景", 1.000, "CASE"),
    ]
    for terms, label, weight, source in rules:
        if contains(text, *terms):
            evidence.append({"factor": label, "weight": weight, "source": source})
    return evidence


def similar_cases(req: RiskRequest, limit: int = 5) -> list[dict[str, Any]]:
    query = normalize(" ".join([
        req.description, req.road_scene, req.weather,
        req.time_period, req.vehicle_type, req.event_type,
    ]))
    tokens = [t for t in ["夜间", "深夜", "凌晨", "晴", "暴雨", "雨", "主线", "匝道", "高速", "货车", "追尾", "侧翻", "火灾"] if t in query]
    scored = []
    for row in ACCIDENTS:
        hay = normalize(" ".join(row.get(k, "") for k in [
            "day_night", "time_period", "road_geometry", "operation_scene",
            "weather", "event_type", "participant_types", "road_name", "cause_tags",
        ]))
        score = sum(1 for t in tokens if t in hay)
        if score:
            scored.append((score, row))
    scored.sort(key=lambda x: (-x[0], x[1].get("case_id", "")))
    result = []
    for score, row in scored[:limit]:
        result.append({
            "case_id": row.get("case_id"),
            "date": row.get("date"),
            "road": row.get("road_name"),
            "location": row.get("mileage"),
            "event_type": row.get("event_type"),
            "weather": row.get("weather"),
            "scene": row.get("road_geometry"),
            "casualty": row.get("casualty"),
            "match_score": score,
            "disposal_tags": row.get("disposal_tags"),
            "evidence_excerpt": row.get("evidence_excerpt"),
        })
    return result


def recommendation(level: str, req: RiskRequest, threshold_hit: bool) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    if level in {"高", "极高"}:
        recs.extend([
            {"priority": "P0", "title": "现场防护与二次事故预警", "action": "建议立即设置现场警示和安全防护区，联动现场处置人员核验。", "source": "规则 + 事故案例"},
            {"priority": "P1", "title": "动态限速/情报板提示", "action": "在风险路段上游发布速度控制和前方事件提示，必要时进行车道引导。", "source": "风险知识图谱"},
            {"priority": "P1", "title": "巡检与清障调度", "action": "根据道路场景和事故类型安排巡检、清障或救援资源，记录到事件工单。", "source": "历史案例处置措施"},
        ])
    elif level == "中":
        recs.extend([
            {"priority": "P2", "title": "加强路段巡检", "action": "在当前时段和道路场景下增加巡检频次，核验是否存在持续性风险。", "source": "SHAP 场景解释"},
            {"priority": "P2", "title": "发布驾驶提示", "action": "通过情报板或管理渠道提醒降低车速、保持车距并注意前方风险。", "source": "案例知识库"},
        ])
    else:
        recs.append({"priority": "P3", "title": "纳入常规观察", "action": "保留事件记录，持续观察同一路段是否出现风险聚集或指标上升。", "source": "风险监测策略"})
    if threshold_hit:
        recs.insert(0, {"priority": "P0", "title": "核验冲突指标与现场状态", "action": "冲突指标已越过论文示例高风险阈值，需优先确认数据质量和现场状态。", "source": "EVT 阈值规则"})
    return recs


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "traffic-risk-road-management", "demo_data": "论文数据快照"}


@app.get("/api/overview")
def overview():
    risk_counter = Counter(row.get("risk_level") or "未知" for row in ACCIDENTS)
    event_counter = Counter(row.get("event_type") or "未知" for row in ACCIDENTS)
    road_counter = Counter(row.get("road_name") or "未知" for row in ACCIDENTS)
    month_counter = Counter((row.get("date") or "")[:7] for row in ACCIDENTS if row.get("date"))
    return {
        "dataset": {
            "accident_cases": len(ACCIDENTS),
            "risk_prediction_cases": len(RISK_PREDICTIONS),
            "shap_factors": len(SHAP_SUMMARY),
            "evt_rules": len(EVT_SUMMARY),
            "graph_nodes": len(GRAPH.get("nodes", [])),
            "graph_edges": len(GRAPH.get("edges", [])),
        },
        "risk_distribution": [{"name": k, "value": v} for k, v in sorted(risk_counter.items(), key=lambda x: LEVEL_ORDER.get(x[0], 0), reverse=True)],
        "event_distribution": [{"name": k, "value": v} for k, v in event_counter.most_common(8)],
        "top_roads": [{"name": k, "value": v} for k, v in road_counter.most_common(6)],
        "monthly_trend": [{"name": k, "value": v} for k, v in sorted(month_counter.items())],
        "recent_cases": [enrich_accident(row) for row in sorted(ACCIDENTS, key=lambda x: x.get("date", ""), reverse=True)[:6]],
    }


@app.get("/api/accidents")
def accidents(
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    search: str = Query(default=""),
    risk_level: str = Query(default=""),
):
    query = normalize(search)
    filtered = []
    for row in ACCIDENTS:
        hay = normalize(" ".join(row.values()))
        if query and query not in hay:
            continue
        if risk_level and row.get("risk_level") != risk_level:
            continue
        filtered.append(enrich_accident(row))
    return {"total": len(filtered), "items": filtered[offset:offset + limit]}


@app.get("/api/accidents/{case_id}")
def accident_detail(case_id: str):
    row = ACCIDENT_BY_CASE.get(case_id)
    if not row:
        raise HTTPException(status_code=404, detail="未找到该论文案例")
    return enrich_accident(row)


@app.get("/api/shap/summary")
def shap_summary():
    return {
        "title": "事故风险 SHAP 先验知识（论文示例）",
        "disclaimer": "当前页面展示论文中已整理的解释结果示例；SHAP 重要性是模型关联解释，不等同于因果效应。",
        "items": SHAP_SUMMARY,
    }


@app.get("/api/evt/summary")
def evt_summary():
    return {
        "title": "轨迹冲突极值模型规则（论文示例）",
        "disclaimer": "TDTC/MTTC 阈值为论文系统设计中的示例规则。正式发布前应绑定路段、时间范围、样本量和模型置信信息。",
        "items": EVT_SUMMARY,
    }


@app.get("/api/knowledge/graph")
def knowledge_graph():
    return GRAPH


@app.get("/api/cases/search")
def cases_search(q: str = Query(default=""), limit: int = Query(default=10, ge=1, le=50)):
    req = RiskRequest(description=q)
    return {"items": similar_cases(req, limit=limit)}


@app.post("/api/risk/analyze")
def analyze_risk(req: RiskRequest):
    factors = factor_evidence(req)
    threshold_hits = []
    if req.tdtc is not None and req.tdtc < 1.5:
        threshold_hits.append({"indicator": "TDTC", "value": req.tdtc, "threshold": 1.5, "level": "高", "source": "EVT"})
    if req.mttc is not None and req.mttc < 2.0:
        threshold_hits.append({"indicator": "MTTC", "value": req.mttc, "threshold": 2.0, "level": "高", "source": "EVT"})

    base_level = "低"
    if len(factors) >= 3:
        base_level = "中"
    if contains(normalize(req.description + req.event_type), "追尾", "侧翻", "火灾") and len(factors) >= 2:
        base_level = "中"
    final_level = max_level(base_level, "高" if threshold_hits else "低")
    confidence = "高" if threshold_hits else ("中" if factors else "低")
    cases = similar_cases(req)
    recs = recommendation(final_level, req, bool(threshold_hits))

    evidence = []
    for item in factors:
        evidence.append({
            "type": item["source"],
            "statement": f"{item['factor']}与事故风险相关，示例权重 {item['weight']}",
            "source": "论文 SHAP/案例知识示例",
        })
    for item in threshold_hits:
        evidence.append({
            "type": "EVT",
            "statement": f"{item['indicator']}={item['value']}，低于高风险阈值 {item['threshold']}",
            "source": "论文冲突阈值规则",
        })
    for case in cases[:3]:
        evidence.append({
            "type": "CASE",
            "statement": f"相似案例 {case['case_id']}：{case.get('event_type','')}，{case.get('scene','')}，{case.get('weather','')}",
            "source": "论文事故案例标签数据",
        })

    report = {
        "summary": f"当前事件被判定为{final_level}风险，置信度{confidence}。系统依据论文事故风险知识、冲突阈值规则和历史案例生成演示性研判。",
        "risk_level": final_level,
        "confidence": confidence,
        "risk_index": round(min(0.99, 0.25 + 0.12 * len(factors) + 0.35 * len(threshold_hits)), 2),
        "risk_causes": [x["factor"] for x in factors] or ["当前输入信息不足，需补充现场信息"],
        "threshold_results": threshold_hits or [{"indicator": "未触发", "value": "-", "threshold": "-", "level": "未触发"}],
        "similar_cases": cases,
        "recommendations": recs,
        "unknown_or_to_verify": ["现场人员伤亡和责任认定需人工核验", "当前系统未对图片像素进行自动事实识别"],
        "evidence": evidence,
        "generated_by": "知识增强 Agent 演示适配器（模板推理；可替换为本地/远程 LLM）",
        "versions": {"knowledge": "paper-seed-v0.1", "rule": "evt-demo-v0.1", "prompt": "risk-agent-demo-v0.1"},
    }
    return report


# Static frontend is mounted last so /api routes keep priority.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
