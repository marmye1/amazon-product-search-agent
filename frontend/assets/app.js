const state = {
  traces: [],
  stages: [],
  selectedStageId: null,
  running: false,
  activeView: "chat",
  lastRequest: null,
  lastResponse: null,
  traceId: null,
  sessionId: (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function")
    ? globalThis.crypto.randomUUID()
    : `ui-session-${Date.now().toString(36)}`,
};

const $ = (selector) => document.querySelector(selector);
const form = $("#query-form");
const input = $("#query-input");
const runButton = $("#run-button");
const chatLog = $("#chat-log");
const flowTrack = $("#flow-track");
const chatView = $("#chat-view");
const traceView = $("#trace-view");

const NODE_PHASES = {
  merge_context: "话题判断 / 上下文合并",
  parse_query: "需求理解",
  validate_fields: "字段校验",
  decide_next: "路径路由",
  ask_clarification: "交互分支",
  build_search_request: "检索准备",
  search_products: "工具调用",
  prepare_candidates: "候选处理",
  rewrite_query: "有限查询改写",
  answer_rag: "回答生成",
  validate_evidence: "证据校验",
  validate: "回答校验",
  reject_pipeline: "安全拒答",
  reject: "安全拒答",
  finalize: "结果汇总",
  category_fallback: "相关类别提示",
  memory_update: "记忆更新",
};

function setHealth(label, className) {
  const element = $("#health-status");
  element.className = `health-status ${className || ""}`;
  element.innerHTML = `<span class="status-dot"></span>${label}`;
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    setHealth(data.status === "ok" ? "本地服务在线" : "依赖服务降级", data.status === "ok" ? "ready" : "degraded");
  } catch (error) {
    setHealth("服务未连接", "degraded");
  }
}

function switchView(view) {
  state.activeView = view;
  const isChat = view === "chat";
  chatView.hidden = !isChat;
  traceView.hidden = isChat;
  chatView.classList.toggle("is-hidden", !isChat);
  traceView.classList.toggle("is-hidden", isChat);
  document.querySelectorAll(".side-tab").forEach((tab) => {
    const active = tab.dataset.view === view;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", String(active));
  });
}

function setSidebarRun(status, detail) {
  $("#sidebar-run-status").textContent = status;
  $("#sidebar-run-count").textContent = detail;
}

function addChatEntry(role, text, meta, recommendations = [], error = false, relatedCategories = []) {
  const entry = document.createElement("div");
  entry.className = `chat-entry ${role}`;

  const metaElement = document.createElement("div");
  metaElement.className = "chat-meta";
  metaElement.textContent = meta;
  entry.appendChild(metaElement);

  const bubble = document.createElement("div");
  bubble.className = `bubble${error ? " error-bubble" : ""}`;
  bubble.textContent = text;
  entry.appendChild(bubble);

  if (recommendations.length) {
    const list = document.createElement("div");
    list.className = "recommendation-list";
    recommendations.forEach((item) => {
      const card = document.createElement("div");
      card.className = "recommendation-item";
      const title = document.createElement("strong");
      title.textContent = `${item.title} · ${item.product_id}`;
      const reason = document.createElement("span");
      reason.textContent = item.reason;
      card.append(title, reason);
      list.appendChild(card);
    });
    entry.appendChild(list);
  }

  if (relatedCategories.length) {
    const categoryBox = document.createElement("div");
    categoryBox.className = "related-category-box";
    const categoryLabel = document.createElement("span");
    categoryLabel.className = "related-category-label";
    categoryLabel.textContent = "相关商品类别";
    const categoryList = document.createElement("div");
    categoryList.className = "related-category-list";
    relatedCategories.forEach((category) => {
      const chip = document.createElement("span");
      chip.className = "related-category-chip";
      chip.textContent = category;
      categoryList.appendChild(chip);
    });
    categoryBox.append(categoryLabel, categoryList);
    entry.appendChild(categoryBox);
  }

  if (role === "agent" && !error && state.stages.length) {
    const traceButton = document.createElement("button");
    traceButton.type = "button";
    traceButton.className = "view-trace-button";
    traceButton.textContent = "查看本次完整流程 ↗";
    traceButton.addEventListener("click", () => switchView("trace"));
    entry.appendChild(traceButton);
  }

  chatLog.appendChild(entry);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function clearWelcome() {
  const welcome = chatLog.querySelector(".welcome-card");
  if (welcome) welcome.remove();
}

function renderJson(element, value) {
  element.textContent = JSON.stringify(value ?? {}, null, 2);
}

function formatDuration(duration) {
  return `${Number(duration || 0).toLocaleString()} ms`;
}

function observableConclusion(stage) {
  if (stage.kind !== "module") return stage.summary;
  const output = stage.output || {};
  const name = stage.node_name;
  if (name === "merge_context") {
    return `${output.context_status || "未使用"} · ${output.effective_query || "—"}`;
  }
  if (name === "parse_query") {
    const terms = output.parsed_constraints?.search_terms || [];
    return `提取 ${terms.length} 个检索词 · 可检索=${String(output.parsed_constraints?.retrieval_eligible ?? "—")}`;
  }
  if (name === "validate_fields") {
    return `字段校验：${output.field_validation_status || "—"} · 可检索=${String(output.parsed_constraints?.retrieval_eligible ?? "—")}`;
  }
  if (name === "decide_next") return `路由结果：${output.next_action || "—"}`;
  if (name === "build_search_request") return `生成 ${output.search_request?.top_k ?? "—"} 个候选的检索请求`;
  if (name === "search_products") {
    const response = output.search_response || {};
    return `召回 ${response.results?.length ?? 0} 条 · ${response.retrieval_method || "—"}`;
  }
  if (name === "prepare_candidates") {
    const response = output.search_response || {};
    return `硬约束后保留 ${response.results?.length ?? 0} 条候选`;
  }
  if (name === "rewrite_query") return `下一步：${output.pipeline_next_action || "—"} · attempt=${output.attempt_count ?? "—"}`;
  if (name === "answer_rag") return `生成结构化回答 · errors=${(output.errors || []).length}`;
  if (name === "validate_evidence" || name === "validate") return `grounded=${String(output.evidence_check?.grounded ?? output.validation_report?.grounded ?? "—")}`;
  if (name === "category_fallback") return `相关类别：${(output.related_categories || []).join("、") || "无"}`;
  if (name === "finalize") return `出口：${output.next_action || "—"} · 版本=${output.agent_version || "—"}`;
  return `输出 ${stage.output_keys?.length || 0} 个字段`;
}

function makeBoundaryStage(id, sequenceLabel, displayName, nodeName, input, output, inputDescription, outputDescription, summary) {
  return {
    id,
    sequence: sequenceLabel,
    sequenceLabel,
    kind: "boundary",
    kindLabel: "API / SERVICE",
    phase: "系统边界",
    display_name: displayName,
    node_name: nodeName,
    input_description: inputDescription,
    output_description: outputDescription,
    input_format: "HTTP JSON",
    output_format: "JSON",
    input_keys: Object.keys(input || {}),
    output_keys: Object.keys(output || {}),
    input: input || {},
    output: output || {},
    duration_ms: 0,
    status: "completed",
    summary,
  };
}

function buildStages(traces, response, request) {
  if (!request || !response) return [];
  const requestBody = {
    message: request.message,
    session_id: request.session_id,
    locale: request.locale,
    top_k: request.top_k,
  };
  const chatRequest = { ...requestBody };
  const agentState = {
    user_query: request.message,
    errors: [],
    top_k: request.top_k,
    max_products: request.top_k,
    trace_id: state.traceId,
    trace_nodes: [],
    request_locale: request.locale,
  };
  const stages = [
    makeBoundaryStage("client-request", "A", "客户端请求", "POST /v1/chat", { user_intent: request.message }, requestBody, "用户问题", "HTTP JSON 请求体", "浏览器发送 message、session_id、locale、top_k"),
    makeBoundaryStage("api-validation", "B", "FastAPI 参数校验", "ChatRequest", requestBody, chatRequest, "HTTP JSON 请求体", "经过 Pydantic 校验的 ChatRequest", "校验 message 非空、top_k 范围和字段类型"),
    makeBoundaryStage("agent-service", "C", "AgentService 初始化", "AgentState", chatRequest, agentState, "ChatRequest", "LangGraph 初始 AgentState", "建立本次请求状态；按 session_id 读取临时上下文，不写入数据库"),
  ];

  traces.forEach((trace, index) => {
    stages.push({
      ...trace,
      id: `module-${trace.sequence}-${index}`,
      sequenceLabel: String(stages.length + 1).padStart(2, "0"),
      kind: "module",
      kindLabel: "LANGGRAPH MODULE",
      phase: NODE_PHASES[trace.node_name] || "Agent 节点",
      summary: "",
    });
  });

  const finalTrace = traces[traces.length - 1] || {};
  const responseBody = {
    answer: response.answer,
    recommendations: response.recommendations || [],
    evidence: response.evidence || [],
    limitations: response.limitations || [],
    related_categories: response.related_categories || [],
    trace_id: response.trace_id,
    agent_version: response.agent_version,
    retrieval_method: response.retrieval_method,
    execution_trace: response.execution_trace || [],
  };
  stages.push(makeBoundaryStage("api-response", "Z", "HTTP 响应序列化", "ChatResponse", {
    final_response: finalTrace.output?.final_response || response.answer,
    executed_modules: traces.length,
    trace_id: response.trace_id,
  }, responseBody, "最终业务结果和 execution_trace", "返回给聊天界面的 ChatResponse", "前端收到回答，同时保留本次所有执行快照"));
  stages.forEach((stage) => {
    if (!stage.summary) stage.summary = observableConclusion(stage);
  });
  return stages;
}

function selectStage(stage) {
  state.selectedStageId = stage.id;
  document.querySelectorAll(".flow-stage-card").forEach((card) => {
    card.classList.toggle("selected", card.dataset.stageId === stage.id);
  });

  $("#selected-sequence").textContent = stage.sequenceLabel;
  $("#selected-name").textContent = stage.display_name;
  $("#selected-description").textContent = `${stage.node_name} · ${stage.input_description} → ${stage.output_description}`;
  $("#selected-meta").textContent = `${stage.input_format} → ${stage.output_format} · ${formatDuration(stage.duration_ms)}`;
  $("#input-contract").textContent = `${stage.input_format} · ${stage.input_description}`;
  $("#output-contract").textContent = `${stage.output_format} · ${stage.output_description}`;
  renderJson($("#input-json"), stage.input);
  renderJson($("#output-json"), stage.output);
}

function edgeLabel(previous, next) {
  if (previous.id === "client-request") return "HTTP JSON";
  if (previous.id === "api-validation") return "ChatRequest";
  if (previous.id === "agent-service") return "AgentState";
  const action = previous.output?.next_action || previous.output?.pipeline_next_action;
  if (action) return String(action);
  const keys = previous.output_keys || [];
  if (keys.includes("search_response")) return "search_response";
  if (keys.includes("answer")) return "answer";
  if (keys.includes("parsed_constraints")) return "constraints";
  if (keys.includes("final_response")) return "final_response";
  return keys[0] || "state";
}

function renderFlow(traces, response = state.lastResponse) {
  state.traces = traces || [];
  state.stages = buildStages(state.traces, response, state.lastRequest);
  flowTrack.innerHTML = "";

  if (!state.stages.length) {
    flowTrack.innerHTML = `<div class="flow-empty"><span class="empty-line"></span><span>输入问题后，这里会显示真实执行顺序</span><span class="empty-line"></span></div>`;
    return;
  }

  state.stages.forEach((stage, index) => {
    if (index > 0) {
      const arrow = document.createElement("span");
      arrow.className = "flow-arrow detailed-arrow";
      arrow.setAttribute("aria-hidden", "true");
      arrow.innerHTML = `<span>${edgeLabel(state.stages[index - 1], stage)}</span>`;
      flowTrack.appendChild(arrow);
    }

    const card = document.createElement("button");
    card.type = "button";
    card.className = `flow-stage-card node-card ${stage.kind === "boundary" ? "boundary-node" : "module-node"} ${stage.status === "degraded" ? "degraded" : ""}`;
    card.dataset.stageId = stage.id;
    card.style.animationDelay = `${index * 35}ms`;
    card.innerHTML = `
      <span class="node-top"><span class="node-seq">${stage.sequenceLabel}</span><span class="node-kind">${stage.kindLabel}</span><i class="node-status"></i></span>
      <h3>${stage.display_name}</h3>
      <span class="node-phase">${stage.phase}</span>
      <span class="node-code">${stage.node_name}</span>
      <span class="node-summary"></span>
      <span class="node-duration">${formatDuration(stage.duration_ms)} · ${stage.status}</span>
    `;
    card.querySelector(".node-summary").textContent = stage.summary;
    card.addEventListener("click", () => selectStage(stage));
    flowTrack.appendChild(card);
  });

  const defaultStage = state.stages.find((stage) => stage.kind === "module" && stage.node_name === "finalize")
    || state.stages.find((stage) => stage.kind === "module")
    || state.stages[state.stages.length - 1];
  selectStage(defaultStage);
}

function setLoading(loading) {
  state.running = loading;
  runButton.disabled = loading;
  runButton.querySelector("span:last-child").textContent = loading ? "Running…" : "Run agent";
  setSidebarRun(loading ? "运行中" : "未运行", loading ? "等待模块快照返回" : "等待一次 Agent 请求");
}

async function runAgent(message) {
  clearWelcome();
  addChatEntry("user", message, "YOU · REQUEST");
  state.lastRequest = { message, session_id: state.sessionId, locale: "us", top_k: 3 };
  state.lastResponse = null;
  setLoading(true);
  renderFlow([]);
  switchView("chat");
  $("#selected-name").textContent = "运行中…";
  $("#selected-description").textContent = "等待 LangGraph 返回本次执行的模块快照。";

  try {
    const traceId = `ui-${Date.now().toString(36)}`;
    state.traceId = traceId;
    const response = await fetch("/v1/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Trace-ID": traceId },
      body: JSON.stringify(state.lastRequest),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data?.error?.message || "Agent 请求失败");

    state.lastResponse = data;
    renderFlow(data.execution_trace || [], data);
    addChatEntry("agent", data.answer, `AGENT ${data.agent_version} · ${data.retrieval_method}`, data.recommendations || [], false, data.related_categories || []);
    setSidebarRun("已完成", `${state.traces.length} 个 Agent 模块 · ${state.stages.length} 个可观测步骤`);
  } catch (error) {
    addChatEntry("agent", error.message || "无法连接 Agent 服务。", "RUN ERROR", [], true);
    renderFlow([]);
    setSidebarRun("失败", "请检查本地 API、OpenSearch 和 LM Studio");
  } finally {
    state.running = false;
    runButton.disabled = false;
    runButton.querySelector("span:last-child").textContent = "Run agent";
    checkHealth();
  }
}

document.querySelectorAll(".side-tab").forEach((tab) => {
  tab.addEventListener("click", () => switchView(tab.dataset.view));
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message || state.running) return;
  input.value = "";
  runAgent(message);
});

input.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") form.requestSubmit();
});

document.querySelectorAll(".example-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    input.value = chip.dataset.example || "";
    input.focus();
  });
});

document.querySelectorAll(".copy-button").forEach((button) => {
  button.addEventListener("click", async () => {
    const target = document.getElementById(button.dataset.copyTarget);
    if (!target) return;
    try {
      await navigator.clipboard.writeText(target.textContent);
      const original = button.textContent;
      button.textContent = "已复制";
      setTimeout(() => { button.textContent = original; }, 1100);
    } catch (error) {
      button.textContent = "复制失败";
    }
  });
});

switchView("chat");
checkHealth();
