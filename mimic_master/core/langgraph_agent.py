"""LangGraph-based DM Agent.

利用 LangGraph 的状态流、Tool 调用和 Checkpointer 机制，
集成项目的 RAG 系统和 StateMemory。
"""

import os
from typing import Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool

from mimic_master.memory.state_memory import StateMemory, get_state_memory
from mimic_master.memory.knowledge_retriever import (
    HybridKnowledgeRetriever,
    get_hybrid_knowledge_retriever,
)
from mimic_master.memory.episodic_retriever import (
    EpisodicRetriever,
    get_episodic_retriever,
)
from mimic_master.memory.intent_classifier import (
    IntentClassifier,
    get_intent_classifier,
)
from mimic_master.models.memory import GameState

# ============== Agent State ==============


class AgentState(TypedDict):
    """LangGraph Agent 状态类型定义."""

    messages: list
    game_state: Optional[GameState]
    intent: Optional[str]
    knowledge_context: Optional[str]
    episodic_context: Optional[str]
    session_id: Optional[str]
    should_retrieve_knowledge: bool
    should_retrieve_episodes: bool
    response: Optional[str]


# ============== RAG Tools ==============


class RAGTools:
    """RAG 工具集合，封装为 LangGraph 可调用的 Tool."""

    def __init__(
        self,
        knowledge_retriever: Optional[HybridKnowledgeRetriever] = None,
        episodic_retriever: Optional[EpisodicRetriever] = None,
        intent_classifier: Optional[IntentClassifier] = None,
    ) -> None:
        self._knowledge_retriever = (
            knowledge_retriever or get_hybrid_knowledge_retriever()
        )
        self._episodic_retriever = episodic_retriever or get_episodic_retriever()
        self._intent_classifier = intent_classifier or get_intent_classifier()

    @tool(
        "retrieve_rules",
        description="""检索 D&D 5E 规则知识库。
    当用户询问规则、法术、怪物、种族、职业等问题时使用此工具。
    返回相关的规则条目和来源。""",
    )
    async def retrieve_rules(self, query: str) -> str:
        """检索 D&D 规则知识库."""
        try:
            results = await self._knowledge_retriever.retrieve(query, top_k=5)
            if not results:
                return "未找到相关规则。"

            lines = ["## 检索到的规则\n"]
            for i, result in enumerate(results, 1):
                content = (
                    result.content[:800] + "..."
                    if len(result.content) > 800
                    else result.content
                )
                lines.append(f"### 来源 {i}")
                lines.append(content)
                if result.metadata:
                    source = result.metadata.get("source", "Unknown")
                    category = result.metadata.get("category", "unknown")
                    lines.append(f"*来源: {source} | 类型: {category}*\n")

            return "\n".join(lines)
        except Exception as e:
            return f"规则检索错误: {str(e)}"

    @tool(
        "retrieve_episodes",
        description="""检索过去的战役/剧集记忆。
    当用户询问过去发生的事件、NPC、任务进展等情况时使用此工具。
    返回相关的历史事件摘要。""",
    )
    async def retrieve_episodes(
        self, query: str, session_id: Optional[str] = None
    ) -> str:
        """检索情景记忆."""
        try:
            episodes = await self._episodic_retriever.retrieve(
                query=query,
                session_id=session_id,
                top_k=3,
            )
            if not episodes:
                return "未找到相关历史记录。"

            lines = ["## 历史事件\n"]
            for i, episode in enumerate(episodes, 1):
                lines.append(f"### 剧集 {i}")
                lines.append(episode.summary)
                if episode.key_events:
                    lines.append("**关键事件:**")
                    for event in episode.key_events:
                        lines.append(f"- {event}")
                lines.append("")

            return "\n".join(lines)
        except Exception as e:
            return f"情景检索错误: {str(e)}"

    def classify_intent(self, query: str) -> str:
        """分类用户意图."""
        intent = self._intent_classifier.classify(query)
        return intent.type.value

    def get_tools(self):
        """获取工具列表."""
        return [self.retrieve_rules, self.retrieve_episodes]


# ============== Memory Integration ==============


class LangGraphMemory:
    """LangGraph 与项目 StateMemory 的桥接层."""

    def __init__(self, state_memory: Optional[StateMemory] = None) -> None:
        self._state_memory = state_memory or get_state_memory()

    def get_game_state(self) -> GameState:
        """获取当前游戏状态."""
        return self._state_memory.state

    def update_game_state(self, **kwargs) -> None:
        """更新游戏状态."""
        self._state_memory.update_scene(**kwargs)

    def update_player(self, player_name: str, **kwargs) -> None:
        """更新玩家状态."""
        self._state_memory.update_player(player_name, **kwargs)

    def add_dialogue(self, role: str, content: str) -> None:
        """添加对话历史."""
        self._state_memory.add_dialogue(role, content)

    def get_state_context(self) -> str:
        """获取状态上下文."""
        return self._state_memory.get_state_context()

    def get_dialogue_context(self, n: int = 10) -> str:
        """获取对话上下文."""
        return self._state_memory.get_dialogue_context(n)

    def to_langgraph_messages(self) -> list:
        """转换为 LangChain Message 格式."""
        history = self._state_memory.get_dialogue_history()
        messages = []
        for turn in history:
            if turn.role == "user":
                messages.append(HumanMessage(content=turn.content))
            else:
                messages.append(AIMessage(content=turn.content))
        return messages


# ============== LangGraph Agent ==============


class LangGraphDMAgent:
    """基于 LangGraph 的 DM Agent.

    特性:
    - ReAct 风格的 Tool 调用 (RAG 检索)
    - Checkpointer 持久化会话状态
    - 集成项目原有的 StateMemory
    """

    def __init__(
        self,
        llm=None,
        checkpointer: Optional[SqliteSaver] = None,
        state_memory: Optional[StateMemory] = None,
        knowledge_retriever: Optional[HybridKnowledgeRetriever] = None,
        episodic_retriever: Optional[EpisodicRetriever] = None,
    ) -> None:
        """
        初始化 LangGraph DM Agent.

        Args:
            llm: LangChain LLM 实例 (需要支持 tool calling)
            checkpointer: 状态持久化器
            state_memory: 项目 StateMemory 实例
            knowledge_retriever: 知识检索器
            episodic_retriever: 情景检索器
        """
        self._llm = llm
        self._checkpointer = checkpointer
        self._memory = LangGraphMemory(state_memory)

        # 初始化 RAG Tools
        self._rag_tools = RAGTools(
            knowledge_retriever=knowledge_retriever,
            episodic_retriever=episodic_retriever,
        )

        # 构建 graph
        self._graph = self._build_graph()
        self._agent_executor = None

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 状态流."""

        def should_retrieve(state: AgentState) -> AgentState:
            """判断是否需要检索."""
            query = (
                state.get("messages", [])[-1].content if state.get("messages") else ""
            )
            intent = self._rag_tools.classify_intent(query)

            # 更新状态
            state["intent"] = intent
            state["should_retrieve_knowledge"] = intent in (
                "query_rules",
                "combat",
                "proceed_story",
            )
            state["should_retrieve_episodes"] = intent in ("recall_history",)

            return state

        async def retrieve(state: AgentState) -> AgentState:
            """执行检索."""
            query = state["messages"][-1].content

            # 并行检索
            tasks = []
            if state.get("should_retrieve_knowledge"):
                tasks.append(self._rag_tools.retrieve_rules.ainvoke(query))
            if state.get("should_retrieve_episodes"):
                tasks.append(
                    self._rag_tools.retrieve_episodes.ainvoke(
                        {"query": query, "session_id": state.get("session_id")}
                    )
                )

            if tasks:
                results = await asyncio.gather(*tasks)
                knowledge = results[0] if state.get("should_retrieve_knowledge") else ""
                episodic = results[1] if state.get("should_retrieve_episodes") else ""
                state["knowledge_context"] = knowledge
                state["episodic_context"] = episodic
            return state

        import asyncio

        def generate_response(state: AgentState) -> AgentState:
            """生成响应 (占位, 实际由 LLM 生成)."""
            # 这里可以扩展为调用实际的 LLM
            # 当前返回占位响应
            state["response"] = (
                "As your Dungeon Master, I'm ready to guide your adventure. "
                "In the full implementation, this would be powered by Claude or another LLM."
            )
            return state

        # 构建图
        graph = StateGraph(AgentState)

        graph.add_node("should_retrieve", should_retrieve)
        graph.add_node("retrieve", retrieve)
        graph.add_node("generate", generate_response)

        graph.set_entry_point("should_retrieve")

        # 使用条件边: 根据 should_retrieve_knowledge 判断
        def route_after_intent(state: AgentState) -> str:
            if state.get("should_retrieve_knowledge") or state.get(
                "should_retrieve_episodes"
            ):
                return "retrieve"
            return "generate"

        graph.add_conditional_edges(
            "should_retrieve",
            route_after_intent,
            {
                "retrieve": "retrieve",
                "generate": "generate",
            },
        )

        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)

        return graph

    def compile(self, checkpointer: Optional[SqliteSaver] = None):
        """编译 agent 并配置 checkpointer."""
        checkpointer = checkpointer or self._checkpointer
        if checkpointer:
            self._agent_executor = self._graph.compile(checkpointer=checkpointer)
        else:
            self._agent_executor = self._graph.compile()
        return self

    async def invoke(
        self,
        query: str,
        session_id: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> str:
        """
        处理用户查询.

        Args:
            query: 用户输入
            session_id: 会话 ID (用于 checkpointer)
            config: 额外的配置

        Returns:
            Agent 响应
        """
        if self._agent_executor is None:
            self.compile()

        # 准备初始状态
        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "session_id": session_id,
            "game_state": None,
            "intent": None,
            "knowledge_context": None,
            "episodic_context": None,
            "should_retrieve_knowledge": False,
            "should_retrieve_episodes": False,
            "response": None,
        }

        # 配置 checkpointer
        run_config = config or {}
        if session_id:
            run_config["configurable"] = {"thread_id": session_id}

        # 执行
        result = await self._agent_executor.ainvoke(initial_state, run_config)

        # 更新对话历史到 StateMemory
        self._memory.add_dialogue("user", query)
        if result.get("messages"):
            last_message = result["messages"][-1]
            if hasattr(last_message, "content"):
                self._memory.add_dialogue("assistant", last_message.content)

        return result.get("response", "No response generated")

    async def stream(
        self,
        query: str,
        session_id: Optional[str] = None,
    ):
        """流式输出响应."""
        if self._agent_executor is None:
            self.compile()

        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "session_id": session_id,
            "game_state": None,
            "intent": None,
            "knowledge_context": None,
            "episodic_context": None,
            "should_retrieve_knowledge": False,
            "should_retrieve_episodes": False,
            "response": None,
        }

        run_config = {"configurable": {"thread_id": session_id}} if session_id else {}

        async for event in self._agent_executor.astream(initial_state, run_config):
            yield event


# ============== Factory ==============


def create_langgraph_agent(
    llm=None,
    db_path: str = ".checkpoints/langgraph.db",
) -> LangGraphDMAgent:
    """创建 LangGraph DM Agent.

    Args:
        llm: LangChain LLM 实例
        db_path: Checkpoint 数据库路径

    Returns:
        LangGraphDMAgent 实例
    """
    # 创建 checkpointer 目录
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # 创建 checkpointer
    checkpointer = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")

    return LangGraphDMAgent(
        llm=llm,
        checkpointer=checkpointer,
    ).compile(checkpointer)
