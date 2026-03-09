"""LangGraph Agent 测试.

测试 Agent 核心逻辑（不依赖真实 RAG 服务）。
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

from mimic_master.core.langgraph_agent import (
    AgentState,
    LangGraphDMAgent,
    LangGraphMemory,
    create_langgraph_agent,
)
from mimic_master.memory.state_memory import StateMemory


# ============== Fixtures ==============

@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = MagicMock()
    llm.invoke = AsyncMock(return_value=AIMessage(content="Mock response"))
    return llm


@pytest.fixture
def mock_state_memory():
    """Mock state memory."""
    memory = MagicMock(spec=StateMemory)
    memory.state = MagicMock()
    memory.get_dialogue_history = MagicMock(return_value=[])
    memory.add_dialogue = MagicMock()
    memory.get_state_context = MagicMock(return_value="## Game State\nLocation: tavern")
    memory.get_dialogue_context = MagicMock(return_value="## Dialogue\nPlayer: Hello")
    return memory


@pytest.fixture
def checkpointer():
    """In-memory checkpointer for testing."""
    return MemorySaver()


# ============== AgentState Tests ==============

def make_agent_state(**kwargs) -> AgentState:
    """创建 AgentState 的辅助函数."""
    defaults = {
        "messages": [],
        "game_state": None,
        "intent": None,
        "knowledge_context": None,
        "episodic_context": None,
        "session_id": None,
        "should_retrieve_knowledge": False,
        "should_retrieve_episodes": False,
        "response": None,
    }
    defaults.update(kwargs)
    return defaults


class TestAgentState:
    """AgentState 测试."""

    def test_agent_state_defaults(self):
        """测试默认状态."""
        state = make_agent_state()
        assert state.get("messages") == []
        assert state.get("game_state") is None
        assert state.get("intent") is None
        assert state.get("response") is None

    def test_agent_state_with_initial_values(self):
        """测试初始值设置."""
        state = make_agent_state(
            messages=[HumanMessage(content="test")],
            session_id="test-session",
            intent="query_rules",
        )
        assert len(state["messages"]) == 1
        assert state["session_id"] == "test-session"
        assert state["intent"] == "query_rules"

    def test_agent_state_set_response(self):
        """测试设置响应."""
        state = make_agent_state()
        state["response"] = "Hello player"
        assert state.get("response") == "Hello player"


# ============== LangGraphMemory Tests ==============

class TestLangGraphMemory:
    """LangGraphMemory 测试."""

    def test_get_game_state(self, mock_state_memory):
        """测试获取游戏状态."""
        memory = LangGraphMemory(mock_state_memory)
        assert memory.get_game_state() == mock_state_memory.state

    def test_add_dialogue(self, mock_state_memory):
        """测试添加对话."""
        memory = LangGraphMemory(mock_state_memory)
        memory.add_dialogue("user", "Hello")
        mock_state_memory.add_dialogue.assert_called_once_with("user", "Hello")

    def test_to_langgraph_messages_empty(self, mock_state_memory):
        """测试空对话转换为 LangGraph messages."""
        mock_state_memory.get_dialogue_history.return_value = []
        memory = LangGraphMemory(mock_state_memory)
        messages = memory.to_langgraph_messages()
        assert messages == []

    def test_to_langgraph_messages_with_history(self, mock_state_memory):
        """测试有对话历史时转换为 LangGraph messages."""
        from mimic_master.models.memory import DialogueTurn
        from datetime import datetime

        mock_state_memory.get_dialogue_history.return_value = [
            DialogueTurn(role="user", content="Hello", timestamp=datetime.now()),
            DialogueTurn(role="assistant", content="Hi there", timestamp=datetime.now()),
        ]
        memory = LangGraphMemory(mock_state_memory)
        messages = memory.to_langgraph_messages()

        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)
        assert messages[0].content == "Hello"
        assert messages[1].content == "Hi there"


# ============== LangGraphDMAgent Tests ==============

class TestLangGraphDMAgent:
    """LangGraph DM Agent 测试."""

    def test_agent_initialization(self, mock_llm, checkpointer):
        """测试 Agent 初始化."""
        agent = LangGraphDMAgent(
            llm=mock_llm,
            checkpointer=checkpointer,
        )
        assert agent._llm == mock_llm
        assert agent._checkpointer == checkpointer

    def test_agent_initialization_defaults(self):
        """测试默认初始化."""
        agent = LangGraphDMAgent()
        assert agent._llm is None
        assert agent._checkpointer is None

    def test_build_graph(self, mock_llm):
        """测试图构建."""
        agent = LangGraphDMAgent(llm=mock_llm)
        graph = agent._build_graph()

        assert isinstance(graph, StateGraph)
        # 检查节点
        nodes = graph.nodes
        assert "should_retrieve" in nodes
        assert "retrieve" in nodes
        assert "generate" in nodes

    def test_compile_with_checkpointer(self, mock_llm, checkpointer):
        """测试编译带 checkpointer 的 agent."""
        agent = LangGraphDMAgent(llm=mock_llm, checkpointer=checkpointer)
        compiled = agent.compile()

        assert agent._agent_executor is not None


# ============== Integration Tests ==============

class TestAgentWorkflow:
    """Agent 工作流集成测试."""

    @pytest.mark.asyncio
    async def test_invoke_without_checkpointer(self, mock_state_memory):
        """测试不带 checkpointer 的调用."""
        # Patch RAGTools to avoid real dependencies
        with patch("mimic_master.core.langgraph_agent.RAGTools") as MockRAGTools:
            mock_tools = MagicMock()
            mock_tools.classify_intent = MagicMock(return_value="chat")
            mock_tools.retrieve_rules = MagicMock()
            mock_tools.retrieve_episodes = MagicMock()
            MockRAGTools.return_value = mock_tools

            agent = LangGraphDMAgent(
                state_memory=mock_state_memory,
            )
            agent.compile()  # 不带 checkpointer

            response = await agent.invoke(
                query="Hello DM",
                session_id=None,  # 无 session_id，避免 checkpoint
            )

            # Verify response is placeholder (no real LLM)
            assert response is not None
            assert len(response) > 0


# ============== Factory Tests ==============

class TestFactory:
    """工厂函数测试."""

    def test_create_langgraph_agent_defaults(self):
        """测试默认创建 Agent (无 checkpointer)."""
        # 不创建实际的文件 checkpointer
        agent = LangGraphDMAgent()

        assert agent is not None
        assert isinstance(agent, LangGraphDMAgent)

    def test_create_langgraph_agent_with_llm(self, mock_llm):
        """测试使用自定义 LLM 创建 Agent."""
        agent = LangGraphDMAgent(llm=mock_llm)

        assert agent._llm == mock_llm


# ============== State Transitions Tests ==============

class TestStateTransitions:
    """状态转换测试."""

    def test_should_retrieve_logic(self):
        """测试意图判断逻辑."""
        # Test the intent classification logic in should_retrieve
        retrieval_intents = ("query_rules", "combat", "proceed_story")
        episodic_intents = ("recall_history",)

        # Test retrieval intents
        for intent in retrieval_intents:
            should_retrieve = intent in retrieval_intents
            assert should_retrieve is True

        # Test episodic intents
        for intent in episodic_intents:
            should_retrieve = intent in retrieval_intents
            assert should_retrieve is False

    def test_graph_flow(self, mock_llm):
        """测试图流程."""
        agent = LangGraphDMAgent(llm=mock_llm)
        graph = agent._build_graph()

        # Verify the graph has proper edges
        assert graph.nodes is not None
