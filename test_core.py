"""ChatDeepSeek CLI 核心逻辑测试（不调用 API）"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from session_manager import SessionManager, Session, Message
from context_builder import ContextBuilder, parse_at_anchors, estimate_tokens
from semantic_search import semantic_search, keyword_search


def test_at_anchors():
    """测试 @ 锚点解析。"""
    # 语义锚点
    text = '@网络超时 结合之前那个问题，给出重试策略'
    clean, semantics, exacts = parse_at_anchors(text)
    assert '网络超时' in semantics, f'语义锚点解析失败: {semantics}'
    assert '网络超时' not in clean, '清理后不应包含锚点词'
    print(f'  [OK] 语义锚点: clean={clean!r}, semantics={semantics}')

    # 精确短语锚点
    text2 = '参考 @"网络超时" 和 @线程池 的内容'
    clean2, s2, e2 = parse_at_anchors(text2)
    assert '网络超时' in e2, f'精确锚点解析失败: {e2}'
    assert '线程池' in s2, f'语义锚点解析失败: {s2}'
    print(f'  [OK] 精确锚点: clean={clean2!r}, semantics={s2}, exacts={e2}')


def test_session_manager():
    """测试会话管理。"""
    sm = SessionManager()
    print(f'  当前会话: {sm.current_name}')
    print(f'  会话数: {len(sm.sessions)}')

    # 创建测试会话
    sm.create_session('test_session')
    sm.current_name = 'test_session'
    s = sm.current
    assert s is not None
    s.add_message(Message(role='user', content='Python异常处理的最佳实践是什么？'))
    s.add_message(Message(role='assistant', content='异常处理最佳实践包括：使用具体的异常类...'))
    s.add_message(Message(role='user', content='网络超时如何处理？'))
    s.add_message(Message(role='assistant', content='网络超时可以通过设置timeout和重试机制...'))
    print(f'  [OK] 消息数: {s.message_count}')

    # 测试列表
    sessions = sm.list_sessions()
    assert len(sessions) > 0

    # 测试重命名
    sm.rename_session('test_renamed')
    assert sm.current.name == 'test_renamed'
    print(f'  [OK] 重命名成功: {sm.current.name}')

    # 测试克隆
    sm.clone_session('test_renamed', 'test_cloned')
    assert 'test_cloned' in sm.sessions
    print(f'  [OK] 克隆成功')

    # 测试删除
    sm.delete_session('test_cloned')
    assert 'test_cloned' not in sm.sessions
    print(f'  [OK] 删除成功')

    # 清理
    sm.delete_session('test_renamed')


def test_context_builder():
    """测试上下文构建器。"""
    sm = SessionManager()
    sm.create_session('ctx_test')
    sm.current_name = 'ctx_test'
    s = sm.current

    # 添加多条消息
    for i in range(15):
        s.add_message(Message(role='user', content=f'这是第{i}条用户消息，关于Python编程的内容'))
        s.add_message(Message(role='assistant', content=f'这是第{i}条助手回复，包含代码示例和解释'))

    s.add_message(Message(role='user', content='网络超时异常应该怎么处理？'))
    s.add_message(Message(role='assistant', content='网络超时可以通过设置timeout参数和实现重试机制来处理'))

    builder = ContextBuilder(s)
    msgs, info = builder.build('网络超时怎么处理？')
    print(f'  [OK] 构建消息数: {len(msgs)}')
    print(f'  [OK] 优化信息: {info}')

    # 测试 token 估算
    tokens = estimate_tokens('Hello, world! 你好世界')
    print(f'  [OK] Token 估算: {tokens}')

    # 清理
    sm.delete_session('ctx_test')


def test_semantic_search():
    """测试语义搜索（仅测试关键词搜索，避免加载大模型）。"""
    candidates = [
        (0, 'Python异常处理的最佳实践是什么'),
        (1, '网络超时可以通过设置timeout来处理'),
        (2, '文件读写操作需要注意编码问题'),
        (3, 'TCP连接超时和HTTP请求超时的区别'),
    ]

    # 测试关键词搜索
    kw_results = keyword_search(['网络超时'], candidates, top_k=2)
    print(f'  [OK] 关键词搜索结果: {kw_results}')
    assert len(kw_results) > 0, '关键词搜索应有结果'

    # 测试简单相似度（来自 semantic_search 模块的 _simple_similarity）
    from semantic_search import _simple_similarity
    sim = _simple_similarity('网络超时', '网络超时可以通过设置timeout来处理')
    print(f'  [OK] 简单相似度: {sim:.3f}')
    assert sim > 0.3, '相似度应足够高'


if __name__ == '__main__':
    print('=== ChatDeepSeek 核心逻辑测试 ===\n')
    try:
        test_at_anchors()
        test_semantic_search()
        test_session_manager()
        test_context_builder()
        print('\n[PASS] 所有核心逻辑测试通过!')
    except AssertionError as e:
        print(f'\n[FAIL] {e}')
        sys.exit(1)
    except Exception as e:
        print(f'\n[ERROR] {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
