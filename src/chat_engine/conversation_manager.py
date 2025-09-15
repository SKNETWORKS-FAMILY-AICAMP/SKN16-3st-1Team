"""
Conversation Manager

대화 세션 관리, 컨텍스트 유지, 대화 히스토리 저장/로드
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os
import uuid
import structlog
from pathlib import Path

logger = structlog.get_logger(__name__)


@dataclass
class ConversationTurn:
    """대화 턴 데이터 클래스"""
    turn_id: str
    timestamp: datetime
    question: str
    answer: str
    context_used: bool = False
    context_content: Optional[str] = None
    source_files: Optional[List[str]] = None
    processing_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """딕셔너리에서 생성"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class ConversationSession:
    """대화 세션 데이터 클래스"""
    session_id: str
    created_at: datetime
    updated_at: datetime
    title: Optional[str] = None
    turns: Optional[List[ConversationTurn]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.turns is None:
            self.turns = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'title': self.title,
            'turns': [turn.to_dict() for turn in self.turns],
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationSession':
        """딕셔너리에서 생성"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        data['turns'] = [ConversationTurn.from_dict(turn_data) for turn_data in data.get('turns', [])]
        return cls(**data)


class ConversationManager:
    """
    대화 관리자

    대화 세션 생성, 대화 기록 저장/로드, 컨텍스트 관리
    """

    def __init__(self, storage_dir: str = "data/conversations"):
        """
        Args:
            storage_dir: 대화 기록 저장 디렉토리
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # 현재 활성 세션
        self.current_session: Optional[ConversationSession] = None

        logger.info("ConversationManager initialized", storage_dir=str(self.storage_dir))

    def create_new_session(self, title: Optional[str] = None) -> str:
        """새로운 대화 세션 생성"""
        session_id = str(uuid.uuid4())
        now = datetime.now()

        self.current_session = ConversationSession(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            title=title or f"대화 {now.strftime('%Y-%m-%d %H:%M')}"
        )

        logger.info(
            "New conversation session created",
            session_id=session_id,
            title=self.current_session.title
        )

        return session_id

    def add_turn(
        self,
        question: str,
        answer: str,
        context_used: bool = False,
        context_content: Optional[str] = None,
        source_files: Optional[List[str]] = None,
        processing_time: Optional[float] = None
    ) -> str:
        """현재 세션에 대화 턴 추가"""
        if not self.current_session:
            self.create_new_session()

        turn_id = str(uuid.uuid4())
        turn = ConversationTurn(
            turn_id=turn_id,
            timestamp=datetime.now(),
            question=question,
            answer=answer,
            context_used=context_used,
            context_content=context_content,
            source_files=source_files or [],
            processing_time=processing_time
        )

        self.current_session.turns.append(turn)
        self.current_session.updated_at = datetime.now()

        # 자동 저장
        self.save_current_session()

        logger.info(
            "Turn added to conversation",
            session_id=self.current_session.session_id,
            turn_id=turn_id,
            context_used=context_used
        )

        return turn_id

    def save_current_session(self) -> bool:
        """현재 세션을 파일에 저장"""
        if not self.current_session:
            logger.warning("No current session to save")
            return False

        try:
            file_path = self.storage_dir / f"{self.current_session.session_id}.json"

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.current_session.to_dict(), f, ensure_ascii=False, indent=2)

            logger.info(
                "Session saved successfully",
                session_id=self.current_session.session_id,
                file_path=str(file_path)
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to save session",
                session_id=self.current_session.session_id,
                error=str(e)
            )
            return False

    def load_session(self, session_id: str) -> bool:
        """세션을 파일에서 로드"""
        try:
            file_path = self.storage_dir / f"{session_id}.json"

            if not file_path.exists():
                logger.warning("Session file not found", session_id=session_id)
                return False

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.current_session = ConversationSession.from_dict(data)

            logger.info(
                "Session loaded successfully",
                session_id=session_id,
                turns_count=len(self.current_session.turns)
            )
            return True

        except Exception as e:
            logger.error("Failed to load session", session_id=session_id, error=str(e))
            return False

    def get_session_list(self) -> List[Dict[str, Any]]:
        """저장된 세션 목록 반환"""
        sessions = []

        try:
            for file_path in self.storage_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    sessions.append({
                        'session_id': data['session_id'],
                        'title': data.get('title', '제목 없음'),
                        'created_at': data['created_at'],
                        'updated_at': data['updated_at'],
                        'turns_count': len(data.get('turns', []))
                    })

                except Exception as e:
                    logger.warning(
                        "Failed to read session file",
                        file_path=str(file_path),
                        error=str(e)
                    )

        except Exception as e:
            logger.error("Failed to get session list", error=str(e))

        # 업데이트 시간 기준으로 정렬 (최신 순)
        sessions.sort(key=lambda x: x['updated_at'], reverse=True)

        logger.info("Session list retrieved", count=len(sessions))
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """세션 삭제"""
        try:
            file_path = self.storage_dir / f"{session_id}.json"

            if file_path.exists():
                file_path.unlink()

                # 현재 세션이 삭제된 세션이면 초기화
                if (self.current_session and
                    self.current_session.session_id == session_id):
                    self.current_session = None

                logger.info("Session deleted", session_id=session_id)
                return True
            else:
                logger.warning("Session file not found for deletion", session_id=session_id)
                return False

        except Exception as e:
            logger.error("Failed to delete session", session_id=session_id, error=str(e))
            return False

    def get_current_session_info(self) -> Optional[Dict[str, Any]]:
        """현재 세션 정보 반환"""
        if not self.current_session:
            return None

        return {
            'session_id': self.current_session.session_id,
            'title': self.current_session.title,
            'created_at': self.current_session.created_at.isoformat(),
            'updated_at': self.current_session.updated_at.isoformat(),
            'turns_count': len(self.current_session.turns)
        }

    def get_conversation_history(
        self,
        limit: Optional[int] = None,
        include_context: bool = False
    ) -> List[Dict[str, Any]]:
        """현재 세션의 대화 히스토리 반환"""
        if not self.current_session:
            return []

        turns = self.current_session.turns
        if limit:
            turns = turns[-limit:]  # 최근 limit개만

        history = []
        for turn in turns:
            turn_data = {
                'turn_id': turn.turn_id,
                'timestamp': turn.timestamp.isoformat(),
                'question': turn.question,
                'answer': turn.answer,
                'context_used': turn.context_used,
                'source_files': turn.source_files or [],
                'processing_time': turn.processing_time
            }

            if include_context and turn.context_content:
                turn_data['context_content'] = turn.context_content

            history.append(turn_data)

        return history

    def search_conversations(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """대화 내용에서 검색"""
        results = []

        try:
            for file_path in self.storage_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)

                    for turn in session_data.get('turns', []):
                        # 질문이나 답변에 검색어가 포함된 경우
                        if (query.lower() in turn['question'].lower() or
                            query.lower() in turn['answer'].lower()):

                            results.append({
                                'session_id': session_data['session_id'],
                                'session_title': session_data.get('title', '제목 없음'),
                                'turn_id': turn['turn_id'],
                                'timestamp': turn['timestamp'],
                                'question': turn['question'],
                                'answer': turn['answer'][:200] + '...' if len(turn['answer']) > 200 else turn['answer']
                            })

                except Exception as e:
                    logger.warning(
                        "Failed to search in session file",
                        file_path=str(file_path),
                        error=str(e)
                    )

        except Exception as e:
            logger.error("Failed to search conversations", error=str(e))

        # 시간 순으로 정렬 (최신 순)
        results.sort(key=lambda x: x['timestamp'], reverse=True)

        logger.info("Conversation search completed", query=query, results_count=len(results))
        return results[:limit]

    def export_session_to_text(self, session_id: Optional[str] = None) -> Optional[str]:
        """세션을 텍스트 형태로 내보내기"""
        session = self.current_session

        if session_id:
            # 다른 세션 로드
            original_session = self.current_session
            if self.load_session(session_id):
                session = self.current_session
                self.current_session = original_session
            else:
                return None

        if not session:
            return None

        lines = [
            f"대화 세션: {session.title}",
            f"세션 ID: {session.session_id}",
            f"생성 시간: {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"총 대화 턴: {len(session.turns)}",
            "=" * 50,
            ""
        ]

        for i, turn in enumerate(session.turns, 1):
            lines.extend([
                f"[턴 {i}] {turn.timestamp.strftime('%H:%M:%S')}",
                f"질문: {turn.question}",
                f"답변: {turn.answer}",
                ""
            ])

            if turn.context_used and turn.source_files:
                lines.append(f"참조 파일: {', '.join(turn.source_files)}")
                lines.append("")

        return "\n".join(lines)