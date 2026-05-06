"""
Cognitive Architecture Module
Implements persistent cognitive identity, memory systems, and self-reflection
Integrated with Supabase.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import numpy as np
from pathlib import Path

from supabase_agent import supabase_agent

logger = logging.getLogger("cognitive_architecture")

class BeliefStrength(Enum):
    WEAK = 0.3
    MODERATE = 0.6
    STRONG = 0.8
    CERTAIN = 1.0


class EmotionalState(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    FRUSTRATED = "frustrated"
    CONFUSED = "confused"
    SATISFIED = "satisfied"
    URGENT = "urgent"


@dataclass
class Belief:
    """Represents a belief held by the agent"""
    content: str
    strength: BeliefStrength = BeliefStrength.MODERATE
    source: str = ""  # Where this belief came from
    created_at: datetime = field(default_factory=datetime.now)
    last_verified: datetime = field(default_factory=datetime.now)
    confidence: float = 0.7
    contradictions: List[str] = field(default_factory=list)


@dataclass
class UserPreference:
    """Tracks user preferences and patterns"""
    category: str
    preference: str
    strength: float = 0.5
    observed_count: int = 1
    last_observed: datetime = field(default_factory=datetime.now)


@dataclass
class EpisodicMemory:
    """A specific event/conversation episode"""
    timestamp: datetime
    user_id: str
    content: str
    context: Dict[str, Any]
    emotional_tone: EmotionalState = EmotionalState.NEUTRAL
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)


@dataclass
class SemanticMemory:
    """Generalized knowledge learned from experiences"""
    knowledge: str
    category: str
    confidence: float = 0.5
    sources: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0


@dataclass
class UserProfle:
    """Comprehensive user model"""
    user_id: str
    expertise_level: str = "beginner"  # beginner, intermediate, expert
    preferences: Dict[str, UserPreference] = field(default_factory=dict)
    emotional_state: EmotionalState = EmotionalState.NEUTRAL
    interaction_count: int = 0
    first_seen: datetime = field(default_factory=datetime.now)
    last_interaction: datetime = field(default_factory=datetime.now)
    communication_style: str = "formal"  # formal, casual, technical
    goals: List[str] = field(default_factory=list)
    frustrations: List[str] = field(default_factory=list)


@dataclass
class AgentGoal:
    """Agent's internal goals"""
    name: str
    description: str
    priority: int = 5  # 1-10
    progress: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    status: str = "active"  # active, completed, abandoned


class CognitiveState:
    """
    The agent's persistent cognitive identity
    Maintains beliefs, goals, preferences that evolve over time
    """
    
    def __init__(self, agent_id: str, db_path: str = "agent_memory.db", load_async: bool = False):
        self.agent_id = agent_id
        # db_path is kept for interface compatibility but ignored
        self.beliefs: Dict[str, Belief] = {}
        self.goals: Dict[str, AgentGoal] = {}
        self.user_profiles: Dict[str, UserProfle] = {}
        self.episodic_memories: List[EpisodicMemory] = []
        self.semantic_memories: Dict[str, SemanticMemory] = {}
        
        # Inner monologue state
        self.current_thoughts: List[str] = []
        self.self_reflection_log: List[Dict] = []
        self.learning_lessons: List[Dict[str, Any]] = []
        
        self.supabase = supabase_agent
        self._live_sync_enabled = False
        self._init_database()
        
        if not load_async:
            self._load_state()
            self._enable_live_sync()
    
    async def load_state_async(self):
        """Asynchronously load state using a thread pool to avoid blocking the event loop"""
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_state)
        await loop.run_in_executor(None, self._enable_live_sync)
    
    def _init_database(self):
        """Database assumed to be initialized externally for Supabase."""
        pass

    def _db_user_id(self, user_id: str) -> str:
        """Convert local user identifiers to a stable UUID string for Supabase.

        Some Supabase tables use UUID columns for user IDs, while the agent
        works with arbitrary conversation/user strings. Using uuid5 keeps the
        mapping deterministic without requiring the caller to know the DB type.
        """
        try:
            return str(uuid.UUID(str(user_id)))
        except Exception:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.agent_id}:{user_id}"))

    def _belief_strength_db_value(self, strength: BeliefStrength) -> str:
        """Return the DB enum value for a belief strength."""
        return strength.name.lower()

    def _has_goal(self, name: str) -> bool:
        return any(goal.name == name for goal in self.goals.values())

    def _ensure_user_row(self, db_user_id: str) -> None:
        """Create a minimal users row so FK-backed tables can reference it."""
        if not self.supabase:
            return

        try:
            existing = self.supabase.read('users', filters={'id': db_user_id}, limit=1)
            if existing:
                return
        except Exception:
            # If the lookup fails, attempt insert anyway; the DB will tell us
            # whether the row already exists or whether permissions are missing.
            pass

        try:
            self.supabase.create('users', {'id': db_user_id})
            logger.debug(f"✓ Seeded users row for {db_user_id}")
        except Exception as e:
            logger.warning(f"Could not seed users row for {db_user_id}: {e}")
    
    def _load_state(self):
        """Load cognitive state from Supabase"""
        if not self.supabase:
            return
        try:
            self.beliefs.clear()
            self.goals.clear()
            self.user_profiles.clear()
            self.episodic_memories.clear()
            self.semantic_memories.clear()

            # Load beliefs
            b_resp = self.supabase.read('beliefs')
            for row in b_resp:
                strength_value = row.get('strength')
                try:
                    strength = BeliefStrength(strength_value) if strength_value is not None else BeliefStrength.MODERATE
                except Exception:
                    strength = BeliefStrength.MODERATE
                created_at_raw = row.get('created_at')
                created_at = datetime.fromisoformat(created_at_raw) if created_at_raw else datetime.now()
                belief = Belief(
                    content=row.get('content'),
                    strength=strength,
                    source=row.get('source'),
                    created_at=created_at,
                    confidence=row.get('confidence') or 0.7,
                )
                self.beliefs[str(row.get('id'))] = belief
            
            # Load semantic memories
            sm_resp = self.supabase.read('semantic_memories')
            for row in sm_resp:
                memory = SemanticMemory(
                    knowledge=row.get('knowledge'),
                    category=row.get('category'),
                    confidence=row.get('confidence'),
                    sources=row.get('sources') if isinstance(row.get('sources'), list) else json.loads(row.get('sources', '[]')),
                    created_at=datetime.fromisoformat(row.get('created_at')),
                    usage_count=row.get('usage_count')
                )
                self.semantic_memories[row.get('id')] = memory
            
            # Load user profiles
            up_resp = self.supabase.read('user_profiles')
            for row in up_resp:
                # Parse preferences correctly from JSON
                prefs_raw = row.get('preferences') if isinstance(row.get('preferences'), dict) else json.loads(row.get('preferences', '{}'))
                prefs = {}
                for k, v in prefs_raw.items():
                    last_observed_raw = v.get('last_observed')
                    prefs[k] = UserPreference(
                        category=v.get('category'),
                        preference=v.get('preference'),
                        strength=v.get('strength', 0.5),
                        observed_count=v.get('observed_count', 1),
                        last_observed=datetime.fromisoformat(last_observed_raw) if last_observed_raw else datetime.now()
                    )
                    
                first_seen_raw = row.get('first_seen')
                last_interaction_raw = row.get('last_interaction')
                profile = UserProfle(
                    user_id=row.get('user_id'),
                    expertise_level=row.get('expertise_level'),
                    preferences=prefs,
                    emotional_state=EmotionalState(row.get('emotional_state') or EmotionalState.NEUTRAL.value),
                    interaction_count=row.get('interaction_count'),
                    first_seen=datetime.fromisoformat(first_seen_raw) if first_seen_raw else datetime.now(),
                    last_interaction=datetime.fromisoformat(last_interaction_raw) if last_interaction_raw else datetime.now(),
                    communication_style=row.get('communication_style'),
                    goals=row.get('goals') if isinstance(row.get('goals'), list) else json.loads(row.get('goals', '[]')),
                    frustrations=row.get('frustrations') if isinstance(row.get('frustrations'), list) else json.loads(row.get('frustrations', '[]'))
                )
                self.user_profiles[row.get('user_id')] = profile
                
            logger.info(f"Loaded cognitive state: {len(self.beliefs)} beliefs, "
                      f"{len(self.semantic_memories)} semantic memories, "
                      f"{len(self.user_profiles)} user profiles")
        except Exception as e:
            logger.warning(f"Could not load previous state from Supabase: {e}")

    def _enable_live_sync(self):
        """Subscribe to Supabase table changes and keep memory in sync."""
        if not self.supabase or self._live_sync_enabled:
            return

        def _refresh_state(_payload: Dict[str, Any]):
            try:
                self._load_state()
            except Exception as exc:
                logger.warning(f"Live sync refresh failed: {exc}")

        for table in [
            'beliefs',
            'semantic_memories',
            'user_profiles',
            'episodic_memories',
            'agent_goals',
            'self_reflections',
        ]:
            try:
                self.supabase.subscribe(table, _refresh_state)
            except Exception as e:
                logger.warning(f"Failed to subscribe to {table} updates: {e}")

        self._live_sync_enabled = True
    
    def add_belief(self, content: str, strength: BeliefStrength, source: str) -> str:
        """Add or update a belief"""
        belief_id = str(uuid.uuid4())
        belief = Belief(content=content, strength=strength, source=source)
        self.beliefs[belief_id] = belief
        
        if self.supabase:
            data = {
                "id": belief_id,
                "content": content,
                "strength": self._belief_strength_db_value(strength),
                "source": source,
                "created_at": belief.created_at.isoformat(),
                "confidence": belief.confidence
            }
            try:
                result = self.supabase.upsert('beliefs', data)
                logger.debug(f"✓ Belief upserted: id={belief_id}, strength={strength.name}, content={content[:50]}, result={result}")
            except Exception as e:
                logger.error(f"✗ Failed to upsert belief: {e}, data={data}")
        else:
            logger.warning("⚠️  Supabase not configured, belief stored locally only")
        
        logger.info(f"Added belief: {content[:50]}...")
        return belief_id
    
    def detect_contradiction(self, new_statement: str) -> Optional[str]:
        """Detect contradictions with existing beliefs"""
        contradictions = []
        for belief_id, belief in self.beliefs.items():
            if belief.confidence < 0.7:
                continue
            if ("not" in new_statement.lower() and 
                "not" not in belief.content.lower()):
                if any(word in new_statement.lower() 
                      for word in belief.content.lower().split()):
                    contradictions.append(belief.content)
        if contradictions:
            return contradictions[0]
        return None
    
    def store_episode(self, user_id: str, content: str, 
                     context: Dict, emotional_tone: EmotionalState,
                     importance: float, tags: List[str]):
        """Store an episodic memory"""
        episode = EpisodicMemory(
            timestamp=datetime.now(),
            user_id=user_id,
            content=content,
            context=context,
            emotional_tone=emotional_tone,
            importance=importance,
            tags=tags
        )
        self.episodic_memories.append(episode)
        
        if len(self.episodic_memories) > 1000:
            self.episodic_memories = self.episodic_memories[-1000:]
        
        if self.supabase:
            db_user_id = self._db_user_id(user_id)
            self._ensure_user_row(db_user_id)
            data = {
                "user_id": db_user_id,
                "content": content,
                "context": episode.context,  # JSON
                "emotional_tone": emotional_tone.value,
                "importance": importance,
                "tags": episode.tags  # JSON
            }
            try:
                result = self.supabase.create('episodic_memories', data)
                logger.debug(f"✓ Episodic memory stored: user_id={user_id}, content_len={len(content)}, result={result}")
            except Exception as e:
                logger.error(f"✗ Failed to insert episodic memory: {e}, data={data}")
    
    def retrieve_relevant_memories(self, query: str, user_id: str, 
                                  limit: int = 5) -> List[Dict]:
        """Retrieve relevant memories based on query"""
        relevant = []
        query_words = set(query.lower().split())
        
        for episode in reversed(self.episodic_memories):
            if episode.user_id != user_id:
                continue
            episode_words = set(episode.content.lower().split())
            similarity = len(query_words & episode_words) / len(query_words) if query_words else 0
            if similarity > 0.1:
                relevant.append({
                    "type": "episodic",
                    "content": episode.content,
                    "timestamp": episode.timestamp,
                    "relevance": similarity * episode.importance
                })
        
        for mem_id, memory in self.semantic_memories.items():
            mem_words = set(memory.knowledge.lower().split())
            similarity = len(query_words & mem_words) / len(query_words) if query_words else 0
            if similarity > 0.1:
                relevant.append({
                    "type": "semantic",
                    "content": memory.knowledge,
                    "relevance": similarity * memory.confidence
                })
        
        relevant.sort(key=lambda x: x["relevance"], reverse=True)
        return relevant[:limit]
    
    def update_user_profile(self, user_id: str, **kwargs):
        """Update or create user profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfle(user_id=user_id)
        
        profile = self.user_profiles[user_id]
        
        for key, value in kwargs.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        
        profile.last_interaction = datetime.now()
        profile.interaction_count += 1
        
        if self.supabase:
            db_user_id = self._db_user_id(profile.user_id)
            self._ensure_user_row(db_user_id)
            data = {
                "user_id": db_user_id,
                "expertise_level": profile.expertise_level,
                "preferences": {k: asdict(v) for k, v in profile.preferences.items()},
                "emotional_state": profile.emotional_state.value,
                "interaction_count": profile.interaction_count,
                "first_seen": profile.first_seen.isoformat(),
                "last_interaction": profile.last_interaction.isoformat(),
                "communication_style": profile.communication_style,
                "goals": profile.goals,
                "frustrations": profile.frustrations
            }
            try:
                self.supabase.upsert('user_profiles', data)
            except Exception as e:
                logger.error(f"Failed to upsert user profile: {e}")
    
    def add_goal(self, name: str, description: str, priority: int = 5,
                deadline: Optional[datetime] = None) -> str:
        """Add a new goal"""
        goal_id = str(uuid.uuid4())
        goal = AgentGoal(
            name=name,
            description=description,
            priority=priority,
            deadline=deadline
        )
        self.goals[goal_id] = goal
        
        if self.supabase:
            payloads = [
                {
                    "id": goal_id,
                    "goal_name": name,
                    "description": description,
                    "priority": priority,
                    "progress": goal.progress,
                    "created_at": goal.created_at.isoformat(),
                    "deadline": deadline.isoformat() if deadline else None,
                    "status": goal.status,
                },
                {
                    "id": goal_id,
                    "goal_name": name,
                    "description": description,
                    "priority": priority,
                    "progress": goal.progress,
                    "created_at": goal.created_at.isoformat(),
                    "deadline": deadline.isoformat() if deadline else None,
                    "status": goal.status,
                },
                {
                    "id": goal_id,
                    "goal_name": name,
                    "description": description,
                    "priority": priority,
                    "progress": goal.progress,
                    "created_at": goal.created_at.isoformat(),
                    "deadline": deadline.isoformat() if deadline else None,
                    "status": goal.status,
                },
            ]
            last_error = None
            for data in payloads:
                try:
                    result = self.supabase.upsert('agent_goals', data)
                    logger.debug(f"✓ Goal upserted: id={goal_id}, name={name}, priority={priority}, result={result}")
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(f"Goal upsert variant failed: {e}, data={data}")
            if last_error:
                logger.error(f"✗ Failed to upsert goal to agent_goals after fallbacks: {last_error}")
        else:
            logger.warning("⚠️  Supabase not configured, goal stored locally only")
        
        return goal_id
    
    def learn_from_feedback(self, response: str, user_feedback: str,
                            satisfaction_score: float, user_id: Optional[str] = None):
        """Turn reflection into a durable lesson and update future strategy."""
        lesson = {
            "timestamp": datetime.now(),
            "response": response[:100],
            "user_feedback": user_feedback,
            "satisfaction_score": satisfaction_score,
            "lesson": "",
        }

        if satisfaction_score < 0.5:
            lesson_text = (
                "The last answer was weak. Prefer shorter, clearer answers, "
                "ask one clarifying question earlier, and reduce unnecessary detail."
            )
            lesson["lesson"] = lesson_text
            self.add_belief(
                content=lesson_text,
                strength=BeliefStrength.STRONG,
                source="self_reflection"
            )
            if not self._has_goal("response_improvement"):
                self.add_goal(
                    name="response_improvement",
                    description="Reduce low-satisfaction responses and improve clarity",
                    priority=9,
                )
            if user_id and user_id in self.user_profiles:
                profile = self.user_profiles[user_id]
                if "confused" not in profile.frustrations:
                    profile.frustrations.append("confused")
                profile.communication_style = "more_direct"
        elif satisfaction_score > 0.8:
            lesson_text = (
                "The current answer style works well. Keep it concise, direct, "
                "and Bangla-first for this user."
            )
            lesson["lesson"] = lesson_text
            self.add_belief(
                content=lesson_text,
                strength=BeliefStrength.MODERATE,
                source="positive_feedback"
            )
            if user_id and user_id in self.user_profiles:
                profile = self.user_profiles[user_id]
                profile.communication_style = "concise"
        else:
            lesson["lesson"] = (
                "Neutral feedback: keep collecting evidence and favor stable, compact answers."
            )

        self.learning_lessons.append(lesson)

        if self.supabase:
            try:
                self.supabase.create(
                    'semantic_memories',
                    {
                        'knowledge': lesson["lesson"],
                        'category': 'feedback',
                        'confidence': max(0.5, min(1.0, satisfaction_score)),
                        'sources': [user_id] if user_id else ['self_reflection'],
                        'created_at': lesson['timestamp'].isoformat(),
                        'usage_count': 0,
                    }
                )
            except Exception as e:
                logger.error(f"Failed to persist feedback lesson: {e}")

    def reflect_on_response(self, response: str, user_feedback: str,
                           satisfaction_score: float, user_id: Optional[str] = None):
        """Self-reflection on response quality"""
        reflection = {
            "timestamp": datetime.now(),
            "response": response[:100],
            "user_feedback": user_feedback,
            "satisfaction_score": satisfaction_score,
            "thoughts": []
        }
        
        if satisfaction_score < 0.5:
            reflection["thoughts"].append("User seemed unsatisfied. I should improve my response.")
        elif satisfaction_score > 0.8:
            reflection["thoughts"].append("User was very satisfied. This approach worked well.")
        
        self.self_reflection_log.append(reflection)

        self.learn_from_feedback(
            response=response,
            user_feedback=user_feedback,
            satisfaction_score=satisfaction_score,
            user_id=user_id,
        )
        
        if self.supabase:
            data = {
                "thought": str(reflection["thoughts"]),
                "evaluation": user_feedback,
                "score": satisfaction_score,
                "action_taken": "none" if satisfaction_score > 0.7 else "need_improvement"
            }
            try:
                self.supabase.create('self_reflections', data)
            except Exception as e:
                logger.error(f"Failed to insert self reflection: {e}")
        
        logger.info(f"Self-reflection: satisfaction={satisfaction_score:.2f}")
    
    def get_active_goals(self) -> List[AgentGoal]:
        """Get all active goals sorted by priority"""
        return sorted(
            [g for g in self.goals.values() if g.status == "active"],
            key=lambda x: x.priority,
            reverse=True
        )
    
    def export_cognitive_state(self) -> Dict:
        """Export full cognitive state for debugging"""
        return {
            "agent_id": self.agent_id,
            "beliefs_count": len(self.beliefs),
            "goals_count": len(self.goals),
            "user_profiles_count": len(self.user_profiles),
            "episodic_memories_count": len(self.episodic_memories),
            "semantic_memories_count": len(self.semantic_memories),
            "self_reflections_count": len(self.self_reflection_log)
        }
