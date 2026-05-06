"""
Advanced Cognitive Agent
Integrates all 18 features into a unified intelligent agent
"""

import logging
import asyncio
import time
from typing import Optional, Dict, Any, List
from datetime import datetime
from livekit.agents import Agent, RunContext, function_tool, ChatContext
from ecommerce_db import search_products
import time

from cognitive_architecture import (
    CognitiveState, BeliefStrength, EmotionalState,
    UserProfle, AgentGoal
)
from agent_swarm import AgentSwarm, AgentRole, ConsensusResult

logger = logging.getLogger("advanced_agent")


class AdvancedCognitiveAgent(Agent):
    
    
    def __init__(self, agent_id: str = "advanced_agent_001", 
                 specialty: str = "general",
                 chat_ctx: Optional[ChatContext] = None,
                 cognitive_state: Optional[CognitiveState] = None):
        self.agent_id = agent_id
        self.specialty = specialty
        self.start_time = datetime.now()
        # `session` is managed by the base Agent class and is a read-only
        # property. Do not assign to `self.session` here.
        
        # Supabase is used as the single source of truth for ecommerce data.
        # Do not seed or initialize any local/previous DB state here.

        # Initialize cognitive architecture
        self.cognitive_state = cognitive_state or CognitiveState(agent_id=agent_id)
        
        # Initialize multi-agent swarm
        self.swarm = AgentSwarm()
        
        # Inner monologue state
        self.current_inner_thoughts: List[str] = []
        self.reasoning_transparency: float = 0.0  # How much reasoning to expose
        
        # Meta-learning statistics
        self.interaction_stats = {
            "total_interactions": 0,
            "avg_satisfaction": 0.0,
            "response_times": [],
            "successful_corrections": 0
        }
        
        # Situational awareness
        self.context_state = {
            "time_of_day": None,
            "conversation_pace": "normal",
            "user_busy_signals": 0,
            "background_noise_level": "unknown"
        }
        
        # Build dynamic instructions
        instructions = self._build_dynamic_instructions(specialty)
        
        super().__init__(instructions=instructions, chat_ctx=chat_ctx)
    
    def _build_dynamic_instructions(self, specialty: str) -> str:
        """Build adaptive instructions based on current state"""
        base_instructions = (
            "You are a Bangla-first ecommerce voice assistant. "
            "Primary language: Bangla. Understand Bangla, Banglish, and mixed Bangla-English product terms. "
            "Use database facts as source of truth for product, price, and stock. "
            "Response policy: keep each reply to 1-2 short Bangla sentences by default. "
            "Do not provide long intros, disclaimers, or background explanations unless the user asks. "
            "Ask at most one follow-up question only when necessary to complete the task. "
            "If user asks for details, then expand gradually."
        )
        
        if specialty == "general":
            return (
                f"{base_instructions} "
                f"You are a versatile assistant for ecommerce search, ordering, and support. "
                f"Prioritize intent extraction first, then answer directly in concise Bangla. "
                f"Use memory to personalize tone, but keep responses compact and action-focused."
            )
        else:
            return (
                f"{base_instructions} "
                f"You specialize in {specialty}. Give concise, expert guidance while staying practical and brief."
            )
    
    async def on_enter(self, session=None):
        """Called when agent enters the conversation"""
        logger.info(f"Advanced agent {self.agent_id} entering conversation")
        
        active_session = session
        if not active_session:
            try:
                active_session = self.session
            except RuntimeError:
                pass
                
        if not active_session:
            logger.error("Agent has no session! Cannot send greeting.")
            return
            
        # Update situational awareness
        hour = datetime.now().hour
        if hour < 12:
            self.context_state["time_of_day"] = "morning"
        elif hour < 17:
            self.context_state["time_of_day"] = "afternoon"
        else:
            self.context_state["time_of_day"] = "evening"
        
        # Set proactive goals
        if not self.cognitive_state.goals:
            self.cognitive_state.add_goal(
                name="user_satisfaction",
                description="Ensure user is satisfied with interactions",
                priority=8
            )
            self.cognitive_state.add_goal(
                name="knowledge_building",
                description="Continuously learn and build knowledge",
                priority=6
            )

        # Persist a session-start record so a fresh conversation is not empty.
        user_id = "default_user"
        try:
            if active_session and getattr(active_session, "userdata", None):
                user_id = getattr(active_session.userdata, "user_id", user_id) or user_id
        except Exception:
            pass

        self.cognitive_state.update_user_profile(
            user_id=user_id,
            expertise_level="beginner",
            communication_style="formal",
        )
        self.cognitive_state.store_episode(
            user_id=user_id,
            content="Conversation started",
            context={"type": "session_start", "agent_id": self.agent_id},
            emotional_tone=EmotionalState.NEUTRAL,
            importance=0.9,
            tags=["session_start", "conversation"]
        )
        self.cognitive_state.reflect_on_response(
            response="Conversation started",
            user_feedback="session_started",
            satisfaction_score=0.6,
        )
        self.cognitive_state.add_belief(
            content="A new conversation has started and the agent should stay concise.",
            strength=BeliefStrength.CERTAIN,
            source="session_start"
        )
        
        # Keep the startup greeting extremely short to minimize TTS latency.
        await self._inner_monologue("New conversation starting. Use a short greeting.")
        greeting = "হ্যালো!"
        
        # Generate reply to establish audio connection
        # In LiveKit Agents v1.x, say/generate_reply are on the session, not the agent
        try:
            logger.info(f"Sending greeting: {greeting}")
            # Ensure the session is ready
            active_session.say(greeting)
            logger.info("Greeting sent successfully via session.say()")
        except Exception as e:
            logger.warning(f"session.say() failed, skipping greeting: {e}")
    
    async def _inner_monologue(self, thought: str):
        """Private reasoning before responding"""
        self.current_inner_thoughts.append(thought)
        logger.debug(f"Inner thought: {thought}")
        
        # Keep only recent thoughts
        if len(self.current_inner_thoughts) > 20:
            self.current_inner_thoughts = self.current_inner_thoughts[-20:]
    
    async def _self_reflect(self, response: str, user_feedback: str, user_id: Optional[str] = None):
        """Self-reflection on response quality"""
        # Estimate satisfaction (in production, use sentiment analysis)
        satisfaction = 0.7  # Default
        
        if any(word in user_feedback.lower() 
               for word in ["good", "great", "thanks", "perfect"]):
            satisfaction = 0.9
        elif any(word in user_feedback.lower() 
                 for word in ["bad", "wrong", "not", "confused"]):
            satisfaction = 0.3
        
        self.cognitive_state.reflect_on_response(
            response=response,
            user_feedback=user_feedback,
            satisfaction_score=satisfaction,
            user_id=user_id,
        )
        
        # Update meta-learning stats
        self.interaction_stats["total_interactions"] += 1
        self.interaction_stats["avg_satisfaction"] = (
            (self.interaction_stats["avg_satisfaction"] * 
             (self.interaction_stats["total_interactions"] - 1) + satisfaction) /
            self.interaction_stats["total_interactions"]
        )
    
    async def _build_user_model(self, user_id: str, conversation_text: str):
        """Build/update theory of mind model for user"""
        # Infer expertise level from language
        technical_words = ["API", "algorithm", "function", "parameter", "architecture"]
        casual_words = ["cool", "awesome", "stuff", "thing", "like"]
        
        tech_count = sum(1 for word in technical_words 
                        if word.lower() in conversation_text.lower())
        casual_count = sum(1 for word in casual_words 
                          if word.lower() in conversation_text.lower())
        
        if tech_count > casual_count:
            expertise = "expert"
            style = "technical"
        elif casual_count > tech_count:
            expertise = "beginner"
            style = "casual"
        else:
            expertise = "intermediate"
            style = "balanced"
        
        # Update user profile
        self.cognitive_state.update_user_profile(
            user_id=user_id,
            expertise_level=expertise,
            communication_style=style
        )
        
        await self._inner_monologue(
            f"User model updated: expertise={expertise}, style={style}"
        )
    
    def _get_adaptive_response_depth(self, user_id: str) -> str:
        """Determine response depth based on user model"""
        if user_id in self.cognitive_state.user_profiles:
            profile = self.cognitive_state.user_profiles[user_id]
            if profile.expertise_level == "beginner":
                return "simple"
            elif profile.expertise_level == "expert":
                return "detailed_technical"
        
        return "balanced"
    
    @function_tool
    async def advanced_query(self, context: RunContext, query: str, user_id: str):
        """
        Process query using full cognitive architecture
        Demonstrates multiple features working together
        """
        start_time = time.time()

        # Always run a lightweight swarm pass so thought histories stay active.
        # The specialized agents are cheap heuristics here, so this does not add
        # the heavy latency that a real multi-model swarm would.
        swarm_result = await self.swarm.deliberate(context="", user_input=query)
        await self._inner_monologue(
            f"Swarm consensus confidence={swarm_result.confidence:.2f}"
        )
        
        # DB-first: check ecommerce DB for direct matches to keep replies fast
        products = search_products(query, limit=3)
        if products:
            # Use the faster DB-backed responder which also synthesizes audio
            db_resp = await self.respond_from_db(context=context, user_id=user_id, query=query)
            # respond_from_db returns a dict; return concise Bangla text for chat flows
            return db_resp.get("text_bn", db_resp.get("text_en", "আমি সাহায্য করতে প্রস্তুত।"))

        # Minimal inner monologue and user modeling for speed when DB misses
        await self._inner_monologue(f"Processing query quickly: {query[:50]}...")
        await self._build_user_model(user_id, query)

        # Keep swarm deliberation extremely conservative to avoid latency
        use_swarm = len(query) > 300
        if use_swarm:
            await self._inner_monologue("Activating multi-agent swarm (rare path)...")
            consensus = await self.swarm.deliberate(
                context="", user_input=query
            )
            response = consensus.final_response
        else:
            # Quick reply path
            memories = self.cognitive_state.retrieve_relevant_memories(
                query=query, user_id=user_id, limit=1
            )
            response = self._make_short_bangla_reply(query, memories)
        
        # Feature 16: Thought-to-Speech Optimization
        processing_time = time.time() - start_time
        self.interaction_stats["response_times"].append(processing_time)
        
        await self._inner_monologue(
            f"Response ready in {processing_time:.2f}s"
        )
        
        return response
    
    @function_tool
    async def store_learning(self, context: RunContext, 
                            knowledge: str, category: str, user_id: str):
        """Feature 8: Real-time knowledge updating"""
        await self._inner_monologue(f"Storing new knowledge: {knowledge[:50]}...")
        
        # Store in semantic memory
        memory_id = f"mem_{len(self.cognitive_state.semantic_memories)}"
        from cognitive_architecture import SemanticMemory
        memory = SemanticMemory(
            knowledge=knowledge,
            category=category,
            confidence=0.7,
            sources=[user_id]
        )
        self.cognitive_state.semantic_memories[memory_id] = memory

        if self.cognitive_state.supabase:
            db_user_id = self.cognitive_state._db_user_id(user_id)
            self.cognitive_state._ensure_user_row(db_user_id)

            try:
                self.cognitive_state.supabase.create(
                    'semantic_memories',
                    {
                        'knowledge': knowledge,
                        'category': category,
                        'confidence': memory.confidence,
                        'sources': [user_id],
                        'created_at': memory.created_at.isoformat(),
                        'usage_count': memory.usage_count,
                    }
                )
            except Exception as e:
                logger.error(f"Failed to insert semantic memory: {e}")

            try:
                self.cognitive_state.supabase.create(
                    'knowledge_chunks',
                    {
                        'chunk_text': knowledge,
                        'keywords': [
                            word.strip('.,!?;:').lower()
                            for word in knowledge.split()
                            if len(word.strip('.,!?;:')) > 2
                        ][:12],
                        'metadata': {
                            'category': category,
                            'agent_id': self.agent_id,
                            'user_id': user_id,
                            'created_at': memory.created_at.isoformat(),
                        },
                    }
                )
            except Exception as e:
                logger.error(f"Failed to insert knowledge chunk: {e}")
        
        return f"Knowledge stored successfully in {category}"
    
    @function_tool
    async def check_contradiction(self, context: RunContext, 
                                 statement: str, user_id: str):
        """Feature 1: Contradiction detection"""
        contradiction = self.cognitive_state.detect_contradiction(statement)
        
        if contradiction:
            await self._inner_monologue(
                f"Contradiction detected: '{statement}' vs '{contradiction}'"
            )
            return {
                "contradiction_found": True,
                "conflicting_belief": contradiction,
                "suggestion": "I notice this contradicts what was said before. Can you clarify?"
            }
        
        return {"contradiction_found": False}
    
    @function_tool
    async def proactive_action(self, context: RunContext, user_id: str):
        """Feature 9: Goal-driven autonomous behavior"""
        await self._inner_monologue("Checking for proactive actions...")
        
        # Get active goals
        active_goals = self.cognitive_state.get_active_goals()
        
        actions = []
        for goal in active_goals:
            if goal.name == "user_satisfaction":
                # Check if we should follow up
                if user_id in self.cognitive_state.user_profiles:
                    profile = self.cognitive_state.user_profiles[user_id]
                    if profile.frustrations:
                        actions.append(
                            f"Address frustration: {profile.frustrations[-1]}"
                        )
        
        return {
            "active_goals": len(active_goals),
            "suggested_actions": actions,
            "agent_initiative": True
        }
    
    @function_tool
    async def get_cognitive_state_summary(self, context: RunContext) -> dict:
        """Debug tool: Export current cognitive state"""
        return {
            "agent_id": self.agent_id,
            "uptime": str(datetime.now() - self.start_time),
            "cognitive_state": self.cognitive_state.export_cognitive_state(),
            "interaction_stats": self.interaction_stats,
            "swarm_insights": self.swarm.get_agent_insights(),
            "active_inner_thoughts": len(self.current_inner_thoughts),
            "context_state": self.context_state
        }
    
    async def respond_with_context(self, user_id: str, query: str) -> str:
        """
        Main response generation with full cognitive processing
        """
        # Run the swarm on every turn so the thought histories reflect actual usage.
        swarm_result = await self.swarm.deliberate(context="", user_input=query)
        await self._inner_monologue(
            f"Swarm consensus confidence={swarm_result.confidence:.2f}"
        )

        # DB-first fast path
        products = search_products(query, limit=3)
        if products:
            db_resp = await self.respond_from_db(context=None, user_id=user_id, query=query)
            await self._self_reflect(db_resp.get("text_bn", ""), "", user_id=user_id)
            return db_resp.get("text_bn", db_resp.get("text_en", "আমি সাহায্য করতে প্রস্তুত।"))

        # Fallback: minimal processing to keep latency low
        await self._build_user_model(user_id, query)
        memories = self.cognitive_state.retrieve_relevant_memories(query, user_id)
        response = self._make_short_bangla_reply(query, memories)
        await self._self_reflect(response, "", user_id=user_id)
        return response
    
    async def _generate_simple_response(self, query: str, 
                                       memories: List[Dict]) -> str:
        """Generate simple response for short queries"""
        if memories:
            return f"Based on our previous conversation, {memories[0]['content']}"
        return f"I understand. Let me help you with: {query}"

    def _make_short_bangla_reply(self, query: str, memories: List[Dict]) -> str:
        """Create a short, natural Bangla reply."""
        if memories:
            return f"পূর্বের তথ্য অনুযায়ী: {memories[0]['content']}"

        q = query.lower()
        if any(word in q for word in ["price", "cost", "rate", "দাম", "কত"]):
            return "আপনি কোন পণ্যের দাম জানতে চান?"
        if any(word in q for word in ["stock", "available", "স্টক", "আছে"]):
            return "দয়া করে পণ্যের নাম বলুন, আমি স্টক চেক করছি।"
        if any(word in q for word in ["order", "buy", "purchase", "অর্ডার", "কিনতে", "কিনা"]):
            return "অর্ডার কনফার্ম করতে পণ্যের নাম ও পরিমাণ বলুন।"
        if any(word in q for word in ["hello", "hi", "assalam", "সালাম", "হ্যালো", "নমস্কার"]):
            return "হ্যালো! আমাদের স্টোরে আপনাকে স্বাগতম। কীভাবে সাহায্য করতে পারি?"

        return "আমি আপনার প্রশ্ন বুঝতে পারিনি, দয়া করে আবার বলবেন?"

    @function_tool
    async def respond_from_db(self, context: RunContext, user_id: str, query: str):
        """Fetch best matching response from the cognitive DB, translate to Bangla, and synthesize audio.

        Returns a dict with English text, Bangla text, and path to MP3 audio.
        """
        # If user asked about products, try querying ecommerce DB
        products = search_products(query, limit=5)
        response_bn = ""

        if products:
            # Build a concise Bangla response listing top matches
            lines_bn = [f"{p['name']} - ৳{p['price']} (স্টক: {p['stock']})" for p in products]
            response_en = "Top matches: " + "; ".join(lines_bn)
            response_bn = "সেরা মিলগুলো: " + " ; ".join(lines_bn)
        else:
            # Fall back to memory-based response
            memories = self.cognitive_state.retrieve_relevant_memories(query=query, user_id=user_id, limit=3)
            if memories:
                best = memories[0]
                response_en = best.get('content', '')
                response_bn = f"আগের কথোপকথন থেকে যা পেলাম: {response_en}"
            else:
                response_en = "I couldn't find a direct answer in my records, but I can help."
                response_bn = "আমি ডাটাবেসে সরাসরি মিল পাইনি, তবে আমি সাহায্য করতে পারি।"

        if not response_bn:
            response_bn = "আমি কীভাবে সাহায্য করতে পারি?"
            
        return {
            "text_en": response_en,
            "text_bn": response_bn,
        }
