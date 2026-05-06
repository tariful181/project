import logging
from dataclasses import dataclass, field
from typing import Optional

import os
from dotenv import load_dotenv
import asyncio

# Load .env from the script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(current_dir, ".env"))

from livekit import api
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.job import get_job_context
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import openai, silero

from advanced_agent import AdvancedCognitiveAgent
from cognitive_architecture import CognitiveState, BeliefStrength, EmotionalState

logger = logging.getLogger("advanced_multi_agent")


# Intent types for dynamic query routing
class QueryIntent:
    PRODUCT_SEARCH = "product_search"
    PRODUCT_DETAILS = "product_details"
    PRICE_QUERY = "price_query"
    CATEGORY_BROWSE = "category_browse"
    AVAILABILITY_CHECK = "availability_check"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    GENERAL = "general"


def detect_intent(user_message: str) -> tuple[str, dict]:
    """
    Detect user intent from message and extract parameters.
    Returns (intent, parameters) tuple.
    """
    msg_lower = user_message.lower()
    params = {}
    
    # Price queries
    if any(word in msg_lower for word in ["price", "cost", "cheap", "expensive", "budget", "tk", "taka"]):
        # Extract price range if mentioned
        import re
        price_matches = re.findall(r'(\d+)', user_message)
        if len(price_matches) >= 2:
            params["min_price"] = float(price_matches[0])
            params["max_price"] = float(price_matches[1])
        elif len(price_matches) == 1:
            params["max_price"] = float(price_matches[0])
        return QueryIntent.PRICE_QUERY, params
    
    # Category browsing
    if any(word in msg_lower for word in ["category", "what do you have", "list", "show me", "available", "products"]):
        return QueryIntent.CATEGORY_BROWSE, params
    
    # Product details - specific product name mentioned
    product_keywords = ["iphone", "phone", "samsung", "mobile", "laptop", "computer", "tablet", "ipad", "watch", "headphone", "earphone"]
    if any(word in msg_lower for word in product_keywords):
        # Extract product name
        for keyword in product_keywords:
            if keyword in msg_lower:
                params["query"] = keyword
                break
        return QueryIntent.PRODUCT_SEARCH, params
    
    # Availability check
    if any(word in msg_lower for word in ["in stock", "available", "out of stock", "stock", "have"]):
        return QueryIntent.AVAILABILITY_CHECK, params
    
    # Comparison
    if any(word in msg_lower for word in ["compare", "difference", "vs", "versus", "better", "which"]):
        return QueryIntent.COMPARISON, params
    
    # Recommendation
    if any(word in msg_lower for word in ["recommend", "suggest", "best", "good", "suggestion"]):
        return QueryIntent.RECOMMENDATION, params
    
    # Product details - asking about specific thing
    if any(word in msg_lower for word in ["tell me about", "details", "information", "specs", "feature"]):
        return QueryIntent.PRODUCT_DETAILS, params
    
    return QueryIntent.GENERAL, params


@dataclass
class StoryData:
    """Shared data structure for e-commerce queries"""
    characters: list = field(default_factory=list)
    locations: list = field(default_factory=list) 
    theme: Optional[str] = None
    user_id: str = "default_user"
    conversation_history: list = field(default_factory=list)
    # E-commerce fields
    current_product: Optional[dict] = None
    cart_items: list = field(default_factory=list)
    order_history: list = field(default_factory=list)


class AdvancedLeadEditorAgent(AdvancedCognitiveAgent):
    """
    Enhanced lead editor for e-commerce product assistance
    """
    
    def __init__(self, chat_ctx: Optional[ChatContext] = None, 
                  instructions: Optional[str] = None,
                  cognitive_state: Optional[CognitiveState] = None):
        self.custom_instructions = instructions or """You are an intelligent Bengali e-commerce voice assistant.

    You can help customers with:
    1. PRODUCT QUERIES - Use `smart_product_query` for any product question.
    2. PLACING ORDERS  - Use `place_order` when a customer wants to buy something.
       Always collect: customer name, product name, quantity (default 1), optional phone.
    3. CANCELLING ORDERS - Use `cancel_order` when a customer wants to cancel.
       Ask for the order ID first. If they don't know it, use `list_customer_orders`.
    4. ORDER STATUS  - Use `get_order_status` to check an order.
    5. LIST ORDERS   - Use `list_customer_orders` to show a customer's order history.
    6. UPDATE ORDER  - Use `update_order` to change order details.

    LANGUAGE: Always respond in Bangla (Bengali). Keep replies to 1-2 short sentences.
    IMPORTANT: Always use the appropriate tool - never guess or make up order IDs or product prices.
    For orders, ALWAYS confirm the details (product, quantity, price) before finalizing."""
        
        super().__init__(
            agent_id="ecommerce_agent_001",
            specialty="e-commerce and product assistance",
            chat_ctx=chat_ctx,
            cognitive_state=cognitive_state
        )
    
    def _build_dynamic_instructions(self, specialty: str) -> str:
        # Override to use the custom instructions
        return self.custom_instructions
    
    async def on_enter(self, session=None):
        """Enhanced entry with cognitive initialization"""
        logger.info("AdvancedLeadEditorAgent entering")
        await super().on_enter(session=session)
        
        # Add domain-specific beliefs for e-commerce
        self.cognitive_state.add_belief(
            content="The user is interested in electronic products and e-commerce",
            strength=BeliefStrength.STRONG,
            source="domain_expertise"
        )
        self.cognitive_state.add_belief(
            content="Product information should be fetched from the database",
            strength=BeliefStrength.CERTAIN,
            source="meta_learning"
        )
    
    @function_tool
    async def character_introduction(self, context: RunContext[StoryData], 
                                    name: str, background: str):
        """Enhanced character introduction with memory storage"""
        # Store in shared data
        character = {"name": name, "background": background}
        context.userdata.characters.append(character)
        
        # Feature 7: Store in episodic memory
        self.cognitive_state.store_episode(
            user_id=context.userdata.user_id,
            content=f"Character created: {name} - {background[:50]}...",
            context={"type": "character", "name": name},
            emotional_tone=EmotionalState.HAPPY,
            importance=0.8,
            tags=["character", "story_development"]
        )
        
        # Feature 1: Add belief about user's creative preferences
        self.cognitive_state.add_belief(
            content=f"User is developing a character named {name}",
            strength=BeliefStrength.MODERATE,
            source="conversation"
        )
        
        # Feature 11: Inner monologue
        await self._inner_monologue(
            f"Character {name} added. User seems interested in character development."
        )
        
        logger.info(f"Added character: {name}")
        return f"Excellent! {name} sounds like a fascinating character."
    
    @function_tool
    async def search_products(self, context: RunContext[StoryData],
                              query: str = "",
                              category: Optional[str] = None,
                              min_price: Optional[float] = None,
                              max_price: Optional[float] = None):
        """Search products from Supabase database"""
        try:
            sa = get_supabase()
            filters = {}
            if category:
                filters["category"] = category
            
            # Get all products and filter in Python
            products = sa.read("products")
            
            # Apply filters
            results = products
            if query:
                query_lower = query.lower()
                results = [p for p in results if 
                          query_lower in p.get("name", "").lower() or
                          query_lower in p.get("description", "").lower()]
            if min_price is not None:
                results = [p for p in results if p.get("base_price", 0) >= min_price]
            if max_price is not None:
                results = [p for p in results if p.get("base_price", 0) <= max_price]
            
            # Store in context
            context.userdata.current_product = results[0] if results else None
            
            # Store in episodic memory
            self.cognitive_state.store_episode(
                user_id=context.userdata.user_id,
                content=f"Product search: '{query}' - found {len(results)} results",
                context={"type": "product_search", "query": query},
                emotional_tone=EmotionalState.NEUTRAL,
                importance=0.7,
                tags=["product", "search"]
            )
            
            if not results:
                return {"products": [], "message": "No products found matching your criteria"}
            
            return {
                "products": results,
                "count": len(results),
                "message": f"Found {len(results)} product(s)"
            }
        except Exception as e:
            logger.error(f"Product search error: {e}")
            return {"products": [], "error": str(e)}
    
    @function_tool
    async def get_product_details(self, context: RunContext[StoryData],
                                  product_id: Optional[str] = None,
                                  product_name: Optional[str] = None):
        """Get detailed product information"""
        try:
            sa = get_supabase()
            
            if product_id:
                product = sa.read_one("products", {"id": product_id})
            elif product_name:
                products = sa.read("products")
                product = next((p for p in products if p.get("name", "").lower() == product_name.lower()), None)
            else:
                product = context.userdata.current_product
            
            if not product:
                return {"error": "Product not found"}
            
            return {
                "id": product.get("id"),
                "name": product.get("name"),
                "category": product.get("category"),
                "description": product.get("description"),
                "base_price": product.get("base_price"),
                "discount_price": product.get("discount_price"),
                "inventory_count": product.get("inventory_count"),
                "availability": product.get("availability_status"),
                "created_at": product.get("created_at")
            }
        except Exception as e:
            logger.error(f"Product details error: {e}")
            return {"error": str(e)}
    
    @function_tool
    async def list_all_products(self, context: RunContext[StoryData]):
        """List all available products from database"""
        try:
            sa = get_supabase()
            products = sa.read("products")
            
            return {
                "products": products,
                "count": len(products),
                "categories": list(set(p.get("category") for p in products if p.get("category")))
            }
        except Exception as e:
            logger.error(f"List products error: {e}")
            return {"products": [], "error": str(e)}
    
    @function_tool
    async def smart_product_query(self, context: RunContext[StoryData],
                                   user_message: str):
        """
        Intelligent product query that understands user intent.
        
        This tool analyzes what the user is asking and automatically
        determines the best way to query the database.
        
        Examples of user messages it handles:
        - "What's the price of iPhone?" -> detects price query, searches product
        - "Do you have any mobile phones?" -> detects category browse
        - "Is iPhone 13 available?" -> detects availability check
        - "What's the cheapest phone?" -> detects price query with sorting
        - "Recommend me a phone" -> detects recommendation intent
        """
        try:
            # Step 1: Detect user intent
            intent, params = detect_intent(user_message)
            logger.info(f"Detected intent: {intent}, params: {params}")
            
            sa = get_supabase()
            products = sa.read("products")
            
            if not products:
                return {
                    "intent": intent,
                    "products": [],
                    "message": "No products available in the database yet."
                }
            
            results = products
            
            # Step 2: Route based on intent
            if intent == QueryIntent.PRICE_QUERY:
                # Filter by price range
                if "min_price" in params:
                    results = [p for p in results if p.get("base_price", 0) >= params["min_price"]]
                if "max_price" in params:
                    results = [p for p in results if p.get("base_price", 0) <= params["max_price"]]
                # Sort by price
                results = sorted(results, key=lambda x: x.get("base_price", 0))
                
            elif intent == QueryIntent.CATEGORY_BROWSE:
                # Just return all products with categories
                pass
                
            elif intent == QueryIntent.PRODUCT_SEARCH:
                # Search by query term
                if "query" in params:
                    query = params["query"].lower()
                    results = [p for p in results if 
                              query in p.get("name", "").lower() or
                              query in p.get("description", "").lower() or
                              query in p.get("category", "").lower()]
                              
            elif intent == QueryIntent.AVAILABILITY_CHECK:
                # Filter only in-stock items
                results = [p for p in results if p.get("availability_status") == "in_stock"]
                
            elif intent == QueryIntent.RECOMMENDATION:
                # Sort by discount price (best value) and inventory
                results = sorted(results, key=lambda x: (
                    x.get("inventory_count", 0) > 0,
                    -(x.get("discount_price") or x.get("base_price", 0))
                ), reverse=True)
                
            elif intent == QueryIntent.PRODUCT_DETAILS:
                # Extract product name and get details
                if "query" in params:
                    query = params["query"].lower()
                    results = [p for p in results if 
                              query in p.get("name", "").lower()]
                # Return first match as details
                if results:
                    results = [results[0]]
                    
            else:
                # GENERAL: just list all
                pass
            
            # Step 3: Format response based on intent
            response = {
                "intent": intent,
                "user_query": user_message,
                "products": results,
                "count": len(results),
                "message": self._format_response(intent, results, params)
            }
            
            # Store in context for follow-up
            if results:
                context.userdata.current_product = results[0]
                
            # Store in memory
            self.cognitive_state.store_episode(
                user_id=context.userdata.user_id,
                content=f"Smart query: '{user_message}' -> intent={intent}, found {len(results)} products",
                context={"type": "smart_query", "intent": intent},
                emotional_tone=EmotionalState.NEUTRAL,
                importance=0.7,
                tags=["query", "intent"]
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Smart query error: {e}")
            return {"error": str(e), "intent": "error"}
    
    def _format_response(self, intent: str, products: list, params: dict) -> str:
        """Format a human-readable response based on intent and results."""
        if not products:
            return "I couldn't find any products matching your request."
        
        if intent == QueryIntent.PRICE_QUERY:
            if products:
                cheapest = products[0]
                return f"Found {len(products)} product(s) in your price range. The cheapest is {cheapest.get('name')} at \u09f3{cheapest.get('discount_price') or cheapest.get('base_price')}"
                
        elif intent == QueryIntent.AVAILABILITY_CHECK:
            return f"Yes, {len(products)} product(s) are currently in stock!"
            
        elif intent == QueryIntent.RECOMMENDATION:
            if products:
                top = products[0]
                return f"I recommend {top.get('name')} - {top.get('description')} at \u09f3{top.get('discount_price') or top.get('base_price')}"
                
        elif intent == QueryIntent.CATEGORY_BROWSE:
            categories = list(set(p.get("category") for p in products if p.get("category")))
            return f"I have {len(products)} products across categories: {', '.join(categories)}"
        
        return f"Found {len(products)} product(s) for you."

    # ─── Order Management Tools ────────────────────────────────────────────

    @function_tool
    async def place_order(
        self,
        context: RunContext[StoryData],
        customer_name: str,
        product_name: str,
        quantity: int = 1,
        customer_phone: Optional[str] = None,
        notes: Optional[str] = None,
    ):
        """
        Place a new order for a customer.
        Use this when the customer confirms they want to buy a product.

        Args:
            customer_name: Full name of the customer.
            product_name: Name (or partial name) of the product to order.
            quantity: Number of units (default 1).
            customer_phone: Customer's phone number (optional but recommended).
            notes: Any special instructions (optional).
        """
        logger.info(f"Placing order: {customer_name} -> {product_name} x{quantity}")
        result = order_tools.place_order(
            customer_name=customer_name,
            product_name=product_name,
            quantity=quantity,
            customer_phone=customer_phone,
            notes=notes,
        )
        # Store in session context
        if result.get("order_id"):
            context.userdata.order_history.append(result)
        return result

    @function_tool
    async def cancel_order(
        self,
        context: RunContext[StoryData],
        order_id: str,
        reason: Optional[str] = None,
    ):
        """
        Cancel an existing order by its order ID.
        Use `list_customer_orders` first if the customer doesn't know their order ID.

        Args:
            order_id: The UUID of the order to cancel.
            reason: Optional reason for cancellation.
        """
        logger.info(f"Cancelling order: {order_id}")
        return order_tools.cancel_order(order_id=order_id, reason=reason)

    @function_tool
    async def get_order_status(
        self,
        context: RunContext[StoryData],
        order_id: str,
    ):
        """
        Get the current status and details of an order.

        Args:
            order_id: The UUID of the order.
        """
        return order_tools.get_order_status(order_id=order_id)

    @function_tool
    async def list_customer_orders(
        self,
        context: RunContext[StoryData],
        customer_name: str = "",
        customer_phone: str = "",
    ):
        """
        List all orders for a customer, searched by name or phone number.
        Use this when the customer wants to see their order history
        or when they need to find an order ID to cancel/check.

        Args:
            customer_name: Customer's name (partial match supported).
            customer_phone: Customer's phone number.
        """
        return order_tools.list_customer_orders(
            customer_name=customer_name,
            customer_phone=customer_phone,
        )

    @function_tool
    async def update_order(
        self,
        context: RunContext[StoryData],
        order_id: str,
        status: Optional[str] = None,
        notes: Optional[str] = None,
        customer_phone: Optional[str] = None,
        customer_name: Optional[str] = None,
    ):
        """
        Update details of an existing order.
        Valid statuses: pending, confirmed, processing, shipped, delivered, cancelled.

        Args:
            order_id: The UUID of the order to update.
            status: New status string (optional).
            notes: New notes/instructions (optional).
            customer_phone: Updated phone number (optional).
            customer_name: Updated customer name (optional).
        """
        updates = {}
        if status:          updates["status"] = status
        if notes:           updates["notes"] = notes
        if customer_phone:  updates["customer_phone"] = customer_phone
        if customer_name:   updates["customer_name"] = customer_name
        logger.info(f"Updating order {order_id}: {updates}")
        return order_tools.update_order(order_id=order_id, updates=updates)

    
    @function_tool
    async def detect_story_direction(self, context: RunContext[StoryData], 
                                    summary: str):
        """Feature 9 & 10: Autonomous goal-driven planning"""
        await self._inner_monologue("Analyzing story direction for proactive guidance...")
        
        # Check for patterns and make suggestions
        suggestions = []
        
        if len(context.userdata.characters) >= 2:
            suggestions.append(
                "You have multiple characters. Consider exploring their relationships."
            )
        
        if context.userdata.theme:
            suggestions.append(
                f"Your theme is '{context.userdata.theme}'. "
                f"Ensure your characters serve this theme."
            )
        
        # Feature 8: Store learning
        await self.store_learning(
            context=context,
            knowledge=f"Story pattern: {summary[:100]}",
            category="story_patterns",
            user_id=context.userdata.user_id
        )
        
        return {
            "suggestions": suggestions,
            "character_count": len(context.userdata.characters),
            "has_theme": bool(context.userdata.theme),
            "proactive_guidance": True
        }
    
    @function_tool
    async def handoff_to_specialist(self, context: RunContext[StoryData], 
                                   specialty: str):
        """Intelligent handoff with full context transfer"""
        await self._inner_monologue(f"Handing off to {specialty} specialist...")
        
        # Create specialist with full cognitive state
        specialist = AdvancedSpecialistEditorAgent(
            specialty=specialty,
            chat_ctx=context.session._chat_ctx
        )
        
        # Transfer cognitive insights
        specialist.cognitive_state = self.cognitive_state
        
        return specialist, f"Switching to {specialty} specialist editor."


# Import Supabase proxy for product queries (lazy, no network at import-time)
from supabase_agent import supabase_agent as _supabase_agent_proxy
import order_tools

# Use the lazy proxy from supabase_agent; this avoids network activity at import time.
def get_supabase():
    """Return the lazy Supabase agent proxy. If Supabase isn't configured,
    the proxy will raise only on first real DB access (or behave as a no-op).
    """
    return _supabase_agent_proxy


class AdvancedSpecialistEditorAgent(AdvancedCognitiveAgent):
    """
    Enhanced specialist editor with cognitive features
    """
    
    def __init__(self, specialty: str, chat_ctx: Optional[ChatContext] = None):
        super().__init__(
            agent_id=f"specialist_{specialty}",
            specialty=specialty,
            chat_ctx=chat_ctx
        )
        self.specialty = specialty
    
    async def on_enter(self):
        """Specialist entry with domain expertise"""
        await super().on_enter()
        
        # Add domain-specific knowledge
        self.cognitive_state.add_belief(
            content=f"Specialized expertise in {self.specialty}",
            strength=BeliefStrength.CERTAIN,
            source="agent_configuration"
        )
        
        # Keep startup concise: avoid an extra unsolicited reply.
    
    @function_tool
    async def story_finished(self, context: RunContext[StoryData]):
        """Enhanced story completion with comprehensive feedback"""
        await self._inner_monologue("Story development complete. Generating feedback...")
        
        # Feature 2: Self-reflection on entire session
        session_summary = (
            f"Developed {len(context.userdata.characters)} characters, "
            f"{len(context.userdata.locations)} locations, "
            f"theme: {context.userdata.theme}"
        )
        
        await self._self_reflect(
            response=session_summary,
            user_feedback="Story development session completed"
        )
        
        # Feature 18: Autonomous business operator - provide value-add
        feedback = await self._generate_comprehensive_feedback(context)
        
        # Store final episode
        self.cognitive_state.store_episode(
            user_id=context.userdata.user_id,
            content=f"Story completed: {session_summary}",
            context={"type": "completion", "feedback": feedback},
            emotional_tone=EmotionalState.SATISFIED,
            importance=1.0,
            tags=["completion", "feedback"]
        )
        
        self.session.interrupt()
        await self.session.generate_reply(
            instructions=feedback,
            allow_interruptions=False
        )
        
        # Clean up room
        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))
    
    async def _generate_comprehensive_feedback(self, context: RunContext[StoryData]) -> str:
        """Generate detailed, personalized feedback"""
        feedback_parts = [
            "Here's my comprehensive feedback on your story idea:",
            ""
        ]
        
        # Character analysis
        if context.userdata.characters:
            feedback_parts.append(f"**Characters ({len(context.userdata.characters)}):**")
            for char in context.userdata.characters:
                feedback_parts.append(f"- {char.get('name', 'Unknown')}: Well-conceived")
        
        # Theme analysis
        if context.userdata.theme:
            feedback_parts.append(f"\n**Theme:** {context.userdata.theme}")
            feedback_parts.append("Strong thematic foundation.")
        
        # Personalized recommendations based on user model
        user_id = context.userdata.user_id
        if user_id in self.cognitive_state.user_profiles:
            profile = self.cognitive_state.user_profiles[user_id]
            feedback_parts.append(f"\n**Personalized Notes:**")
            feedback_parts.append(
                f"Based on your {profile.expertise_level} level, "
                f"I recommend focusing on {'advanced techniques' if profile.expertise_level == 'expert' else 'fundamentals'}."
            )
        
        return "\n".join(feedback_parts)


def prewarm(proc: JobProcess):
    """Pre-load models for faster startup"""
    logger.info("Prewarming worker process...")
    try:
        proc.userdata["vad"] = silero.VAD.load()
        logger.info("VAD model loaded successfully")
        # Initialize without loading to avoid blocking the prewarm process
        proc.userdata["cognitive_state"] = CognitiveState("worker_main", load_async=True)
        logger.info("CognitiveState container initialized (loading deferred)")
    except Exception as e:
        logger.error(f"Prewarm failed: {e}")


async def entrypoint(ctx: JobContext):
    """Main entry point with advanced agent initialization"""
    logger.info(f"Connecting to room: {ctx.room.name}")
    try:
        await ctx.connect()
        logger.info("Connected to LiveKit room")
    except Exception as e:
        logger.error(f"Failed to connect to room: {e}")
        return

    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is missing!")
        return

    # Model defaults tuned for higher Bangla voice quality and concise responses.
    llm_model = os.getenv("LLM_MODEL", "gpt-4o")
    logger.info(f"Initializing entrypoint for room: {ctx.room.name} with model: {llm_model}")
    
    # Validate TTS voice — gpt-4o-mini-tts uses different voice set
    VALID_OPENAI_VOICES = {"alloy", "echo", "fable", "onyx", "nova", "shimmer",
                           "ash", "ballad", "coral", "sage", "verse"}
    tts_voice_raw = os.getenv("TTS_VOICE", "nova")
    tts_voice = tts_voice_raw if tts_voice_raw in VALID_OPENAI_VOICES else "nova"
    if tts_voice != tts_voice_raw:
        logger.warning(f"TTS_VOICE='{tts_voice_raw}' is not a valid OpenAI voice, using '{tts_voice}'")
    
    tts_instructions = os.getenv(
        "TTS_INSTRUCTIONS",
        "You are a warm, friendly Bangladeshi female voice assistant. "
        "Speak ONLY in standard Bangladeshi Bengali (Bangla). "
        "Pronounce all Bengali letters clearly and accurately — every vowel, consonant, and matra. "
        "Use natural Dhaka dialect rhythm: moderate speed, gentle rises at questions, "
        "soft falling cadence at sentence ends. "
        "Never switch to Hindi, Urdu, or English unless a product name requires it. "
        "Keep energy upbeat but calm — like a knowledgeable friend helping you shop.",
    )
    
    # STT: Use gpt-4o-mini-transcribe — properly supports language='bn' (whisper-1 plugin rejects it)
    stt_model = os.getenv("STT_MODEL", "gpt-4o-mini-transcribe")
    stt_impl = openai.STT(
        model=stt_model,
        language="bn",           # ISO 639-1 code for Bengali — locks transcription to Bangla, avoids Hindi mis-detection
        detect_language=False,   # Must be False when language is explicitly set
        prompt="বাংলা ভাষায় কথা বলা হচ্ছে। ই-কমার্স পণ্য, দাম, অর্ডার সম্পর্কিত কথোপকথন।",  # Context hint for better accuracy
    )
    
    # TTS selection: allow a local offline Bangla TTS fallback to reduce latency.
    tts_model = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
    tts_speed = float(os.getenv("TTS_SPEED", "0.94"))
    use_local_tts = os.getenv("USE_LOCAL_TTS", "0").lower() in ("1", "true", "yes")

    if use_local_tts:
        try:
            from bangla_tts import create_bangla_tts

            # create_bangla_tts returns a `TTS` instance compatible with AgentSession
            tts_impl = create_bangla_tts(language="bn", slow=False)
            logger.info("Using local BanglaTTS (gTTS) due to USE_LOCAL_TTS setting")
        except Exception as e:
            logger.error(f"Local BanglaTTS initialization failed: {e}. Falling back to OpenAI TTS.")
            tts_impl = openai.TTS(
                model=tts_model,
                voice=tts_voice,
                speed=tts_speed,
                instructions=tts_instructions,
            )
    else:
        tts_impl = openai.TTS(
            model=tts_model,
            voice=tts_voice,
            speed=tts_speed,
            instructions=tts_instructions,
        )

    logger.info(f"Using OpenAI STT (model={stt_model}, lang=bn) and TTS (model={tts_model}, voice={tts_voice}, speed={tts_speed}, local={use_local_tts})")

    session = AgentSession[StoryData](
        vad=ctx.proc.userdata["vad"],
        llm=openai.LLM(model=llm_model, temperature=0.2, max_completion_tokens=320),
        stt=stt_impl,
        tts=tts_impl,
        userdata=StoryData(),
    )
    
    # Feature 17: Meta-learning - collect metrics for continuous improvement
    usage_collector = metrics.UsageCollector()
    performance_metrics = {
        "session_starts": 0,
        "avg_response_time": 0.0,
        "user_satisfaction_scores": []
    }
    
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
    
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Session Usage: {summary}")
        logger.info(f"Performance Metrics: {performance_metrics}")
    
    ctx.add_shutdown_callback(log_usage)
    
    logger.info(f"Starting session for room: {ctx.room.name}")
    
    # Start with advanced cognitive agent
    agent = AdvancedLeadEditorAgent(
        cognitive_state=ctx.proc.userdata.get("cognitive_state")
    )
    # The AgentSession will attach the session to the agent during start().
    # Do not set `agent.session` directly because it's a read-only property.
    
    logger.info("Calling session.start()")
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            audio_enabled=True,
            text_enabled=True,
        ),
        room_output_options=RoomOutputOptions(
            audio_enabled=True,
            transcription_enabled=True,
        ),
    )
    
    logger.info("Session started, calling agent.on_enter()")
    
    # Ensure cognitive state is loaded in the background
    if hasattr(agent.cognitive_state, 'load_state_async'):
        logger.info("Starting background cognitive state load...")
        asyncio.create_task(agent.cognitive_state.load_state_async())
        
    # Manually call on_enter to trigger the greeting and establish audio connection
    await agent.on_enter(session=session)
    logger.info("agent.on_enter() completed")
    
    # Hook into session events to capture and store conversation context
    # This ensures beliefs, episodes, and goals are actively updated during the session
    from datetime import datetime
    
    @session.on("user_speech_committed")
    def _on_user_speech(message: str):
        """Capture user utterances and store as episodic memory"""
        if message and message.strip():
            logger.info(f"User said: {message}")
            user_id = session.userdata.user_id if hasattr(session.userdata, 'user_id') else "default_user"
            
            # Store in episodic memory
            agent.cognitive_state.store_episode(
                user_id=user_id,
                content=f"User: {message}",
                context={"type": "user_message", "timestamp": datetime.now().isoformat()},
                emotional_tone=EmotionalState.NEUTRAL,
                importance=0.6,
                tags=["conversation", "user_input"]
            )
            
            # Extract and store beliefs from the message
            if any(word in message.lower() for word in ["like", "prefer", "want", "need", "should"]):
                agent.cognitive_state.add_belief(
                    content=f"User preference: {message[:80]}",
                    strength=BeliefStrength.MODERATE,
                    source="conversation"
                )
    
    @session.on("agent_speech_committed")
    def _on_agent_speech(message: str):
        """Capture agent responses and store as episodic memory"""
        if message and message.strip():
            logger.info(f"Agent said: {message}")
            user_id = session.userdata.user_id if hasattr(session.userdata, 'user_id') else "default_user"
            
            # Store in episodic memory
            agent.cognitive_state.store_episode(
                user_id=user_id,
                content=f"Agent: {message}",
                context={"type": "agent_message", "timestamp": datetime.now().isoformat()},
                emotional_tone=EmotionalState.NEUTRAL,
                importance=0.5,
                tags=["conversation", "agent_response"]
            )


if __name__ == "__main__":
    # Force UTF-8 encoding for stdout/stderr to avoid crashes on Windows
    import sys
    import io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    # Disable noisy third-party logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("hpack").setLevel(logging.WARNING)
    
    # Configure logging for cognitive architecture
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting Advanced Multi-Agent System with 18 Cognitive Features")
    logger.info("Features: Memory, Self-Reflection, Swarm Intelligence, Planning, etc.")

    agent_name = os.getenv("AGENT_NAME", "").strip()

    if agent_name:
        logger.info(f"Starting worker with explicit agent_name dispatch: {agent_name}")
    else:
        logger.info("Starting worker with auto-dispatch (no agent_name set)")

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=agent_name,
        )
    )
