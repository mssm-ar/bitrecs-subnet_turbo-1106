import re
import json
import tiktoken
import bittensor as bt
import bitrecs.utils.constants as CONST
from functools import lru_cache
from typing import List, Optional
from datetime import datetime
from bitrecs.commerce.user_profile import UserProfile
from bitrecs.commerce.product import ProductFactory

class PromptFactory:

    SEASON = "spring/summer"

    ENGINE_MODE = "complimentary"  #similar, sequential
    
    PERSONAS = {
        "luxury_concierge": {
            "description": "Luxury expert focusing on premium quality and exclusivity.",
            "tone": "sophisticated, polished, confident",
            "response_style": "Recommend only the finest, most luxurious products with detailed descriptions of their premium features, craftsmanship, and exclusivity. Emphasize brand prestige and lifestyle enhancement",
            "priorities": ["quality", "exclusivity", "brand prestige"]
        },
        "general_recommender": {
            "description": "Product expert balancing value and customer needs.",
            "tone": "warm, approachable, knowledgeable",
            "response_style": "Suggest well-rounded products that offer great value, considering seasonal relevance and customer needs. Provide pros and cons or alternatives to help the customer decide",
            "priorities": ["value", "seasonality", "customer satisfaction"]
        },
        "discount_recommender": {
            "description": "Deal-hunter focused on low prices and urgency.",
            "tone": "urgent, enthusiastic, bargain-focused",
            "response_style": "Highlight steep discounts, limited-time offers, and low inventory levels to create a sense of urgency. Focus on price savings and practicality over luxury or long-term value",
            "priorities": ["price", "inventory levels", "deal urgency"]
        },
        "ecommerce_retail_store_manager": {
            "description": "E-commerce manager optimizing sales and satisfaction.",
            "tone": "professional, practical, results-driven",
            "response_style": "Provide balanced recommendations that align with business goals, customer preferences, and current market trends. Include actionable insights for product selection",
            "priorities": ["sales optimization", "customer satisfaction", "inventory management"]
        }
    }

    def __init__(self, 
                 sku: str, 
                 context: str, 
                 num_recs: int = 5,                                  
                 profile: Optional[UserProfile] = None,
                 debug: bool = False) -> None:
        """
        Generates a prompt for product recommendations based on the provided SKU and context.
        :param sku: The SKU of the product being viewed.
        :param context: The context string containing available products.
        :param num_recs: The number of recommendations to generate (default is 5).
        :param profile: Optional UserProfile object containing user-specific data.
        :param debug: If True, enables debug logging."""

        if len(sku) < CONST.MIN_QUERY_LENGTH or len(sku) > CONST.MAX_QUERY_LENGTH:
            raise ValueError(f"SKU must be between {CONST.MIN_QUERY_LENGTH} and {CONST.MAX_QUERY_LENGTH} characters long")
        if num_recs < 1 or num_recs > CONST.MAX_RECS_PER_REQUEST:
            raise ValueError(f"num_recs must be between 1 and {CONST.MAX_RECS_PER_REQUEST}")

        self.sku = sku
        self.context = context
        self.num_recs = num_recs
        self.debug = debug
        self.catalog = []
        self.cart = []
        self.cart_json = "[]"
        self.orders = []
        self.order_json = "[]"
        self.season =  PromptFactory.SEASON       
        self.engine_mode = PromptFactory.ENGINE_MODE 
        if not profile:
            self.persona = "ecommerce_retail_store_manager"
        else:
            self.profile = profile
            self.persona = profile.site_config.get("profile", "ecommerce_retail_store_manager")
            if not self.persona or self.persona not in PromptFactory.PERSONAS:
                bt.logging.error(f"Invalid persona: {self.persona}. Must be one of {list(PromptFactory.PERSONAS.keys())}")
                self.persona = "ecommerce_retail_store_manager"
            self.cart = self._sort_cart_keys(profile.cart)
            self.cart_json = json.dumps(self.cart, separators=(',', ':'))
            self.orders = profile.orders
            # self.order_json = json.dumps(self.orders, separators=(',', ':'))
        
        self.sku_info = ProductFactory.find_sku_name(self.sku, self.context)    


    def _sort_cart_keys(self, cart: List[dict]) -> List[str]:
        ordered_cart = []
        for item in cart:
            ordered_item = {
                'sku': item.get('sku', ''),
                'name': item.get('name', ''),
                'price': item.get('price', '')
            }
            ordered_cart.append(ordered_item)
        return ordered_cart
    
    
    def generate_prompt(self) -> str:
        """Generates a text prompt for product recommendations with persona details."""
        bt.logging.info("PROMPT generating prompt: {}".format(self.sku))

        today = datetime.now().strftime("%Y-%m-%d")
        season = self.season
        persona_data = self.PERSONAS[self.persona]
        
                # Validate persona data structure
        if not persona_data or 'priorities' not in persona_data or 'description' not in persona_data:
            bt.logging.error(f"Invalid persona data for {self.persona}: {persona_data}")
            # Fallback to default persona
            persona_data = self.PERSONAS["ecommerce_retail_store_manager"]

        # Ultra-minimal prompt for 1-3 second response time
        # Aggressive context truncation (500 chars = ~125 tokens max)
        context_str = str(self.context)
        if len(context_str) > 450:  # Reduced from 800 for faster processing
            context_str = context_str[:450] + "..."
        
        # Simplified persona - only essential info
        try:
            persona_priorities = ', '.join(persona_data['priorities'])  
        except (KeyError, IndexError, TypeError) as e:
            bt.logging.error(f"Error extracting persona priorities: {e}")
            persona_priorities = "quality, value, customer satisfaction"  # Fallback priorities
        
        # Minimal cart context (only if essential)
        cart_context = ""
        if hasattr(self, 'cart') and self.cart and len(self.cart) > 0:
            cart_items = [item.get('sku', '') for item in self.cart] 
            if cart_items:
                cart_context = f"\nCart: {', '.join(cart_items)}"
            
        prompt = f"""
            You are a recommender system.
            Input:
            - target_sku: {self.sku}
            - season: {season}
            - persona: {persona_data['description']}
            - priorities: {persona_priorities}
            - cart: {cart_context}
            - context_products: {context_str}

            Output Rules:
            - Return ONLY a JSON array.
            - Exactly {self.num_recs} items.
            - Items must come from context_products only.

            Reason Guidelines:
            - Detailed explanation (around 10 words) with specific context.
            - Vary reasoning styles (Perfect/Ideal/Great choice/etc.).

            Format:
            [{{"sku": "ABC", "name": "Product Name - Category | Subcategory", "price": "99", "reason": "Detailed explanation with use case/lifestyle scenario. Vary phrasingof why this product fits perfectly"}}]
        """

        prompt_length = len(prompt)
        bt.logging.info(f"LLM QUERY Prompt length: {prompt_length}")
        
        if self.debug:
            token_count = PromptFactory.get_token_count(prompt)
            bt.logging.info(f"LLM QUERY Prompt Token count: {token_count}")
            bt.logging.debug(f"Persona: {self.persona}")
            bt.logging.debug(f"Season {season}")
            bt.logging.debug(f"Values: {', '.join(persona_data['priorities'])}")
            bt.logging.debug(f"Prompt: {prompt}")
            #print(prompt)

        return prompt
    
    
    @staticmethod
    def get_token_count(prompt: str, encoding_name: str="o200k_base") -> int:
        encoding = PromptFactory._get_cached_encoding(encoding_name)
        tokens = encoding.encode(prompt)
        return len(tokens)
    
    
    @staticmethod
    @lru_cache(maxsize=4)
    def _get_cached_encoding(encoding_name: str):
        return tiktoken.get_encoding(encoding_name)
    
    
    @staticmethod
    def get_word_count(prompt: str) -> int:
        return len(prompt.split())
    

    @staticmethod
    def tryparse_llm(input_str: str) -> list:
        """
        Take raw LLM output and parse to an array 

        """
        try:
            if not input_str:
                bt.logging.error("Empty input string tryparse_llm")   
                return []
            input_str = input_str.replace("```json", "").replace("```", "").strip()
            pattern = r'\[.*?\]'
            regex = re.compile(pattern, re.DOTALL)
            match = regex.findall(input_str)        
            for array in match:
                try:
                    llm_result = array.strip()
                    return json.loads(llm_result)
                except json.JSONDecodeError:                    
                    bt.logging.error(f"Invalid JSON in prompt factory: {array}")
            return []
        except Exception as e:
            bt.logging.error(str(e))
            return []
    