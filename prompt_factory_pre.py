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
from bitrecs.llms.preselect import pre_select_context

class PromptFactory:

    # Response cache for similar queries (5 minute TTL)
    # _response_cache: Dict[str, List] = {}
    # _cache_timestamps: Dict[str, float] = {}
    # CACHE_TTL = 300  # 5 minutes
    
    # Context file cache (static - loaded once)
    _context_cache: Optional[str] = None
    _context_loaded: bool = False
    
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

        # Load context.json file with caching (load once, use many times)
        if not PromptFactory._context_loaded:
            try:
                import os
                # Get the project root directory (3 levels up from this file)
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                context_file_path = os.path.join(project_root, 'context.json')
                with open(context_file_path, 'r') as f:
                    PromptFactory._context_cache = f.read()
                PromptFactory._context_loaded = True
                bt.logging.info(f"📁 CACHED context.json from {context_file_path} with {len(PromptFactory._context_cache)} characters")
            except Exception as e:
                bt.logging.warning(f"Failed to load context.json: {e}, using synapse context")
                PromptFactory._context_cache = str(self.context)
                PromptFactory._context_loaded = True
        
        full_context = PromptFactory._context_cache
        
        # Pre-select exactly num_recs products for faster processing
        context_str = pre_select_context(self.sku, full_context, max_products=self.num_recs, num_recs=self.num_recs)
        
        # Parse preselected products to extract relevant info
        try:
            preselected_products = json.loads(context_str)
            product_names = [p.get('name', '') for p in preselected_products[:self.num_recs]]
            product_skus = [p.get('sku', '') for p in preselected_products[:self.num_recs]]
        except:
            preselected_products = []
            product_names = []
            product_skus = []
        
        # Optimized prompt for description generation only
        prompt = f"""
            You are a product description generator.
            
            Task: Generate descriptions for {self.num_recs} preselected products.
            
            Target Product: {self.sku}
            Season: {season}
            Persona: {persona_data['description']}
            
            Preselected Products: {json.dumps(preselected_products[:self.num_recs], separators=(',', ':'))}
            
            Requirements:
            - Return ONLY a JSON array with exactly {self.num_recs} items
            - Each item must include: sku, name, price, reason
            - Reason should be 8-12 words describing why this product is suitable
            - Use varied reasoning styles: "Perfect for", "Ideal choice", "Great option", "Excellent fit", ....
            - Focus on seasonal relevance and persona priorities
            
            Output Format:
            [{{"sku": "ABC", "name": "Product Name", "price": "99", "reason": "Perfect seasonal choice for active lifestyle"}}]
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
    